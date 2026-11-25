"""Etapa 06 del Job de ML: entrena, compara y registra el clasificador de retraso >= 15 min."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # genera gráficos sin abrir ventanas (script y Databricks)
import matplotlib.pyplot as plt

import mlflow
import mlflow.spark
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ml.features import (
    add_weight_column,
    build_preprocessing_stages,
    build_tree_preprocessing_stages,
    compute_class_weights,
    feature_columns,
    get_splits,
    load_ml_config,
    prepare_feature_types,
    read_training_data,
    validate_feature_columns,
)
from src.ml.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

PROB_COL = "p_delay"  # probabilidad de la clase 1 (demorado)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05, 0.10, ..., 0.95

# ---------------------------------------------------------------------------
# Modelos candidatos
# ---------------------------------------------------------------------------
def _build_classifier(name: str, params: Dict[str, Any], ml_cfg: Dict[str, Any]):
    label = ml_cfg["data"]["label_col"]
    common = {"featuresCol": "features", "labelCol": label}
    if ml_cfg["training"]["use_class_weights"]:
        common["weightCol"] = "class_weight"

    if name == "logistic_regression":
        return LogisticRegression(**common, **params)
    if name == "gbt":
        return GBTClassifier(**common, seed=int(ml_cfg["training"]["seed"]), **params)
    raise ValueError(f"Modelo no soportado en ml.yaml: {name}")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def _score(model: PipelineModel, df: DataFrame) -> DataFrame:
    """Aplica el pipeline y extrae la probabilidad de la clase 1."""
    return model.transform(df).withColumn(PROB_COL, vector_to_array("probability")[1])


def _auc_metrics(scored: DataFrame, label: str) -> Dict[str, float]:
    """PR-AUC y ROC-AUC: no dependen del umbral."""
    ev = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="probability")
    return {
        "areaUnderPR": ev.setMetricName("areaUnderPR").evaluate(scored),
        "areaUnderROC": ev.setMetricName("areaUnderROC").evaluate(scored),
    }


def _threshold_metrics(scored: DataFrame, label: str, thresholds: List[float]) -> List[Dict[str, float]]:
    """Matriz de confusión para cada umbral en UNA sola pasada sobre los datos."""
    y = F.col(label) == 1.0
    aggs = []
    for i, t in enumerate(thresholds):
        pred = F.col(PROB_COL) >= t
        aggs += [
            F.sum((pred & y).cast("long")).alias(f"tp_{i}"),
            F.sum((pred & ~y).cast("long")).alias(f"fp_{i}"),
            F.sum((~pred & y).cast("long")).alias(f"fn_{i}"),
            F.sum((~pred & ~y).cast("long")).alias(f"tn_{i}"),
        ]
    row = scored.agg(*aggs).first()

    results = []
    for i, t in enumerate(thresholds):
        tp, fp, fn, tn = (int(row[f"{k}_{i}"] or 0) for k in ("tp", "fp", "fn", "tn"))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        results.append({
            "threshold": t, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return results


# ---------------------------------------------------------------------------
# Importancia de variables
# ---------------------------------------------------------------------------
def _feature_importance(model: PipelineModel, scored: DataFrame, ml_cfg: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Importancia por variable ORIGINAL (suma = 1).
    GBT: featureImportances de Spark. Regresión logística: |coeficiente| agregado por variable."""
    clf = model.stages[-1]

    # Nombre de cada posición del vector 'features' (lo deja el VectorAssembler en los metadatos)
    slot_names = {}
    for attrs in scored.schema["features"].metadata["ml_attr"]["attrs"].values():
        for a in attrs:
            slot_names[a["idx"]] = a["name"]

    if hasattr(clf, "featureImportances"):
        raw = clf.featureImportances.toArray()
    else:
        raw = [abs(c) for c in clf.coefficients.toArray()]

    feats = feature_columns(ml_cfg)
    totals = {f: 0.0 for f in feats}
    for idx, value in enumerate(raw):
        name = slot_names.get(idx, "")
        # name puede ser 'x_idx' (árboles), 'x_imp' (numéricas) o 'x_ohe_<valor>' (one-hot)
        for f in feats:
            if name in (f"{f}_idx", f"{f}_imp") or name.startswith(f"{f}_ohe"):
                totals[f] += float(value)
                break

    total = sum(totals.values()) or 1.0
    return sorted(((f, v / total) for f, v in totals.items()), key=lambda x: x[1], reverse=True)


def _log_importance_to_mlflow(importance: List[Tuple[str, float]], model_name: str) -> None:
    """Gráfico de barras + JSON en la corrida de MLflow del modelo."""
    feats = [f for f, _ in importance][::-1]
    vals = [v for _, v in importance][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feats, vals, color="#7FB3D5")
    ax.set_xlabel("Importancia relativa (suma = 1)")
    ax.set_title(f"Importancia de variables: {model_name}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    mlflow.log_figure(fig, f"feature_importance_{model_name}.png")
    mlflow.log_dict(dict(importance), f"feature_importance_{model_name}.json")
    plt.close(fig)


def _save_feature_importance(spark: SparkSession, config: Dict[str, Any], ml_cfg: Dict[str, Any], rows: List[tuple]) -> None:
    schema = ("run_id string, trained_at string, model_type string, feature string, "
              "importance double, rank int, is_champion boolean")
    df = spark.createDataFrame(rows, schema=schema)
    table = ml_cfg["outputs"]["feature_importance_table"]

    if config["environment"] == "local":
        path = Path(config["_repo_root"]) / ml_cfg["paths"]["local_output_dir"] / table
        df.write.format("delta").mode("append").save(str(path))
        logger.info("Importancia de variables guardada en %s", path)
    else:
        reg = ml_cfg["registry"]
        full_name = f"{reg['catalog']}.{reg['schema']}.{table}"
        df.write.format("delta").mode("append").saveAsTable(full_name)
        logger.info("Importancia de variables guardada en %s", full_name)


# ---------------------------------------------------------------------------
# Registro en Unity Catalog (solo Databricks)
# ---------------------------------------------------------------------------
def _register_in_uc(model_uri: str, ml_cfg: Dict[str, Any], model_type: str, threshold: float) -> str:
    reg = ml_cfg["registry"]
    full_name = f"{reg['catalog']}.{reg['schema']}.{reg['model_name']}"
    mv = mlflow.register_model(model_uri, full_name)

    client = MlflowClient()
    client.set_model_version_tag(full_name, mv.version, "model_type", model_type)
    client.set_model_version_tag(full_name, mv.version, "decision_threshold", str(threshold))
    client.set_registered_model_alias(full_name, reg["alias"], mv.version)
    logger.info("Modelo registrado: %s v%s con alias @%s", full_name, mv.version, reg["alias"])
    return mv.version


# ---------------------------------------------------------------------------
# Tabla de métricas (Delta, append: queda el historial de entrenamientos)
# ---------------------------------------------------------------------------
def _save_metrics_table(spark: SparkSession, config: Dict[str, Any], ml_cfg: Dict[str, Any], rows: List[tuple]) -> None:
    schema = ("run_id string, trained_at string, model_type string, split string, "
              "metric string, value double, is_champion boolean")
    df = spark.createDataFrame(rows, schema=schema)
    table = ml_cfg["outputs"]["metrics_table"]

    if config["environment"] == "local":
        path = Path(config["_repo_root"]) / ml_cfg["paths"]["local_output_dir"] / table
        df.write.format("delta").mode("append").save(str(path))
        logger.info("Métricas guardadas en %s", path)
    else:
        reg = ml_cfg["registry"]
        full_name = f"{reg['catalog']}.{reg['schema']}.{table}"
        df.write.format("delta").mode("append").saveAsTable(full_name)
        logger.info("Métricas guardadas en %s", full_name)


# ---------------------------------------------------------------------------
# Entrada principal (la llama _job_runner.py)
# ---------------------------------------------------------------------------
def run_train_model(spark: SparkSession, config: Dict[str, Any]) -> None:
    ml_cfg = load_ml_config(config)
    is_local = setup_mlflow(config, ml_cfg)
    label = ml_cfg["data"]["label_col"]
    trained_at = datetime.now(timezone.utc).isoformat()

    # 1. Datos: contrato, tipos y splits temporales
    df = read_training_data(spark, config, ml_cfg)
    validate_feature_columns(df, ml_cfg)
    df = prepare_feature_types(df, ml_cfg)
    train, val, test = get_splits(df, ml_cfg)

    # 2. Pesos por clase calculados SOLO con train
    if ml_cfg["training"]["use_class_weights"]:
        weights = compute_class_weights(train, ml_cfg)
        train = add_weight_column(train, weights, ml_cfg)

    # cache() acelera en local; serverless no lo soporta
    if is_local:
        train, val = train.cache(), val.cache()

    metric_rows: List[tuple] = []
    importance_rows: List[tuple] = []
    candidates = []

    with mlflow.start_run(run_name=f"train_{trained_at[:19]}") as parent_run:
        run_id = parent_run.info.run_id
        mlflow.log_dict(ml_cfg, "ml_config.yaml")
        mlflow.log_params({
            "sample_fraction": ml_cfg["training"]["sample_fraction"],
            "seed": ml_cfg["training"]["seed"],
            "use_class_weights": ml_cfg["training"]["use_class_weights"],
            "n_features_declared": len(feature_columns(ml_cfg)),
            "selection_metric": ml_cfg["training"]["selection_metric"],
        })

        # 3. Entrenar cada candidato y evaluarlo en VALIDATION
        for name, params in ml_cfg["training"]["models"].items():
            with mlflow.start_run(run_name=name, nested=True):
                logger.info("Entrenando %s con %s ...", name, params)
                mlflow.log_params(params)

                prep = build_tree_preprocessing_stages(ml_cfg) if name == "gbt" else build_preprocessing_stages(ml_cfg)
                pipeline = Pipeline(stages=prep + [_build_classifier(name, params, ml_cfg)])
                model = pipeline.fit(train)

                scored_val = _score(model, val)
                auc = _auc_metrics(scored_val, label)
                best_t = max(_threshold_metrics(scored_val, label, THRESHOLDS), key=lambda m: m["f1"])
                val_pos_rate = scored_val.agg(F.avg(label)).first()[0]

                mlflow.log_metrics({
                    "val_pr_auc": auc["areaUnderPR"],
                    "val_roc_auc": auc["areaUnderROC"],
                    "val_best_f1": best_t["f1"],
                    "val_best_threshold": best_t["threshold"],
                    "val_precision": best_t["precision"],
                    "val_recall": best_t["recall"],
                    "val_baseline_pr_auc": val_pos_rate,
                })
                logger.info("%s en validation: PR-AUC=%.4f (azar=%.4f) ROC-AUC=%.4f F1=%.4f (umbral %.2f)",
                            name, auc["areaUnderPR"], val_pos_rate, auc["areaUnderROC"], best_t["f1"], best_t["threshold"])

                importance = _feature_importance(model, scored_val, ml_cfg)
                _log_importance_to_mlflow(importance, name)
                logger.info("%s: top 5 variables -> %s", name,
                            ", ".join(f"{f} ({v:.1%})" for f, v in importance[:5]))
                for rank, (feat, value) in enumerate(importance, start=1):
                    importance_rows.append((run_id, trained_at, name, feat, float(value), rank, False))

                candidates.append({"name": name, "model": model, "auc": auc, "best": best_t})
                for metric, value in [("pr_auc", auc["areaUnderPR"]), ("roc_auc", auc["areaUnderROC"]),
                                      ("f1", best_t["f1"]), ("precision", best_t["precision"]),
                                      ("recall", best_t["recall"]), ("threshold", best_t["threshold"])]:
                    metric_rows.append((run_id, trained_at, name, "validation", metric, float(value), False))

        # 4. Elegir el mejor por la métrica de selección (en validation)
        sel = ml_cfg["training"]["selection_metric"]
        champion = max(candidates, key=lambda c: c["auc"][sel])
        threshold = champion["best"]["threshold"]
        logger.info("Modelo elegido: %s (%s=%.4f), umbral %.2f",
                    champion["name"], sel, champion["auc"][sel], threshold)

        # 5. Evaluar UNA sola vez en TEST
        scored_test = _score(champion["model"], test)
        test_auc = _auc_metrics(scored_test, label)
        test_m = _threshold_metrics(scored_test, label, [threshold])[0]
        positive_rate = scored_test.agg(F.avg(label)).first()[0]

        mlflow.log_params({"chosen_model": champion["name"], "decision_threshold": threshold})
        mlflow.log_metrics({
            "test_pr_auc": test_auc["areaUnderPR"],
            "test_roc_auc": test_auc["areaUnderROC"],
            "test_f1": test_m["f1"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_accuracy": test_m["accuracy"],
            # Referencias para interpretar: un modelo al azar tiene PR-AUC = tasa de positivos,
            # y el que dice siempre "puntual" tiene accuracy = 1 - tasa de positivos
            "test_baseline_pr_auc": positive_rate,
            "test_baseline_accuracy": 1 - positive_rate,
        })
        mlflow.log_dict(test_m, "test_confusion_matrix.json")
        logger.info("TEST: PR-AUC=%.4f (azar=%.4f) ROC-AUC=%.4f F1=%.4f Precision=%.4f Recall=%.4f",
                    test_auc["areaUnderPR"], positive_rate, test_auc["areaUnderROC"],
                    test_m["f1"], test_m["precision"], test_m["recall"])

        for metric, value in [("pr_auc", test_auc["areaUnderPR"]), ("roc_auc", test_auc["areaUnderROC"]),
                              ("f1", test_m["f1"]), ("precision", test_m["precision"]),
                              ("recall", test_m["recall"]), ("accuracy", test_m["accuracy"]),
                              ("threshold", threshold)]:
            metric_rows.append((run_id, trained_at, champion["name"], "test", metric, float(value), True))

        # 6. Guardar el modelo con firma (Unity Catalog la exige)
        sample_in = test.select(*feature_columns(ml_cfg)).limit(5)
        signature = infer_signature(
            sample_in.toPandas(),
            champion["model"].transform(sample_in).select("prediction").toPandas(),
        )
        log_kwargs = {"artifact_path": "model", "signature": signature}
        if not is_local:
            log_kwargs["dfs_tmpdir"] = ml_cfg["paths"]["databricks_tmp_volume"]
        model_info = mlflow.spark.log_model(champion["model"], **log_kwargs)
        mlflow.set_tags({"champion": "true", "model_type": champion["name"]})

        # 7. Registrar en Unity Catalog con alias @champion (solo Databricks)
        if not is_local:
            _register_in_uc(model_info.model_uri, ml_cfg, champion["name"], threshold)
        else:
            logger.info("Local: modelo guardado en MLflow (runs:/%s/model), sin registro en UC.", run_id)

    # 8. Historial de métricas en Delta
    _save_metrics_table(spark, config, ml_cfg, metric_rows)

    importance_rows = [r[:6] + (r[2] == champion["name"],) for r in importance_rows]
    _save_feature_importance(spark, config, ml_cfg, importance_rows)

    if is_local:
        train.unpersist()
        val.unpersist()
    logger.info("Entrenamiento finalizado OK.")

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_train_model)