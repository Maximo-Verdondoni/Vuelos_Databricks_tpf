"""Etapa 07 del Job de ML: carga el modelo campeón y puntúa un lote de vuelos."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ml.features import (
    load_ml_config,
    prepare_feature_types,
    read_training_data,
    validate_feature_columns,
)
from src.ml.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encontrar el modelo campeón
# ---------------------------------------------------------------------------
def _find_champion(ml_cfg: Dict[str, Any], is_local: bool) -> Tuple[str, float, Dict[str, str]]:
    """Devuelve (model_uri, umbral, info para trazabilidad)."""
    if is_local:
        # Local: la última corrida con tag champion = true
        runs = mlflow.search_runs(
            experiment_names=[ml_cfg["mlflow"]["local_experiment_name"]],
            filter_string="tags.champion = 'true'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError("No hay ningún modelo campeón en MLflow local. Corré primero 06_train_model.")
        row = runs.iloc[0]
        run_id = row["run_id"]
        threshold = float(row["params.decision_threshold"])
        info = {"model_ref": f"run:{run_id}", "model_type": row["tags.model_type"]}
        return f"runs:/{run_id}/model", threshold, info

    # Databricks: Unity Catalog con alias @champion
    reg = ml_cfg["registry"]
    full_name = f"{reg['catalog']}.{reg['schema']}.{reg['model_name']}"
    mv = MlflowClient().get_model_version_by_alias(full_name, reg["alias"])
    threshold = float(mv.tags["decision_threshold"])
    info = {"model_ref": f"{full_name}:v{mv.version}", "model_type": mv.tags.get("model_type", "unknown")}
    return f"models:/{full_name}@{reg['alias']}", threshold, info


def _load_pipeline(model_uri: str, is_local: bool, ml_cfg: Dict[str, Any]) -> PipelineModel:
    if is_local:
        return mlflow.spark.load_model(model_uri)

    tmp = ml_cfg["paths"]["databricks_tmp_volume"]
    try:
        return mlflow.spark.load_model(model_uri, dfs_tmpdir=tmp)
    except Exception as exc:  # noqa: BLE001
        # Plan B para serverless: bajar el modelo a un Volume y cargarlo con Spark directamente
        logger.warning("mlflow.spark.load_model falló (%s). Uso descarga a Volume + PipelineModel.load.", exc)
        local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=f"{tmp}/scoring_model")
        return PipelineModel.load(f"{local_dir.rstrip('/')}/sparkml")


# ---------------------------------------------------------------------------
# Escritura de predicciones
# ---------------------------------------------------------------------------
def _write_predictions(df: DataFrame, config: Dict[str, Any], ml_cfg: Dict[str, Any]) -> None:
    table = ml_cfg["outputs"]["predictions_table"]
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")

    if config["environment"] == "local":
        path = Path(config["_repo_root"]) / ml_cfg["paths"]["local_output_dir"] / table
        writer.save(str(path))
        logger.info("Predicciones guardadas en %s", path)
    else:
        reg = ml_cfg["registry"]
        full_name = f"{reg['catalog']}.{reg['schema']}.{table}"
        writer.saveAsTable(full_name)
        logger.info("Predicciones guardadas en %s", full_name)


# ---------------------------------------------------------------------------
# Entrada principal (la llama _job_runner.py)
# ---------------------------------------------------------------------------
def run_batch_scoring(spark: SparkSession, config: Dict[str, Any]) -> None:
    ml_cfg = load_ml_config(config)
    is_local = setup_mlflow(config, ml_cfg)
    label = ml_cfg["data"]["label_col"]
    split = ml_cfg["scoring"]["split"]
    scored_at = datetime.now(timezone.utc).isoformat()

    # 1. Modelo campeón y su umbral de decisión
    model_uri, threshold, info = _find_champion(ml_cfg, is_local)
    logger.info("Modelo campeón: %s (%s), umbral %.2f", info["model_ref"], info["model_type"], threshold)
    model = _load_pipeline(model_uri, is_local, ml_cfg)

    # 2. Lote a puntuar: la partición completa, sin muestreo
    df = read_training_data(spark, config, ml_cfg)
    validate_feature_columns(df, ml_cfg)
    batch = prepare_feature_types(df, ml_cfg).filter(F.col(ml_cfg["data"]["split_col"]) == split)

    # 3. Predicción con el umbral elegido en validation (no el 0,5 interno del modelo)
    scored = (
        model.transform(batch)
        .withColumn("p_delay", vector_to_array("probability")[1])
        .select(
            "flight_sk",
            "flight_date",
            "airline_code",
            "origin_airport_code",
            "dest_airport_code",
            "scheduled_dep_hour",
            F.round("p_delay", 4).alias("p_delay"),
            (F.col("p_delay") >= threshold).cast("int").alias("predicted_delay_15"),
            F.col(label).cast("int").alias("actual_delay_15"),  # disponible porque es histórico: sirve para monitorear
            F.lit(threshold).alias("decision_threshold"),
            F.lit(info["model_ref"]).alias("model_ref"),
            F.lit(info["model_type"]).alias("model_type"),
            F.lit(scored_at).alias("scored_at"),
        )
    )

    _write_predictions(scored, config, ml_cfg)

    # 4. Resumen de monitoreo
    s = scored.agg(
        F.count("*").alias("n"),
        F.avg("predicted_delay_15").alias("pred_rate"),
        F.avg("actual_delay_15").alias("actual_rate"),
        F.sum(F.col("predicted_delay_15") * F.col("actual_delay_15")).alias("tp"),
        F.sum("predicted_delay_15").alias("pred_pos"),
        F.sum("actual_delay_15").alias("actual_pos"),
    ).first()
    precision = s["tp"] / s["pred_pos"] if s["pred_pos"] else 0.0
    recall = s["tp"] / s["actual_pos"] if s["actual_pos"] else 0.0
    logger.info(
        "Scoring '%s': %d vuelos | predichos demorados %.1f%% | reales %.1f%% | precision %.3f | recall %.3f",
        split, s["n"], 100 * s["pred_rate"], 100 * s["actual_rate"], precision, recall,
    )
    logger.info("Batch scoring finalizado OK.")