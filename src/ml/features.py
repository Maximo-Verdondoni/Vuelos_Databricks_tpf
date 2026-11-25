"""Definición de features, splits y preprocesamiento del modelo de retraso de vuelos."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuración y lectura (local / Databricks)
# ---------------------------------------------------------------------------
def load_ml_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Lee conf/ml.yaml usando la raíz del repo que agrega _job_runner.py."""
    repo_root = Path(config.get("_repo_root", "."))
    with open(repo_root / "conf" / "ml.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def feature_columns(ml_cfg: Dict[str, Any]) -> List[str]:
    return ml_cfg["features"]["categorical"] + ml_cfg["features"]["numeric"]


def read_training_data(spark: SparkSession, config: Dict[str, Any], ml_cfg: Dict[str, Any]) -> DataFrame:
    """Lee Gold B desde disco local o desde Unity Catalog según el entorno."""
    table_name = ml_cfg["data"]["training_table"]

    if config["environment"] == "local":
        repo_root = Path(config.get("_repo_root", "."))
        local_path = str(repo_root / config["local"]["gold_path"] / table_name)
        logger.info("Leyendo tabla de entrenamiento local: %s", local_path)
        return spark.read.format("delta").load(local_path)

    db = config["databricks"]
    full_name = f"{db['catalog']}.{db['gold_schema']}.{table_name}"
    logger.info("Leyendo tabla de entrenamiento de Unity Catalog: %s", full_name)
    return spark.table(full_name)


# ---------------------------------------------------------------------------
# Contrato de datos y anti-leakage
# ---------------------------------------------------------------------------
def validate_feature_columns(df: DataFrame, ml_cfg: Dict[str, Any]) -> None:
    """Falla si una feature está prohibida o si falta alguna columna esperada."""
    feats = feature_columns(ml_cfg)
    label = ml_cfg["data"]["label_col"]
    split_col = ml_cfg["data"]["split_col"]

    leaked = set(feats) & set(ml_cfg["features"]["forbidden"])
    if leaked:
        raise ValueError(f"Data leakage: columnas prohibidas usadas como features: {sorted(leaked)}")

    expected = feats + [label, split_col] + ml_cfg["data"]["id_cols"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Contrato roto: faltan columnas en la tabla de entrenamiento: {missing}")

    logger.info("Contrato OK: %d features verificadas, 0 columnas prohibidas.", len(feats))


# ---------------------------------------------------------------------------
# Tipado, splits y pesos por clase
# ---------------------------------------------------------------------------
def prepare_feature_types(df: DataFrame, ml_cfg: Dict[str, Any]) -> DataFrame:
    """Categóricas a string, numéricas y label a double."""
    for c in ml_cfg["features"]["categorical"]:
        df = df.withColumn(c, F.col(c).cast("string"))
    for c in ml_cfg["features"]["numeric"]:
        df = df.withColumn(c, F.col(c).cast("double"))
    label_col = ml_cfg["data"]["label_col"]
    return df.withColumn(label_col, F.col(label_col).cast("double"))


def get_splits(df: DataFrame, ml_cfg: Dict[str, Any]) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Separa train / validation / test según la partición temporal de Gold B."""
    split_col = ml_cfg["data"]["split_col"]
    frac = float(ml_cfg["training"]["sample_fraction"])
    seed = int(ml_cfg["training"]["seed"])

    parts = {}
    for name in ("train", "validation", "test"):
        part = df.filter(F.col(split_col) == name)
        if frac < 1.0:
            part = part.sample(withReplacement=False, fraction=frac, seed=seed)
        parts[name] = part

    counts = {name: part.count() for name, part in parts.items()}
    empty = [name for name, n in counts.items() if n == 0]
    if empty:
        raise ValueError(
            f"Contrato roto: splits vacíos {empty}. "
            f"Valores esperados en '{split_col}': train, validation, test"
        )

    logger.info("Splits cargados (sample_fraction=%.2f): %s", frac, counts)
    return parts["train"], parts["validation"], parts["test"]


def compute_class_weights(train: DataFrame, ml_cfg: Dict[str, Any]) -> Dict[float, float]:
    """Pesos 'balanced' calculados SOLO con train: w_c = N / (2 * n_c)."""
    label = ml_cfg["data"]["label_col"]
    counts = {float(row[label]): int(row["count"]) for row in train.groupBy(label).count().collect()}
    total = sum(counts.values())
    weights = {cls: total / (2.0 * n) for cls, n in counts.items() if n > 0}
    logger.info("Distribución de clases en train: %s -> pesos: %s", counts, weights)
    return weights


def add_weight_column(df: DataFrame, weights: Dict[float, float], ml_cfg: Dict[str, Any]) -> DataFrame:
    label = ml_cfg["data"]["label_col"]
    w_pos = float(weights.get(1.0, 1.0))
    w_neg = float(weights.get(0.0, 1.0))
    return df.withColumn(
        "class_weight",
        F.when(F.col(label) == 1.0, F.lit(w_pos)).otherwise(F.lit(w_neg)),
    )


# ---------------------------------------------------------------------------
# Preprocesamiento (se ajusta SOLO con train dentro del Pipeline)
# ---------------------------------------------------------------------------
def build_preprocessing_stages(ml_cfg: Dict[str, Any]) -> List:
    """StringIndexer + OneHotEncoder para categóricas, Imputer para numéricas, VectorAssembler."""
    cat = ml_cfg["features"]["categorical"]
    num = ml_cfg["features"]["numeric"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat],
        outputCols=[f"{c}_ohe" for c in cat],
        handleInvalid="keep",
    )
    imputer = Imputer(
        inputCols=num,
        outputCols=[f"{c}_imp" for c in num],
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_ohe" for c in cat] + [f"{c}_imp" for c in num],
        outputCol="features",
        handleInvalid="keep",
    )
    return indexers + [encoder, imputer, assembler]

def build_tree_preprocessing_stages(ml_cfg: Dict[str, Any]) -> List:
    """Para árboles: categóricas como índice (sin one-hot), numéricas imputadas.
    StringIndexer deja metadatos 'nominal', así que el GBT las trata como categorías."""
    cat = ml_cfg["features"]["categorical"]
    num = ml_cfg["features"]["numeric"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat
    ]
    imputer = Imputer(
        inputCols=num,
        outputCols=[f"{c}_imp" for c in num],
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=[f"{c}_idx" for c in cat] + [f"{c}_imp" for c in num],
        outputCol="features",
        handleInvalid="keep",
    )
    return indexers + [imputer, assembler]


# ---------------------------------------------------------------------------
# Prueba de punta a punta (se corre con _job_runner.py)
# ---------------------------------------------------------------------------
def run_feature_smoke_test(spark: SparkSession, config: Dict[str, Any]) -> None:
    """Verifica contrato, splits, pesos y preprocesamiento sin entrenar ningún modelo."""
    ml_cfg = load_ml_config(config)

    df = read_training_data(spark, config, ml_cfg)
    validate_feature_columns(df, ml_cfg)
    df = prepare_feature_types(df, ml_cfg)

    train, val, test = get_splits(df, ml_cfg)
    weights = compute_class_weights(train, ml_cfg)
    train = add_weight_column(train, weights, ml_cfg)

    preprocessing = Pipeline(stages=build_preprocessing_stages(ml_cfg)).fit(train)
    val_features = preprocessing.transform(val)

    n_features = val_features.select("features").first()["features"].size
    logger.info("Preprocesamiento OK: vector de %d posiciones.", n_features)
    val_features.select(ml_cfg["data"]["label_col"], "features").show(5, truncate=80)
    logger.info("Smoke test de features finalizado OK.")

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_feature_smoke_test)