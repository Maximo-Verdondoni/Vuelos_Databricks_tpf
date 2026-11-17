"""Tests de la partición temporal train / validation / test."""
import copy

import pytest
from pyspark.sql import functions as F

from src.common.config_loader import load_config
from src.ml.features import get_splits


# --- Configuración (sin Spark) ----------------------------------------------

def test_config_de_splits_es_temporal_y_no_se_solapa(repo_root):
    config = load_config(str(repo_root / "conf" / "parameters.yaml"))
    s = config["pipeline"]["ml_splits"]
    assert 1 <= s["train_end_month"] < s["val_end_month"] < 12, (
        "Debe quedar: train = 1..train_end, validation = train_end+1..val_end, test = val_end+1..12"
    )


# --- get_splits (Spark local con datos de juguete) --------------------------

def _cfg_sin_muestreo(ml_cfg):
    cfg = copy.deepcopy(ml_cfg)
    cfg["training"]["sample_fraction"] = 1.0
    return cfg


@pytest.mark.spark
def test_get_splits_separa_las_tres_particiones(spark, ml_cfg):
    cfg = _cfg_sin_muestreo(ml_cfg)
    split_col = cfg["data"]["split_col"]
    df = spark.createDataFrame(
        [("train",)] * 5 + [("validation",)] * 3 + [("test",)] * 2, [split_col]
    )
    train, val, test = get_splits(df, cfg)
    assert (train.count(), val.count(), test.count()) == (5, 3, 2)


@pytest.mark.spark
def test_get_splits_falla_si_un_split_esta_vacio(spark, ml_cfg):
    cfg = _cfg_sin_muestreo(ml_cfg)
    split_col = cfg["data"]["split_col"]
    df = spark.createDataFrame([("train",), ("test",)], [split_col])
    with pytest.raises(ValueError, match="splits vacíos"):
        get_splits(df, cfg)


@pytest.mark.spark
def test_get_splits_rechaza_nombres_no_esperados(spark, ml_cfg):
    # Si 04b escribiera "val" en vez de "validation", queremos que falle claro
    cfg = _cfg_sin_muestreo(ml_cfg)
    split_col = cfg["data"]["split_col"]
    df = spark.createDataFrame([("train",), ("val",), ("test",)], [split_col])
    with pytest.raises(ValueError, match="splits vacíos"):
        get_splits(df, cfg)


# --- Tabla real de Gold B (si existe en local) ------------------------------

@pytest.mark.spark
def test_gold_b_respeta_el_orden_temporal(spark, repo_root):
    config = load_config(str(repo_root / "conf" / "parameters.yaml"))
    path = repo_root / config["local"]["gold_path"] / "ml_flight_delay_training"
    if not path.exists():
        pytest.skip(f"No existe {path}: corré 04b en local para habilitar este test")

    rangos = {
        r["dataset_split"]: (r["min_m"], r["max_m"])
        for r in (
            spark.read.format("delta").load(str(path))
            .groupBy("dataset_split")
            .agg(F.min("month").alias("min_m"), F.max("month").alias("max_m"))
            .collect()
        )
    }
    s = config["pipeline"]["ml_splits"]

    assert set(rangos) == {"train", "validation", "test"}
    assert rangos["train"] == (1, s["train_end_month"])
    assert rangos["validation"] == (s["train_end_month"] + 1, s["val_end_month"])
    assert rangos["test"] == (s["val_end_month"] + 1, 12)