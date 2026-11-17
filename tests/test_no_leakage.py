"""Tests anti-leakage: ninguna variable conocida DESPUÉS de la salida puede ser feature."""
import copy
from types import SimpleNamespace

import pytest

from src.ml.features import feature_columns, validate_feature_columns

# Fragmentos de nombres de columnas que solo existen después de la salida
POST_DEPARTURE_PATTERNS = (
    "departure_delay", "departure_time", "arrival_", "taxi_", "wheels_",
    "air_time", "elapsed_time", "_delay_min", "cancel", "divert",
)


def _fake_df(columns):
    """validate_feature_columns solo usa df.columns: alcanza con un objeto simple, sin Spark."""
    return SimpleNamespace(columns=list(columns))


def _all_expected_columns(ml_cfg):
    d = ml_cfg["data"]
    return feature_columns(ml_cfg) + [d["label_col"], d["split_col"]] + d["id_cols"]


# --- Reglas sobre la configuración (conf/ml.yaml) ---------------------------

def test_ninguna_feature_esta_en_la_lista_prohibida(ml_cfg):
    leaked = set(feature_columns(ml_cfg)) & set(ml_cfg["features"]["forbidden"])
    assert not leaked, f"Features prohibidas en ml.yaml: {sorted(leaked)}"


def test_ninguna_feature_tiene_nombre_de_variable_post_salida(ml_cfg):
    sospechosas = [
        f for f in feature_columns(ml_cfg)
        if any(p in f for p in POST_DEPARTURE_PATTERNS)
    ]
    assert not sospechosas, f"Features con nombre de variable posterior a la salida: {sospechosas}"


def test_la_variable_objetivo_no_es_feature(ml_cfg):
    assert ml_cfg["data"]["label_col"] not in feature_columns(ml_cfg)


def test_month_no_es_feature(ml_cfg):
    # Decisión de diseño: train solo tiene meses 1-8, así que month obligaría a extrapolar en test
    assert "month" not in feature_columns(ml_cfg)


def test_no_hay_features_duplicadas(ml_cfg):
    feats = feature_columns(ml_cfg)
    assert len(feats) == len(set(feats))


# --- Comportamiento de validate_feature_columns (el contrato del Job de ML) --

def test_validate_acepta_un_contrato_completo(ml_cfg):
    validate_feature_columns(_fake_df(_all_expected_columns(ml_cfg)), ml_cfg)


def test_validate_falla_si_se_cuela_una_columna_prohibida(ml_cfg):
    cfg = copy.deepcopy(ml_cfg)
    cfg["features"]["numeric"].append("departure_delay_min")
    df = _fake_df(_all_expected_columns(cfg))
    with pytest.raises(ValueError, match="leakage"):
        validate_feature_columns(df, cfg)


def test_validate_falla_si_falta_una_columna(ml_cfg):
    columnas = [c for c in _all_expected_columns(ml_cfg) if c != "distance_miles"]
    with pytest.raises(ValueError, match="Contrato roto"):
        validate_feature_columns(_fake_df(columnas), ml_cfg)