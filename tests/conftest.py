"""Fixtures compartidas por los tests."""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def ml_cfg(repo_root):
    from src.ml.features import load_ml_config
    return load_ml_config({"_repo_root": str(repo_root)})


@pytest.fixture(scope="session")
def spark():
    """Una sola SparkSession local para todos los tests marcados con @pytest.mark.spark."""
    from src.common.spark_session import get_spark_session
    session = get_spark_session(app_name="tests", is_local=True)
    yield session
    session.stop()