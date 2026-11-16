import builtins
import importlib
import logging
import os
import sys
from pathlib import Path

# Asegurar que la raíz del repo esté en sys.path cuando Databricks ejecuta desde notebooks/
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.common.config_loader import load_config
from src.common.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils
        return dbutils
    except ImportError:
        return getattr(builtins, "dbutils", None)


def run_step(module_path: str, func_name: str) -> None:
    """Detecta entorno Databricks/Local, lee parámetros del Job y ejecuta la etapa."""
    is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    overrides = {}

    if is_databricks:
        spark = get_spark_session(is_local=False)
        overrides["environment"] = "databricks"
        dbutils = _get_dbutils()

        if dbutils is not None:
            dbutils.widgets.text("environment", "databricks")
            dbutils.widgets.text("catalog", "vuelos_dev")
            dbutils.widgets.text("start_date", "2015-01-01")
            dbutils.widgets.text("end_date", "2015-12-31")

            overrides.update({
                "environment": dbutils.widgets.get("environment") or "databricks",
                "databricks.catalog": dbutils.widgets.get("catalog") or "vuelos_dev",
                "pipeline.start_date": dbutils.widgets.get("start_date") or "2015-01-01",
                "pipeline.end_date": dbutils.widgets.get("end_date") or "2015-12-31",
            })
    else:
        spark = get_spark_session(is_local=True)

    config = load_config(overrides=overrides)
    logging.info("Ejecutando %s.%s en modo='%s'", module_path, func_name, config["environment"])

    module = importlib.import_module(module_path)
    getattr(module, func_name)(spark, config)