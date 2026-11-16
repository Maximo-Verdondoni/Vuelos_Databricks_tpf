import importlib
import logging
import sys
from pathlib import Path

# Asegurar que la raíz del repo esté en sys.path cuando Databricks ejecuta desde notebooks/
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.common.config_loader import load_config
from src.common.spark_session import get_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_step(module_path: str, func_name: str) -> None:
    """Lee parámetros del Job de Databricks (widgets), carga config y ejecuta la etapa."""
    overrides = {}
    try:
        from pyspark.dbutils import DBUtils
        spark = get_spark_session(is_local=False)
        dbutils = DBUtils(spark)

        dbutils.widgets.text("environment", "databricks")
        dbutils.widgets.text("catalog", "vuelos_dev")
        dbutils.widgets.text("start_date", "2015-01-01")
        dbutils.widgets.text("end_date", "2015-12-31")

        overrides = {
            "environment": dbutils.widgets.get("environment"),
            "databricks.catalog": dbutils.widgets.get("catalog"),
            "pipeline.start_date": dbutils.widgets.get("start_date"),
            "pipeline.end_date": dbutils.widgets.get("end_date"),
        }
    except Exception:
        spark = get_spark_session(is_local=True)

    config = load_config(overrides=overrides)
    module = importlib.import_module(module_path)
    getattr(module, func_name)(spark, config)