import builtins
import importlib
import logging
import os
import sys
from pathlib import Path

# Asegurar que la raíz del repo esté en sys.path (el runner vive en <repo>/notebooks/)
def _find_repo_root() -> Path:
    """Busca la raíz del repo (carpeta que contiene conf/parameters.yaml y src/).
    En Databricks serverless __file__ no existe, así que probamos varias pistas."""
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    if sys.argv and sys.argv[0].endswith(".py"):
        starts.append(Path(sys.argv[0]).resolve().parent)
    starts.append(Path.cwd())

    for start in starts:
        for p in [start, *start.parents]:
            if (p / "conf" / "parameters.yaml").exists() and (p / "src").is_dir():
                return p
    raise RuntimeError(f"No se encontró la raíz del repo. Probé desde: {starts}")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

CONFIG_PATH = str(REPO_ROOT / "conf" / "parameters.yaml")


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
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        overrides["environment"] = "databricks"

        dbutils = _get_dbutils()
        if dbutils is not None:
            try:
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
            except Exception as exc:
                logging.warning("Widgets no disponibles, uso valores del YAML: %s", exc)
    else:
        from src.common.spark_session import get_spark_session
        spark = get_spark_session(is_local=True)

    config = load_config(CONFIG_PATH, overrides=overrides)
    logging.info("Ejecutando %s.%s en modo='%s'", module_path, func_name, config["environment"])

    module = importlib.import_module(module_path)
    getattr(module, func_name)(spark, config)
    logging.info("Etapa %s.%s finalizada OK.", module_path, func_name)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Uso: _job_runner.py <module_path> <func_name>")
    run_step(sys.argv[1], sys.argv[2])