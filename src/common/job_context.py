"""Contexto de ejecución compartido por todos los scripts del pipeline.

Cada script de capa termina con:

    if __name__ == "__main__":
        from src.common.job_context import run_job
        run_job(run_xxx)

Así cada tarea del Job ejecuta su propio script, y la detección de entorno,
la sesión de Spark y la configuración se resuelven en un único lugar.
"""
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from pyspark.sql import SparkSession

from src.common.config_loader import load_config

# src/common/job_context.py -> parents[2] = raíz del repo
# (los módulos importados sí tienen __file__, también en serverless)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "parameters.yaml"


def is_databricks() -> bool:
    """Databricks define esta variable en todo su cómputo; en una compu no existe."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def get_spark_and_config(app_name: str = "VuelosDatabricksTPI") -> Tuple[SparkSession, Dict[str, Any]]:
    """Devuelve la SparkSession y la configuración según el entorno."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    overrides: Dict[str, Any] = {}

    if is_databricks():
        spark = SparkSession.builder.getOrCreate()
        overrides["environment"] = "databricks"
    else:
        from src.common.spark_session import get_spark_session
        spark = get_spark_session(app_name=app_name, is_local=True)

    config = load_config(str(CONFIG_PATH), overrides=overrides)
    config["_repo_root"] = str(REPO_ROOT)
    return spark, config


def run_job(func: Callable[[SparkSession, Dict[str, Any]], None]) -> None:
    """Punto de entrada de cada script: arma el contexto y ejecuta la etapa.
    No atrapa excepciones: si la etapa falla, la tarea queda en rojo."""
    spark, config = get_spark_and_config()
    logging.info("Ejecutando %s en modo='%s'", func.__name__, config["environment"])
    func(spark, config)
    logging.info("Etapa %s finalizada OK.", func.__name__)