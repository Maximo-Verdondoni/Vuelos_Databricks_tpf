"""Configuración de MLflow compartida por las etapas del Job de ML."""
import logging
import os
from pathlib import Path
from typing import Any, Dict

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(config: Dict[str, Any], ml_cfg: Dict[str, Any]) -> bool:
    """Local: SQLite + carpeta mlartifacts/. Databricks: workspace + Unity Catalog.
    Devuelve True si corre en local."""
    is_local = config["environment"] == "local"
    if is_local:
        repo_root = Path(config["_repo_root"])
        db_path = repo_root / "mlflow.db"
        artifacts_dir = repo_root / "mlartifacts"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")

        exp_name = ml_cfg["mlflow"]["local_experiment_name"]
        if mlflow.get_experiment_by_name(exp_name) is None:
            mlflow.create_experiment(exp_name, artifact_location=artifacts_dir.as_uri())
        mlflow.set_experiment(exp_name)
        logger.info("MLflow local: base %s, artefactos en %s", db_path, artifacts_dir)
    else:
        # En serverless, Spark ML y MLflow necesitan un Volume de UC para archivos temporales
        tmp = ml_cfg["paths"]["databricks_tmp_volume"]
        os.environ["SPARKML_TEMP_DFS_PATH"] = tmp
        os.environ["SPARK_ML_TMP_DFS_PATH"] = tmp
        os.environ["MLFLOW_DFS_TMP"] = tmp
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(ml_cfg["mlflow"]["experiment_path"])
        logger.info("MLflow en Databricks, registro en Unity Catalog, temporales en %s", tmp)
    return is_local