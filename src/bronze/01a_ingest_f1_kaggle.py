import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _get_target_table_or_path(config: Dict[str, Any], table_name: str) -> str:
    """Resuelve ruta Delta local o tabla de Unity Catalog según el entorno."""
    if config["environment"] == "local":
        return str(Path(config["local"]["bronze_path"]) / table_name)
    db_cfg = config["databricks"]
    return f"{db_cfg['catalog']}.{db_cfg['bronze_schema']}.{table_name}"


def _write_delta(df, target: str, is_local: bool, mode: str = "overwrite") -> None:
    """Escribe en Delta Lake de forma idempotente (re-ejecutable sin duplicar)."""
    writer = df.write.format("delta").mode(mode).option("overwriteSchema", "true")
    if is_local:
        writer.save(target)
    else:
        writer.saveAsTable(target)


def run_ingest_f1(spark: SparkSession, config: Dict[str, Any], batch_id: str | None = None) -> None:
    """
    Ingesta cruda de F1 (flights, airlines, airports) hacia la capa Bronze.
    Garantiza copia fiel con metadatos de trazabilidad y registro de errores.
    """
    is_local = config["environment"] == "local"
    raw_dir = config["local"]["raw_path"] if is_local else config["databricks"]["volume_raw_path"]
    batch_id = batch_id or str(uuid.uuid4())
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    files_to_ingest = {
        "flights_raw": "flights.csv",
        "airlines_raw": "airlines.csv",
        "airports_raw": "airports.csv",
    }

    error_records = []

    for table_name, filename in files_to_ingest.items():
        file_path = f"{raw_dir.rstrip('/')}/{filename}"
        logger.info("Ingestando %s -> %s (batch_id=%s)", file_path, table_name, batch_id)

        try:
            # Leemos todo como string en Bronze para asegurar copia fiel sin fallos de casteo
            df_raw = (
                spark.read.option("header", "true")
                .option("inferSchema", "false")
                .option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .csv(file_path)
            )

            # Separar registros corruptos si los hubiera para auditoría de errores
            if "_corrupt_record" in df_raw.columns:
                df_corrupt = df_raw.filter(F.col("_corrupt_record").isNotNull())
                corrupt_count = df_corrupt.count()
                if corrupt_count > 0:
                    error_records.append({
                        "batch_id": batch_id,
                        "ingestion_ts": ingestion_ts,
                        "source": filename,
                        "entity": table_name,
                        "error_type": "CORRUPT_CSV_ROWS",
                        "error_detail": f"{corrupt_count} filas malformadas detectadas",
                    })
                df_clean_raw = df_raw.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")
            else:
                df_clean_raw = df_raw

            # Agregar columnas de trazabilidad requeridas en Bronze
            df_bronze = (
                df_clean_raw
                .withColumn("_ingestion_ts", F.lit(ingestion_ts))
                .withColumn("_batch_id", F.lit(batch_id))
                .withColumn("_source_file", F.lit(file_path))
            )

            target = _get_target_table_or_path(config, table_name)
            _write_delta(df_bronze, target, is_local, mode="overwrite")
            logger.info("Ingesta exitosa de %s (%d filas)", table_name, df_bronze.count())

        except Exception as exc:
            logger.error("Error ingestando %s: %s", filename, str(exc))
            error_records.append({
                "batch_id": batch_id,
                "ingestion_ts": ingestion_ts,
                "source": filename,
                "entity": table_name,
                "error_type": type(exc).__name__,
                "error_detail": str(exc)[:500],
            })
            raise

    # Persistir registro de errores (incluso si está vacío, asegura esquema de auditoría)
    if error_records:
        df_errors = spark.createDataFrame(error_records)
        err_target = _get_target_table_or_path(config, "ingestion_errors_f1")
        _write_delta(df_errors, err_target, is_local, mode="append")

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_ingest_f1)