import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession
from src.common.api_client import fetch_json_with_retry

logger = logging.getLogger(__name__)


def run_ingest_f3_holidays(spark: SparkSession, config: Dict[str, Any], batch_id: str | None = None) -> None:
    """Ingesta cruda de feriados federales de EE.UU. (Nager.Date API) hacia Bronze."""
    is_local = config["environment"] == "local"
    batch_id = batch_id or str(uuid.uuid4())
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    year = config["pipeline"]["year"]
    nager_cfg = config["apis"]["nager_date"]
    url = f"{nager_cfg['base_url'].rstrip('/')}/{year}/{nager_cfg['country_code']}"

    logger.info("Consultando feriados federales en %s", url)
    holidays_payload = fetch_json_with_retry(url)

    records = [
        {
            "raw_json": json.dumps(item),
            "_ingestion_ts": ingestion_ts,
            "_batch_id": batch_id,
            "_source_url": url,
        }
        for item in holidays_payload
    ]

    df_holidays = spark.createDataFrame(records)
    target = (
        str(Path(config["local"]["bronze_path"]) / "holidays_raw")
        if is_local
        else f"{config['databricks']['catalog']}.{config['databricks']['bronze_schema']}.holidays_raw"
    )

    writer = df_holidays.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    writer.save(target) if is_local else writer.saveAsTable(target)
    logger.info("Feriados Bronze guardados (%d registros).", len(records))

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_ingest_f3_holidays)