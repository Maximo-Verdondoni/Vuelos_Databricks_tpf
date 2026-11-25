import os
import logging
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

logger = logging.getLogger(__name__)


def _read_bronze(spark: SparkSession, config: Dict[str, Any], table_name: str):
    if config["environment"] == "local":
        return spark.read.format("delta").load(str(Path(config["local"]["bronze_path"]) / table_name))
    db_cfg = config["databricks"]
    return spark.table(f"{db_cfg['catalog']}.{db_cfg['bronze_schema']}.{table_name}")


def _write_silver(df, config: Dict[str, Any], table_name: str) -> None:
    """Escribe la tabla Silver en disco local (si está en dev local) o en Unity Catalog (si está en Databricks)."""
    is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    is_local = (config.get("environment") == "local") and not is_databricks

    if is_local:
        target = str(Path(config["local"]["silver_path"]) / table_name)
        writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        writer.save(target)
    else:
        db_cfg = config.get("databricks", {})
        catalog = db_cfg.get("catalog", "vuelos_dev")
        schema = db_cfg.get("silver_schema", "silver")
        target = f"{catalog}.{schema}.{table_name}"
        
        writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        writer.saveAsTable(target)


def run_clean_weather(spark: SparkSession, config: Dict[str, Any]) -> None:
    """Desanida, tipifica y valida los datos horarios de Open-Meteo hacia Silver."""
    logger.info("Procesando clima horario Open-Meteo hacia Silver...")

    weather_payload_schema = StructType([
        StructField("timezone", StringType(), True),
        StructField("utc_offset_seconds", IntegerType(), True),
        StructField(
            "hourly",
            StructType([
                StructField("time", ArrayType(StringType()), True),
                StructField("temperature_2m", ArrayType(DoubleType()), True),
                StructField("precipitation", ArrayType(DoubleType()), True),
                StructField("snowfall", ArrayType(DoubleType()), True),
                StructField("wind_speed_10m", ArrayType(DoubleType()), True),
                StructField("wind_gusts_10m", ArrayType(DoubleType()), True),
                StructField("cloud_cover", ArrayType(DoubleType()), True),
            ]),
            True,
        ),
    ])

    df_raw = _read_bronze(spark, config, "weather_raw").withColumn(
        "parsed", F.from_json(F.col("raw_json"), weather_payload_schema)
    )

    df_exploded = df_raw.select(
        F.col("iata_code"),
        F.col("parsed.timezone").alias("timezone"),
        F.explode(
            F.arrays_zip(
                F.col("parsed.hourly.time"),
                F.col("parsed.hourly.temperature_2m"),
                F.col("parsed.hourly.precipitation"),
                F.col("parsed.hourly.snowfall"),
                F.col("parsed.hourly.wind_speed_10m"),
                F.col("parsed.hourly.wind_gusts_10m"),
                F.col("parsed.hourly.cloud_cover"),
            )
        ).alias("h"),
    )

    df_weather_silver = (
        df_exploded
        .select(
            F.col("iata_code"),
            F.col("timezone"),
            F.to_timestamp(F.col("h.time"), "yyyy-MM-dd'T'HH:mm").alias("weather_ts"),
            F.to_date(F.col("h.time")).alias("weather_date"),
            F.hour(F.to_timestamp(F.col("h.time"), "yyyy-MM-dd'T'HH:mm")).alias("weather_hour"),
            F.col("h.temperature_2m").alias("temperature_2m"),
            F.coalesce(F.col("h.precipitation"), F.lit(0.0)).alias("precipitation"),
            F.coalesce(F.col("h.snowfall"), F.lit(0.0)).alias("snowfall"),
            F.col("h.wind_speed_10m").alias("wind_speed_10m"),
            F.col("h.wind_gusts_10m").alias("wind_gusts_10m"),
            F.col("h.cloud_cover").alias("cloud_cover"),
        )
        # Controles de calidad de rangos físicos
        .filter(
            (F.col("weather_ts").isNotNull())
            & (F.col("temperature_2m").between(-60.0, 60.0))
            & (F.col("precipitation") >= 0.0)
            & (F.col("snowfall") >= 0.0)
            & (F.col("wind_speed_10m") >= 0.0)
            & (F.col("cloud_cover").between(0.0, 100.0))
        )
        .dropDuplicates(["iata_code", "weather_date", "weather_hour"])
    )

    _write_silver(df_weather_silver, config, "weather_silver")
    logger.info("Weather Silver guardado exitosamente (%d registros horarios).", df_weather_silver.count())

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_clean_weather)