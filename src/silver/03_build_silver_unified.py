import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _read_silver(spark: SparkSession, config: Dict[str, Any], table_name: str):
    if config["environment"] == "local":
        return spark.read.format("delta").load(str(Path(config["local"]["silver_path"]) / table_name))
    db_cfg = config["databricks"]
    return spark.table(f"{db_cfg['catalog']}.{db_cfg['silver_schema']}.{table_name}")


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


def run_build_silver_unified(spark: SparkSession, config: Dict[str, Any]) -> None:
    """
    Integra vuelos, dimensiones, clima horario y feriados en la visión empresarial única (flights_silver)
    y registra los controles de calidad de datos.
    """
    logger.info("Construyendo tabla empresarial unificada flights_silver...")

    df_flights = _read_silver(spark, config, "flights_clean_silver")
    df_dot_map = _read_silver(spark, config, "dot_airport_mapping")
    df_airlines = _read_silver(spark, config, "airlines_silver")
    df_airports = _read_silver(spark, config, "airports_silver")
    df_weather = _read_silver(spark, config, "weather_silver")
    df_holidays = _read_silver(spark, config, "holidays_silver")

    # 1. Resolver códigos IATA reales en origen y destino (traduciendo IDs numéricos de octubre)
    map_orig = df_dot_map.select(F.col("dot_id").alias("raw_origin_airport"), F.col("mapped_iata").alias("orig_mapped"))
    map_dest = df_dot_map.select(F.col("dot_id").alias("raw_dest_airport"), F.col("mapped_iata").alias("dest_mapped"))

    df_resolved = (
        df_flights
        .join(F.broadcast(map_orig), on="raw_origin_airport", how="left")
        .join(F.broadcast(map_dest), on="raw_dest_airport", how="left")
        .withColumn("origin_airport_code", F.coalesce(F.col("orig_mapped"), F.col("raw_origin_airport")))
        .withColumn("dest_airport_code", F.coalesce(F.col("dest_mapped"), F.col("raw_dest_airport")))
        .drop("orig_mapped", "dest_mapped")
    )

    # 2. Enriquecer con dimensiones maestras (inner join para garantizar integridad referencial)
    orig_airports = df_airports.select(
        F.col("iata_code").alias("origin_airport_code"),
        F.col("airport_name").alias("origin_airport_name"),
        F.col("city").alias("origin_city"),
        F.col("state").alias("origin_state"),
        F.col("latitude").alias("origin_latitude"),
        F.col("longitude").alias("origin_longitude"),
    )
    dest_airports = df_airports.select(
        F.col("iata_code").alias("dest_airport_code"),
        F.col("airport_name").alias("dest_airport_name"),
        F.col("city").alias("dest_city"),
        F.col("state").alias("dest_state"),
        F.col("latitude").alias("dest_latitude"),
        F.col("longitude").alias("dest_longitude"),
    )

    # 3. Cruce con clima horario en el aeropuerto de origen y con feriados federales
    df_unified = (
        df_resolved
        .join(F.broadcast(df_airlines), on="airline_code", how="inner")
        .join(F.broadcast(orig_airports), on="origin_airport_code", how="inner")
        .join(F.broadcast(dest_airports), on="dest_airport_code", how="inner")
        .join(
            df_weather,
            on=[
                df_resolved["origin_airport_code"] == df_weather["iata_code"],
                df_resolved["flight_date"] == df_weather["weather_date"],
                df_resolved["scheduled_dep_hour"] == df_weather["weather_hour"],
            ],
            how="left",
        )
        .drop(df_weather["iata_code"], df_weather["weather_date"], df_weather["weather_hour"])
        .join(
            F.broadcast(df_holidays),
            on=df_resolved["flight_date"] == df_holidays["holiday_date"],
            how="left",
        )
        .withColumn("is_us_holiday", F.col("holiday_date").isNotNull())
        .drop("holiday_date")
        # Flag de disponibilidad climática (Top 50 aeropuertos) y variable objetivo
        .withColumn("has_weather_data", F.col("temperature_2m").isNotNull())
        .withColumn(
            "is_delayed_15",
            F.when(
                (~F.col("is_cancelled")) & (~F.col("is_diverted")),
                (F.col("arrival_delay_min") >= config["pipeline"]["delay_threshold_minutes"]).cast("int"),
            ).otherwise(F.lit(None).cast("int")),
        )
        # Clave subrogada única por vuelo para trazabilidad
        .withColumn("flight_sk", F.monotonically_increasing_id())
    )

    _write_silver(df_unified, config, "flights_silver")

    # 4. Controles de calidad y auditoría de Silver
    total_clean = df_flights.count()
    total_unified = df_unified.count()
    with_weather = df_unified.filter(F.col("has_weather_data")).count()
    completed_with_weather = df_unified.filter(
        (~F.col("is_cancelled")) & (~F.col("is_diverted")) & F.col("has_weather_data")
    ).count()

    quality_rows = [
        ("flights_clean_input", total_clean, "Filas válidas tras limpieza inicial"),
        ("flights_silver_output", total_unified, "Filas con integridad referencial completa (airlines + airports)"),
        ("flights_with_weather", with_weather, "Vuelos en Top 50 aeropuertos con cruce horario exitoso"),
        ("gold_b_eligible_rows", completed_with_weather, "Vuelos completados con clima disponible para ML"),
    ]
    df_quality = (
        spark.createDataFrame(quality_rows, ["metric_name", "row_count", "description"])
        .withColumn("checked_at_utc", F.lit(datetime.now(timezone.utc).isoformat()))
    )
    _write_silver(df_quality, config, "silver_quality_metrics")
    logger.info("Silver unificado finalizado: %d filas totales, %d elegibles Gold B.", total_unified, completed_with_weather)

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_build_silver_unified)