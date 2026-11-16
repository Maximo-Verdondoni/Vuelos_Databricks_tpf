import logging
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _read_bronze(spark: SparkSession, config: Dict[str, Any], table_name: str):
    if config["environment"] == "local":
        return spark.read.format("delta").load(str(Path(config["local"]["bronze_path"]) / table_name))
    db_cfg = config["databricks"]
    return spark.table(f"{db_cfg['catalog']}.{db_cfg['bronze_schema']}.{table_name}")


def _write_silver(df, config: Dict[str, Any], table_name: str) -> None:
    is_local = config["environment"] == "local"
    target = (
        str(Path(config["local"]["silver_path"]) / table_name)
        if is_local
        else f"{config['databricks']['catalog']}.{config['databricks']['silver_schema']}.{table_name}"
    )
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
    writer.save(target) if is_local else writer.saveAsTable(target)


def run_clean_flights(spark: SparkSession, config: Dict[str, Any]) -> None:
    """Tipifica, valida y conforma vuelos crudos de Bronze hacia flights_clean_silver."""
    logger.info("Procesando y tipificando vuelos hacia Silver...")

    df_raw = _read_bronze(spark, config, "flights_raw")

    df_typed = (
        df_raw
        .select(
            # Calendario e identificación (conocidos antes del vuelo)
            F.col("YEAR").cast("int").alias("year"),
            F.col("MONTH").cast("int").alias("month"),
            F.col("DAY").cast("int").alias("day"),
            F.col("DAY_OF_WEEK").cast("int").alias("day_of_week"),
            F.make_date(
                F.col("YEAR").cast("int"),
                F.col("MONTH").cast("int"),
                F.col("DAY").cast("int"),
            ).alias("flight_date"),
            F.trim(F.col("AIRLINE")).alias("airline_code"),
            F.col("FLIGHT_NUMBER").cast("int").alias("flight_number"),
            F.trim(F.col("TAIL_NUMBER")).alias("tail_number"),
            F.trim(F.col("ORIGIN_AIRPORT")).alias("raw_origin_airport"),
            F.trim(F.col("DESTINATION_AIRPORT")).alias("raw_dest_airport"),
            # Programación (conocida antes del vuelo)
            F.col("SCHEDULED_DEPARTURE").cast("int").alias("scheduled_departure_hhmm"),
            F.col("SCHEDULED_ARRIVAL").cast("int").alias("scheduled_arrival_hhmm"),
            F.col("SCHEDULED_TIME").cast("double").alias("scheduled_time_min"),
            F.col("DISTANCE").cast("double").alias("distance_miles"),
            # Operación real (NO disponibles antes del vuelo)
            F.col("DEPARTURE_TIME").cast("int").alias("departure_time_hhmm"),
            F.col("DEPARTURE_DELAY").cast("double").alias("departure_delay_min"),
            F.col("TAXI_OUT").cast("double").alias("taxi_out_min"),
            F.col("WHEELS_OFF").cast("int").alias("wheels_off_hhmm"),
            F.col("ELAPSED_TIME").cast("double").alias("elapsed_time_min"),
            F.col("AIR_TIME").cast("double").alias("air_time_min"),
            F.col("WHEELS_ON").cast("int").alias("wheels_on_hhmm"),
            F.col("TAXI_IN").cast("double").alias("taxi_in_min"),
            F.col("ARRIVAL_TIME").cast("int").alias("arrival_time_hhmm"),
            F.col("ARRIVAL_DELAY").cast("double").alias("arrival_delay_min"),
            # Estado final y causas de retraso (NO disponibles antes del vuelo)
            (F.col("DIVERTED").cast("int") == 1).alias("is_diverted"),
            (F.col("CANCELLED").cast("int") == 1).alias("is_cancelled"),
            F.trim(F.col("CANCELLATION_REASON")).alias("cancellation_reason"),
            F.col("AIR_SYSTEM_DELAY").cast("double").alias("air_system_delay_min"),
            F.col("SECURITY_DELAY").cast("double").alias("security_delay_min"),
            F.col("AIRLINE_DELAY").cast("double").alias("airline_delay_min"),
            F.col("LATE_AIRCRAFT_DELAY").cast("double").alias("late_aircraft_delay_min"),
            F.col("WEATHER_DELAY").cast("double").alias("weather_delay_min"),
        )
        # Hora programada (0 a 23) para join con clima horario en origen
        .withColumn("scheduled_dep_hour", (F.floor(F.col("scheduled_departure_hhmm") / 100) % 24).cast("int"))
        .withColumn("scheduled_dep_minute", (F.col("scheduled_departure_hhmm") % 100).cast("int"))
        # Controles de calidad estructurales
        .filter(
            (F.col("flight_date").isNotNull())
            & (F.col("airline_code").isNotNull())
            & (F.col("scheduled_departure_hhmm").between(0, 2359))
            & (F.col("distance_miles") > 0)
        )
    )

    _write_silver(df_typed, config, "flights_clean_silver")
    logger.info("Flights clean Silver guardado (%d filas).", df_typed.count())