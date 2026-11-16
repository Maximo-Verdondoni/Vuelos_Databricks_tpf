import logging
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType, StructField, StructType

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


def run_clean_dimensions(spark: SparkSession, config: Dict[str, Any]) -> None:
    """Limpia aerolíneas, aeropuertos, feriados y genera el mapeo DOT ID -> IATA de octubre."""
    logger.info("Procesando dimensiones Silver (airlines, airports, holidays, dot_mapping)...")

    # 1. Airlines
    df_airlines = (
        _read_bronze(spark, config, "airlines_raw")
        .select(
            F.trim(F.col("IATA_CODE")).alias("airline_code"),
            F.trim(F.col("AIRLINE")).alias("airline_name"),
        )
        .dropDuplicates(["airline_code"])
        .filter(F.col("airline_code").isNotNull())
    )
    _write_silver(df_airlines, config, "airlines_silver")

    # 2. Airports (con imputación de latitud/longitud por estado validada en EDA)
    df_airports_raw = _read_bronze(spark, config, "airports_raw").select(
        F.trim(F.col("IATA_CODE")).alias("iata_code"),
        F.trim(F.col("AIRPORT")).alias("airport_name"),
        F.trim(F.col("CITY")).alias("city"),
        F.trim(F.col("STATE")).alias("state"),
        F.trim(F.col("COUNTRY")).alias("country"),
        F.col("LATITUDE").cast("double").alias("latitude"),
        F.col("LONGITUDE").cast("double").alias("longitude"),
    )

    state_win = Window.partitionBy("state")
    df_airports = (
        df_airports_raw
        .withColumn("latitude", F.coalesce(F.col("latitude"), F.avg("latitude").over(state_win)))
        .withColumn("longitude", F.coalesce(F.col("longitude"), F.avg("longitude").over(state_win)))
        .dropDuplicates(["iata_code"])
    )
    _write_silver(df_airports, config, "airports_silver")

    # 3. Tabla de mapeo DOT ID (5 dígitos en octubre) -> IATA (3 letras)
    df_flights_raw = _read_bronze(spark, config, "flights_raw").select(
        F.col("MONTH").cast("int").alias("month"),
        F.trim(F.col("AIRLINE")).alias("airline"),
        F.trim(F.col("FLIGHT_NUMBER")).alias("flight_number"),
        F.trim(F.col("ORIGIN_AIRPORT")).alias("origin_airport"),
    )

    # Referencia desde los meses con códigos IATA válidos (MONTH != 10)
    w_mode_ref = Window.partitionBy("airline", "flight_number").orderBy(F.desc("cnt"), F.asc("origin_airport"))
    df_ref = (
        df_flights_raw
        .filter((F.col("month") != 10) & F.col("origin_airport").rlike("^[A-Za-z]{3}$"))
        .groupBy("airline", "flight_number", "origin_airport")
        .agg(F.count("*").alias("cnt"))
        .withColumn("rn", F.row_number().over(w_mode_ref))
        .filter(F.col("rn") == 1)
        .select("airline", "flight_number", F.col("origin_airport").alias("mapped_iata"))
    )

    w_mode_dot = Window.partitionBy("dot_id").orderBy(F.desc("cnt"), F.asc("mapped_iata"))
    df_dot_map = (
        df_flights_raw
        .filter((F.col("month") == 10) & (~F.col("origin_airport").rlike("^[A-Za-z]{3}$")))
        .join(df_ref, on=["airline", "flight_number"], how="inner")
        .groupBy(F.col("origin_airport").alias("dot_id"), "mapped_iata")
        .agg(F.count("*").alias("cnt"))
        .withColumn("rn", F.row_number().over(w_mode_dot))
        .filter(F.col("rn") == 1)
        .select("dot_id", "mapped_iata")
    )

    # Agregar excepción puntual detectada en el EDA (10666 -> BLI) si no está presente
    df_exception = spark.createDataFrame([("10666", "BLI")], ["dot_id", "mapped_iata"])
    df_dot_map_final = df_dot_map.unionByName(df_exception).dropDuplicates(["dot_id"])
    _write_silver(df_dot_map_final, config, "dot_airport_mapping")

    # 4. Holidays (F3 opcional)
    holiday_schema = StructType([
        StructField("date", StringType(), True),
        StructField("localName", StringType(), True),
        StructField("name", StringType(), True),
        StructField("global", BooleanType(), True),
    ])
    df_holidays = (
        _read_bronze(spark, config, "holidays_raw")
        .withColumn("parsed", F.from_json(F.col("raw_json"), holiday_schema))
        .select(
            F.to_date(F.col("parsed.date")).alias("holiday_date"),
            F.col("parsed.name").alias("holiday_name"),
            F.col("parsed.global").alias("is_global_holiday"),
        )
        .filter(F.col("holiday_date").isNotNull())
        .dropDuplicates(["holiday_date"])
    )
    _write_silver(df_holidays, config, "holidays_silver")
    logger.info("Dimensiones Silver guardadas exitosamente.")