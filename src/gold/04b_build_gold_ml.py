import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def _read_silver(spark: SparkSession, config: Dict[str, Any], table_name: str):
    if config["environment"] == "local":
        return spark.read.format("delta").load(str(Path(config["local"]["silver_path"]) / table_name))
    db_cfg = config["databricks"]
    return spark.table(f"{db_cfg['catalog']}.{db_cfg['silver_schema']}.{table_name}")


def _write_gold(df, config: Dict[str, Any], table_name: str) -> None:
    # Forzar Databricks si estamos ejecutando en el runtime de Databricks
    is_databricks = "DATABRICKS_RUNTIME_VERSION" in os.environ
    is_local = (config.get("environment") == "local") and not is_databricks

    if is_local:
        target = str(Path(config["local"]["gold_path"]) / table_name)
        writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        writer.save(target)
    else:
        db_cfg = config["databricks"]
        catalog = db_cfg.get("catalog", "vuelos_dev")
        schema = db_cfg.get("gold_schema", "gold")
        target = f"{catalog}.{schema}.{table_name}"
        
        writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        writer.saveAsTable(target)


def run_build_gold_ml(spark: SparkSession, config: Dict[str, Any]) -> None:
    """
    Construye ml_flight_delay_training sin fuga de información (data leakage),
    con variables históricas estrictamente previas, partición temporal y auditoría de filas.
    """
    logger.info("Construyendo Gold B: Dataset de entrenamiento para Machine Learning...")

    df_silver = _read_silver(spark, config, "flights_silver")
    splits_cfg = config["pipeline"]["ml_splits"]

    # 1. Auditoría de población: cuántas filas entran y cuántas se excluyen por criterio
    total_silver = df_silver.count()
    excluded_cancelled = df_silver.filter(F.col("is_cancelled")).count()
    excluded_diverted = df_silver.filter((~F.col("is_cancelled")) & F.col("is_diverted")).count()
    excluded_no_weather = df_silver.filter(
        (~F.col("is_cancelled")) & (~F.col("is_diverted")) & (~F.col("has_weather_data"))
    ).count()

    df_eligible = df_silver.filter(
        (~F.col("is_cancelled"))
        & (~F.col("is_diverted"))
        & F.col("has_weather_data")
        & F.col("is_delayed_15").isNotNull()
    )

    # 2. Variables históricas SIN fuga de información (solo días estrictamente anteriores al vuelo)
    # Agregamos primero por (airline_code, flight_date) y usamos ventana ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    w_airline_hist = Window.partitionBy("airline_code").orderBy("flight_date").rowsBetween(Window.unboundedPreceding, -1)
    df_airline_daily = (
        df_eligible
        .groupBy("airline_code", "flight_date")
        .agg(
            F.count("*").alias("daily_flights"),
            F.sum("is_delayed_15").alias("daily_delays"),
            F.sum("departure_delay_min").alias("daily_dep_delay_sum"),
        )
        .withColumn("cum_flights", F.sum("daily_flights").over(w_airline_hist))
        .withColumn("cum_delays", F.sum("daily_delays").over(w_airline_hist))
        .withColumn("cum_dep_delay_sum", F.sum("daily_dep_delay_sum").over(w_airline_hist))
        .select(
            "airline_code",
            "flight_date",
            F.coalesce(F.col("cum_delays") / F.col("cum_flights"), F.lit(0.19)).alias("airline_hist_delay_rate"),
            F.coalesce(F.col("cum_dep_delay_sum") / F.col("cum_flights"), F.lit(0.0)).alias("airline_hist_avg_dep_delay"),
        )
    )

    # Lo mismo para el aeropuerto de origen (congestión histórica previa al día del vuelo)
    w_origin_hist = Window.partitionBy("origin_airport_code").orderBy("flight_date").rowsBetween(Window.unboundedPreceding, -1)
    df_origin_daily = (
        df_eligible
        .groupBy("origin_airport_code", "flight_date")
        .agg(
            F.count("*").alias("daily_flights"),
            F.sum("is_delayed_15").alias("daily_delays"),
            F.sum("departure_delay_min").alias("daily_dep_delay_sum"),
        )
        .withColumn("cum_flights", F.sum("daily_flights").over(w_origin_hist))
        .withColumn("cum_delays", F.sum("daily_delays").over(w_origin_hist))
        .withColumn("cum_dep_delay_sum", F.sum("daily_dep_delay_sum").over(w_origin_hist))
        .select(
            "origin_airport_code",
            "flight_date",
            F.coalesce(F.col("cum_delays") / F.col("cum_flights"), F.lit(0.19)).alias("origin_hist_delay_rate"),
            F.coalesce(F.col("cum_dep_delay_sum") / F.col("cum_flights"), F.lit(0.0)).alias("origin_hist_avg_dep_delay"),
        )
    )

    # 3. Ensamblar dataset final con solo variables conocidas ANTES de la salida programada
    df_ml = (
        df_eligible
        .join(df_airline_daily, on=["airline_code", "flight_date"], how="left")
        .join(df_origin_daily, on=["origin_airport_code", "flight_date"], how="left")
        .withColumn(
            "dataset_split",
            F.when(F.col("month") <= splits_cfg["train_end_month"], F.lit("train"))
            .when(F.col("month") <= splits_cfg["val_end_month"], F.lit("validation"))
            .otherwise(F.lit("test")),
        )
        .select(
            # Identificador y partición temporal
            F.col("flight_sk"),
            F.col("flight_date"),
            F.col("dataset_split"),
            # Variable objetivo (binaria: arrival_delay_min >= 15)
            F.col("is_delayed_15").alias("target_is_delayed_15"),
            # Predictoras de calendario (pre-salida)
            F.col("month"),
            F.col("day_of_week"),
            F.col("day_of_week").isin([6, 7]).cast("int").alias("is_weekend"),
            F.col("is_us_holiday").cast("int").alias("is_us_holiday"),
            # Predictoras de ruta y programación (pre-salida)
            F.col("airline_code"),
            F.col("origin_airport_code"),
            F.col("dest_airport_code"),
            F.col("scheduled_dep_hour"),
            F.col("scheduled_time_min"),
            F.col("distance_miles"),
            # Predictoras de clima en origen (pronóstico perfecto asumido a hora de salida)
            F.col("temperature_2m"),
            F.col("precipitation"),
            F.col("snowfall"),
            F.col("wind_speed_10m"),
            F.coalesce(F.col("wind_gusts_10m"), F.col("wind_speed_10m")).alias("wind_gusts_10m"),
            F.coalesce(F.col("cloud_cover"), F.lit(0.0)).alias("cloud_cover"),
            # Predictoras de comportamiento histórico previo (sin data leakage)
            F.round(F.col("airline_hist_delay_rate"), 4).alias("airline_hist_delay_rate"),
            F.round(F.col("airline_hist_avg_dep_delay"), 2).alias("airline_hist_avg_dep_delay"),
            F.round(F.col("origin_hist_delay_rate"), 4).alias("origin_hist_delay_rate"),
            F.round(F.col("origin_hist_avg_dep_delay"), 2).alias("origin_hist_avg_dep_delay"),
        )
    )

    _write_gold(df_ml, config, "ml_flight_delay_training")

    # 4. Guardar auditoría de población y particiones
    final_ml_count = df_ml.count()
    split_counts = {row["dataset_split"]: row["count"] for row in df_ml.groupBy("dataset_split").count().collect()}

    audit_rows = [
        ("input_flights_silver", total_silver, "Total de vuelos ingresados desde Silver"),
        ("excluded_cancelled", -excluded_cancelled, "Excluidos por vuelo cancelado"),
        ("excluded_diverted", -excluded_diverted, "Excluidos por vuelo desviado"),
        ("excluded_no_weather", -excluded_no_weather, "Excluidos por aeropuerto fuera del Top 50 sin clima"),
        ("output_ml_training_total", final_ml_count, "Total filas válidas en ml_flight_delay_training"),
        ("split_train_m1_m8", split_counts.get("train", 0), "Partición Train (Enero - Agosto)"),
        ("split_val_m9_m10", split_counts.get("validation", 0), "Partición Validation (Septiembre - Octubre)"),
        ("split_test_m11_m12", split_counts.get("test", 0), "Partición Test (Noviembre - Diciembre)"),
    ]
    df_audit = (
        spark.createDataFrame(audit_rows, ["stage", "row_count", "description"])
        .withColumn("audited_at_utc", F.lit(datetime.now(timezone.utc).isoformat()))
    )
    _write_gold(df_audit, config, "gold_b_audit_log")
    logger.info("Gold B guardado exitosamente (%d filas totales).", final_ml_count)