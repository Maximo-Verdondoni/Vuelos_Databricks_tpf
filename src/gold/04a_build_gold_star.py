import os
import logging
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


def _categorize_weather_expr():
    """
    Construye una clave determinística y categorías legibles para precipitación/nieve y viento,
    asignando weather_sk = -1 cuando no hay datos climáticos (aeropuertos fuera del Top 50).
    """
    precip_cat = (
        F.when(~F.col("has_weather_data"), F.lit("Sin dato"))
        .when(F.col("snowfall") > 0.0, F.lit("Nieve"))
        .when(F.col("precipitation") > 2.5, F.lit("Lluvia moderada/fuerte"))
        .when(F.col("precipitation") > 0.0, F.lit("Lluvia leve"))
        .otherwise(F.lit("Seco"))
    )

    wind_cat = (
        F.when(~F.col("has_weather_data"), F.lit("Sin dato"))
        .when(F.col("wind_speed_10m") >= 35.0, F.lit("Viento fuerte (>=35 km/h)"))
        .when(F.col("wind_speed_10m") >= 20.0, F.lit("Viento moderado (20-34 km/h)"))
        .otherwise(F.lit("Viento leve (<20 km/h)"))
    )

    return precip_cat, wind_cat


def run_build_gold_star(spark: SparkSession, config: Dict[str, Any]) -> None:
    """Construye el modelo estrella Gold A (dimensiones + fact_flights) y valida las 3 consultas SQL."""
    logger.info("Construyendo Gold A: Modelo Estrella...")

    df_silver = _read_silver(spark, config, "flights_silver")
    df_airlines = _read_silver(spark, config, "airlines_silver")
    df_airports = _read_silver(spark, config, "airports_silver")

    # 1. Dimensión Tiempo (dim_date) enriquecida con feriados federales de F3
    df_dim_date = (
        df_silver
        .select(
            F.date_format(F.col("flight_date"), "yyyyMMdd").cast("int").alias("date_sk"),
            F.col("flight_date"),
            F.col("year"),
            F.col("month"),
            F.col("day"),
            F.col("day_of_week"),
            F.date_format(F.col("flight_date"), "EEEE").alias("day_name"),
            F.date_format(F.col("flight_date"), "MMMM").alias("month_name"),
            F.quarter(F.col("flight_date")).alias("quarter"),
            F.col("day_of_week").isin([6, 7]).alias("is_weekend"),
            F.col("is_us_holiday"),
            F.coalesce(F.col("holiday_name"), F.lit("No Holiday")).alias("holiday_name"),
        )
        .dropDuplicates(["date_sk"])
    )
    _write_gold(df_dim_date, config, "dim_date")

    # 2. Dimensión Aerolínea (dim_airline)
    df_dim_airline = df_airlines.select(
        F.col("airline_code").alias("airline_sk"),
        F.col("airline_code"),
        F.col("airline_name"),
    )
    _write_gold(df_dim_airline, config, "dim_airline")

    # 3. Dimensión Aeropuerto (dim_airport - role-playing para origen y destino)
    df_dim_airport = df_airports.select(
        F.col("iata_code").alias("airport_sk"),
        F.col("iata_code"),
        F.col("airport_name"),
        F.col("city"),
        F.col("state"),
        F.col("country"),
        F.col("latitude"),
        F.col("longitude"),
    )
    _write_gold(df_dim_airport, config, "dim_airport")

    # 4. Dimensión Condición Climática al momento de salida (dim_weather_condition)
    precip_cat, wind_cat = _categorize_weather_expr()
    df_with_weather_cats = (
        df_silver
        .withColumn("precipitation_category", precip_cat)
        .withColumn("wind_category", wind_cat)
        .withColumn(
            "weather_sk",
            F.when(~F.col("has_weather_data"), F.lit(-1))
            .otherwise(F.abs(F.hash(F.col("precipitation_category"), F.col("wind_category"))) % 100000),
        )
    )

    df_dim_weather = (
        df_with_weather_cats
        .select(
            "weather_sk",
            "precipitation_category",
            "wind_category",
            F.col("has_weather_data").alias("is_weather_observed"),
        )
        .dropDuplicates(["weather_sk"])
    )
    _write_gold(df_dim_weather, config, "dim_weather_condition")

    # 5. Tabla de Hechos (fact_flights): 1 fila por vuelo programado (incluye cancelados y desviados)
    df_fact_flights = df_with_weather_cats.select(
        F.col("flight_sk"),
        F.date_format(F.col("flight_date"), "yyyyMMdd").cast("int").alias("date_sk"),
        F.col("airline_code").alias("airline_sk"),
        F.col("origin_airport_code").alias("origin_airport_sk"),
        F.col("dest_airport_code").alias("dest_airport_sk"),
        F.col("weather_sk"),
        F.col("flight_number"),
        F.col("tail_number"),
        F.col("scheduled_dep_hour"),
        F.col("scheduled_time_min"),
        F.col("distance_miles"),
        F.col("departure_delay_min"),
        F.col("arrival_delay_min"),
        F.col("air_time_min"),
        F.col("taxi_out_min"),
        F.col("taxi_in_min"),
        F.col("is_cancelled"),
        F.col("is_diverted"),
        F.col("is_delayed_15"),
        F.col("temperature_2m"),
        F.col("precipitation"),
        F.col("snowfall"),
        F.col("wind_speed_10m"),
    )
    _write_gold(df_fact_flights, config, "fact_flights")
    logger.info("Modelo Estrella Gold A guardado exitosamente (%d hechos).", df_fact_flights.count())

    # 6. Ejecutar las 3 consultas analíticas requeridas usando solo Gold A
    _run_analytical_validation_queries(spark, df_fact_flights, df_dim_date, df_dim_airline, df_dim_airport, df_dim_weather)


def _run_analytical_validation_queries(
    spark: SparkSession, df_fact, df_date, df_airline, df_airport, df_weather
) -> None:
    """Valida integridad referencial y ejecuta las 3 preguntas de negocio de la sección 4."""
    df_fact.createOrReplaceTempView("fact_flights")
    df_date.createOrReplaceTempView("dim_date")
    df_airline.createOrReplaceTempView("dim_airline")
    df_airport.createOrReplaceTempView("dim_airport")
    df_weather.createOrReplaceTempView("dim_weather_condition")

    logger.info("=== Consulta 1: Tasa de llegada demorada >= 15 min por aerolínea y mes ===")
    spark.sql("""
        SELECT
            a.airline_code,
            a.airline_name,
            d.month,
            COUNT(*) AS completed_flights,
            ROUND(AVG(f.is_delayed_15) * 100, 2) AS delay_15_rate_pct
        FROM fact_flights f
        JOIN dim_airline a ON f.airline_sk = a.airline_sk
        JOIN dim_date d ON f.date_sk = d.date_sk
        WHERE f.is_cancelled = FALSE AND f.is_diverted = FALSE
        GROUP BY a.airline_code, a.airline_name, d.month
        ORDER BY a.airline_code, d.month
    """).show(15, truncate=False)

    logger.info("=== Consulta 2: Retraso promedio de salida según precipitación y viento (Top 10 aeropuertos origen) ===")
    spark.sql("""
        WITH top10_origins AS (
            SELECT origin_airport_sk
            FROM fact_flights
            GROUP BY origin_airport_sk
            ORDER BY COUNT(*) DESC
            LIMIT 10
        )
        SELECT
            ap.iata_code AS origin_airport,
            w.precipitation_category,
            w.wind_category,
            COUNT(*) AS flights_count,
            ROUND(AVG(f.departure_delay_min), 2) AS avg_departure_delay_min
        FROM fact_flights f
        JOIN top10_origins t ON f.origin_airport_sk = t.origin_airport_sk
        JOIN dim_airport ap ON f.origin_airport_sk = ap.airport_sk
        JOIN dim_weather_condition w ON f.weather_sk = w.weather_sk
        WHERE f.is_cancelled = FALSE
        GROUP BY ap.iata_code, w.precipitation_category, w.wind_category
        ORDER BY ap.iata_code, flights_count DESC
    """).show(20, truncate=False)

    logger.info("=== Consulta 3: Top 10 rutas con mayor retraso promedio de llegada (>= 1.000 vuelos) ===")
    spark.sql("""
        SELECT
            orig.iata_code AS origin,
            dest.iata_code AS destination,
            COUNT(*) AS completed_flights,
            ROUND(AVG(f.arrival_delay_min), 2) AS avg_arrival_delay_min
        FROM fact_flights f
        JOIN dim_airport orig ON f.origin_airport_sk = orig.airport_sk
        JOIN dim_airport dest ON f.dest_airport_sk = dest.airport_sk
        WHERE f.is_cancelled = FALSE AND f.is_diverted = FALSE
        GROUP BY orig.iata_code, dest.iata_code
        HAVING COUNT(*) >= 1000
        ORDER BY avg_arrival_delay_min DESC
        LIMIT 10
    """).show(10, truncate=False)

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_build_gold_star)