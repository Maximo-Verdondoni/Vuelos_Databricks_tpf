import logging
from typing import Any, Dict
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def run_apply_uc_governance(spark: SparkSession, config: Dict[str, Any]) -> None:
    """
    Documenta las tablas y columnas de Gold en Unity Catalog (diccionario de datos).
    """
    if config["environment"] == "local":
        logger.info("Entorno local detectado: omitiendo DDL de Unity Catalog.")
        return

    db_cfg = config["databricks"]
    catalog = db_cfg["catalog"]
    gold_schema = db_cfg["gold_schema"]

    logger.info("Aplicando comentarios de diccionario de datos en %s.%s...", catalog, gold_schema)

    table_comments = {
        "dim_date": "Dimensión calendario de 2015 enriquecida con feriados federales de EE.UU. (Nager.Date).",
        "dim_airline": "Dimensión maestra de aerolíneas comerciales (código IATA y nombre oficial).",
        "dim_airport": "Dimensión role-playing de aeropuertos de EE.UU. con ubicación geográfica y coordenadas.",
        "dim_weather_condition": "Dimensión de categorías climáticas (precipitación/nieve y viento) al momento de salida.",
        "fact_flights": "Tabla de hechos de vuelos domésticos programados en 2015 (incluye cancelados y desviados).",
        "ml_flight_delay_training": "Dataset de entrenamiento para predicción binaria de retraso >= 15 min sin fuga de datos.",
        "gold_b_audit_log": "Auditoría de Gold B: filas de entrada, exclusiones por criterio, total válido y tamaño de cada partición temporal.",
    }

    for table, comment in table_comments.items():
        spark.sql(f"COMMENT ON TABLE {catalog}.{gold_schema}.{table} IS '{comment}'")

    ml_column_comments = {
        "flight_sk": "Clave subrogada única del vuelo (BIGINT). Origen: generado en Silver. Disponibilidad: pre-salida.",
        "flight_date": "Fecha programada del vuelo (DATE). Origen: F1 (YEAR/MONTH/DAY). Disponibilidad: pre-salida.",
        "dataset_split": "Partición temporal: train (meses 1-8), validation (9-10), test (11-12). Disponibilidad: pre-salida.",
        "target_is_delayed_15": "Variable objetivo binaria (1 si arrival_delay_min >= 15, 0 en caso contrario). Origen: F1.",
        "month": "Mes del vuelo (1-12). Origen: F1. Disponibilidad: pre-salida.",
        "day_of_week": "Día de la semana (1=Lunes a 7=Domingo). Origen: F1. Disponibilidad: pre-salida.",
        "is_weekend": "Indicador binario de fin de semana (1 si sábado o domingo). Origen: derivado F1. Disponibilidad: pre-salida.",
        "is_us_holiday": "Indicador binario de feriado federal en EE.UU. Origen: F3 Nager.Date. Disponibilidad: pre-salida.",
        "airline_code": "Código IATA de la aerolínea. Origen: F1. Disponibilidad: pre-salida.",
        "origin_airport_code": "Código IATA de 3 letras del aeropuerto de origen (normalizado en octubre). Origen: F1.",
        "dest_airport_code": "Código IATA de 3 letras del aeropuerto de destino (normalizado en octubre). Origen: F1.",
        "scheduled_dep_hour": "Hora local programada de salida (0-23). Origen: F1 SCHEDULED_DEPARTURE. Disponibilidad: pre-salida.",
        "scheduled_time_min": "Duración programada del vuelo en minutos. Origen: F1 SCHEDULED_TIME. Disponibilidad: pre-salida.",
        "distance_miles": "Distancia de la ruta en millas. Origen: F1 DISTANCE. Disponibilidad: pre-salida.",
        "temperature_2m": "Temperatura a 2m en aeropuerto origen (°C) a la hora de salida. Origen: F2 Open-Meteo.",
        "precipitation": "Precipitación acumulada en la hora previa a la salida (mm). Origen: F2 Open-Meteo.",
        "snowfall": "Nieve acumulada en la hora previa a la salida (cm). Origen: F2 Open-Meteo.",
        "wind_speed_10m": "Velocidad del viento a 10m en origen (km/h). Origen: F2 Open-Meteo.",
        "wind_gusts_10m": "Ráfagas máximas de viento a 10m en origen (km/h). Origen: F2 Open-Meteo.",
        "cloud_cover": "Nubosidad total en origen (%). Origen: F2 Open-Meteo.",
        "airline_hist_delay_rate": "Tasa histórica acumulada de retrasos de la aerolínea hasta el día anterior (sin leakage).",
        "airline_hist_avg_dep_delay": "Retraso promedio de salida histórico de la aerolínea hasta el día anterior (min).",
        "origin_hist_delay_rate": "Tasa histórica acumulada de retrasos del aeropuerto de origen hasta el día anterior.",
        "origin_hist_avg_dep_delay": "Retraso promedio de salida histórico del aeropuerto de origen hasta el día anterior (min).",
    }

    for col_name, col_comment in ml_column_comments.items():
        spark.sql(
            f"ALTER TABLE {catalog}.{gold_schema}.ml_flight_delay_training "
            f"ALTER COLUMN {col_name} COMMENT '{col_comment}'"
        )

    gold_column_comments = {
        "dim_airline": {
            "airline_sk": "Clave subrogada de la aerolínea.",
            "airline_code": "Código IATA de la aerolínea. Origen: F1.",
            "airline_name": "Nombre de la aerolínea. Origen: F1.",
        },
        "dim_airport": {
            "airport_sk": "Clave subrogada del aeropuerto.",
            "iata_code": "Código IATA de 3 letras. Origen: F1.",
            "airport_name": "Nombre del aeropuerto. Origen: F1.",
            "city": "Ciudad. Origen: F1.",
            "state": "Estado de EE.UU. Origen: F1.",
            "country": "País. Origen: F1.",
            "latitude": "Latitud en grados. Origen: F1.",
            "longitude": "Longitud en grados. Origen: F1.",
        },
        "dim_date": {
            "date_sk": "Clave de la fecha con formato yyyyMMdd.",
            "flight_date": "Fecha calendario.",
            "year": "Año.",
            "month": "Mes (1-12).",
            "day": "Día del mes.",
            "day_of_week": "Día de la semana (numérico).",
            "day_name": "Nombre del día.",
            "month_name": "Nombre del mes.",
            "quarter": "Trimestre (1-4).",
            "is_weekend": "Verdadero si es sábado o domingo.",
            "is_us_holiday": "Verdadero si es feriado federal. Origen: F3 Nager.Date.",
            "holiday_name": "Nombre del feriado; nulo si no lo es. Origen: F3 Nager.Date.",
        },
        "dim_weather_condition": {
            "weather_sk": "Clave subrogada de la condición climática; -1 cuando no hay dato.",
            "precipitation_category": "Categoría de precipitación: Seco, Lluvia leve, Lluvia moderada/fuerte o Nieve.",
            "wind_category": "Categoría de viento: leve (<20 km/h), moderado (20-34 km/h) o fuerte (>=35 km/h).",
            "is_weather_observed": "Falso para la fila -1, que agrupa los vuelos sin clima disponible.",
        },
        "fact_flights": {
            "flight_sk": "Clave subrogada del vuelo, generada en Silver.",
            "date_sk": "Clave foránea a dim_date.",
            "airline_sk": "Clave foránea a dim_airline.",
            "origin_airport_sk": "Clave foránea a dim_airport (rol origen).",
            "dest_airport_sk": "Clave foránea a dim_airport (rol destino).",
            "weather_sk": "Clave foránea a dim_weather_condition.",
            "flight_number": "Número de vuelo. Origen: F1.",
            "tail_number": "Matrícula de la aeronave. Origen: F1.",
            "scheduled_dep_hour": "Hora local programada de salida (0-23). Origen: F1.",
            "scheduled_time_min": "Duración programada (min). Origen: F1.",
            "distance_miles": "Distancia de la ruta (millas). Origen: F1.",
            "departure_delay_min": "Retraso de salida (min); negativo si salió antes. Origen: F1.",
            "arrival_delay_min": "Retraso de llegada (min); negativo si llegó antes. Origen: F1.",
            "air_time_min": "Tiempo en el aire (min). Origen: F1.",
            "taxi_out_min": "Tiempo de rodaje a la salida (min). Origen: F1.",
            "taxi_in_min": "Tiempo de rodaje a la llegada (min). Origen: F1.",
            "is_cancelled": "Verdadero si el vuelo se canceló. Origen: F1.",
            "is_diverted": "Verdadero si el vuelo se desvió. Origen: F1.",
            "is_delayed_15": "1 si el retraso de llegada es de 15 min o más.",
            "temperature_2m": "Temperatura a 2m en origen a la hora de salida (°C). Origen: F2 Open-Meteo.",
            "precipitation": "Precipitación en la hora previa a la salida (mm). Origen: F2 Open-Meteo.",
            "snowfall": "Nieve en la hora previa a la salida (cm). Origen: F2 Open-Meteo.",
            "wind_speed_10m": "Velocidad del viento a 10m en origen (km/h). Origen: F2 Open-Meteo.",
        },
        "gold_b_audit_log": {
            "stage": "Etapa: entrada, criterio de exclusión, total válido o partición.",
            "row_count": "Filas de la etapa; negativo en las exclusiones.",
            "description": "Descripción de la etapa.",
            "audited_at_utc": "Momento de la auditoría (UTC).",
        },
    }

    for table, columns in gold_column_comments.items():
        for col_name, col_comment in columns.items():
            spark.sql(
                f"ALTER TABLE {catalog}.{gold_schema}.{table} "
                f"ALTER COLUMN {col_name} COMMENT '{col_comment}'"
            )

    # El acceso de la cátedra se gestiona con Delta Sharing (share_vuelos_tpf), no por código
    logger.info("Diccionario de datos aplicado correctamente en Unity Catalog.")

if __name__ == "__main__":
    from src.common.job_context import run_job
    run_job(run_apply_uc_governance)