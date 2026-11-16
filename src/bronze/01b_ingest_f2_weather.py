import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from src.common.api_client import fetch_open_meteo_batch

logger = logging.getLogger(__name__)

# Mapeo de excepción puntual detectado en el EDA (ID DOT 10666 -> BLI)
DOT_EXCEPTION_MAP = {"10666": "BLI"}


def _get_target_table_or_path(config: Dict[str, Any], table_name: str) -> str:
    if config["environment"] == "local":
        return str(Path(config["local"]["bronze_path"]) / table_name)
    db_cfg = config["databricks"]
    return f"{db_cfg['catalog']}.{db_cfg['bronze_schema']}.{table_name}"


def _resolve_top_n_airports(raw_dir: str, top_n: int) -> List[Dict[str, Any]]:
    """
    Determina los Top N aeropuertos de origen con más vuelos considerando los 12 meses
    (incluyendo la traducción de códigos numéricos DOT de octubre) y asegura coordenadas válidas.
    """
    airports_df = pd.read_csv(f"{raw_dir.rstrip('/')}/airports.csv")
    flights_df = pd.read_csv(
        f"{raw_dir.rstrip('/')}/flights.csv",
        usecols=["MONTH", "AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT"],
        dtype=str,
    )

    # 1. Imputar coordenadas faltantes por promedio de estado (validado en EDA para ECP, PBG, UST)
    airports_df["LATITUDE"] = airports_df["LATITUDE"].fillna(
        airports_df.groupby("STATE")["LATITUDE"].transform("mean")
    )
    airports_df["LONGITUDE"] = airports_df["LONGITUDE"].fillna(
        airports_df.groupby("STATE")["LONGITUDE"].transform("mean")
    )

    # 2. Traducir IDs numéricos de octubre a códigos IATA de 3 letras
    ref_flights = flights_df[(flights_df["MONTH"] != "10") & flights_df["ORIGIN_AIRPORT"].str.isalpha()]
    oct_flights = flights_df[flights_df["MONTH"] == "10"]

    route_ref = (
        ref_flights.groupby(["AIRLINE", "FLIGHT_NUMBER"])["ORIGIN_AIRPORT"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    dot_to_iata = dict(DOT_EXCEPTION_MAP)
    for row in oct_flights[["AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT"]].drop_duplicates().itertuples():
        if row.ORIGIN_AIRPORT not in dot_to_iata:
            mapped = route_ref.get((row.AIRLINE, row.FLIGHT_NUMBER))
            if mapped:
                dot_to_iata[row.ORIGIN_AIRPORT] = mapped

    flights_df["ORIGIN_CLEAN"] = flights_df["ORIGIN_AIRPORT"].replace(dot_to_iata)
    top_codes = flights_df["ORIGIN_CLEAN"].value_counts().head(top_n).index.tolist()

    top_airports = airports_df[airports_df["IATA_CODE"].isin(top_codes)][
        ["IATA_CODE", "LATITUDE", "LONGITUDE"]
    ].to_dict(orient="records")

    return top_airports


def run_ingest_f2_weather(spark: SparkSession, config: Dict[str, Any], batch_id: str | None = None) -> None:
    """
    Ingesta respuestas crudas JSON de Open-Meteo para los Top N aeropuertos de origen
    en lotes configurables, registrando trazabilidad y errores de ingesta.
    """
    is_local = config["environment"] == "local"
    raw_dir = config["local"]["raw_path"] if is_local else config["databricks"]["volume_raw_path"]
    batch_id = batch_id or str(uuid.uuid4())
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    pipe_cfg = config["pipeline"]
    meteo_cfg = config["apis"]["open_meteo"]

    top_airports = _resolve_top_n_airports(raw_dir, pipe_cfg["top_n_airports"])
    logger.info("Consultando clima Open-Meteo para %d aeropuertos...", len(top_airports))

    batch_size = meteo_cfg["batch_size_locations"]
    weather_records = []
    error_records = []

    for i in range(0, len(top_airports), batch_size):
        chunk = top_airports[i : i + batch_size]
        iata_list = [c["IATA_CODE"] for c in chunk]
        logger.info("Lote %d: consultando %s", (i // batch_size) + 1, iata_list)

        try:
            results = fetch_open_meteo_batch(
                base_url=meteo_cfg["base_url"],
                locations=chunk,
                start_date=pipe_cfg["start_date"],
                end_date=pipe_cfg["end_date"],
                hourly_vars=meteo_cfg["hourly_variables"],
                timezone=meteo_cfg["timezone"],
                sleep_seconds=meteo_cfg["sleep_seconds"],
            )

            for item in results:
                weather_records.append({
                    "iata_code": item["iata_code"],
                    "query_latitude": float(item["query_latitude"]),
                    "query_longitude": float(item["query_longitude"]),
                    "raw_json": json.dumps(item["raw_payload"]),
                    "_ingestion_ts": ingestion_ts,
                    "_batch_id": batch_id,
                    "_source_url": meteo_cfg["base_url"],
                })

        except Exception as exc:
            logger.error("Fallo al consultar lote %s: %s", iata_list, str(exc))
            for loc in chunk:
                error_records.append({
                    "batch_id": batch_id,
                    "ingestion_ts": ingestion_ts,
                    "iata_code": loc["IATA_CODE"],
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc)[:500],
                })

    if weather_records:
        schema = StructType([
            StructField("iata_code", StringType(), False),
            StructField("query_latitude", DoubleType(), True),
            StructField("query_longitude", DoubleType(), True),
            StructField("raw_json", StringType(), False),
            StructField("_ingestion_ts", StringType(), False),
            StructField("_batch_id", StringType(), False),
            StructField("_source_url", StringType(), False),
        ])
        df_weather = spark.createDataFrame(weather_records, schema=schema)
        target = _get_target_table_or_path(config, "weather_raw")
        writer = df_weather.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
        writer.save(target) if is_local else writer.saveAsTable(target)
        logger.info("Weather Bronze guardado exitosamente (%d aeropuertos).", len(weather_records))

    if error_records:
        df_err = spark.createDataFrame(error_records)
        err_target = _get_target_table_or_path(config, "ingestion_errors_f2")
        err_writer = df_err.write.format("delta").mode("append")
        err_writer.save(err_target) if is_local else err_writer.saveAsTable(err_target)