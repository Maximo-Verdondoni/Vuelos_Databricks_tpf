import logging
import time
from typing import Any, Dict, List
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Excepción específica para HTTP 429 Too Many Requests."""


@retry(
    retry=retry_if_exception_type((requests.RequestException, RateLimitError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=15, min=60, max=120),
    reraise=True,
)
def fetch_json_with_retry(url: str, params: Dict[str, Any] | None = None, timeout: int = 45) -> Any:
    """Ejecuta una petición GET con reintentos y pausa de 60s ante HTTP 429."""
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        logger.warning("HTTP 429 recibido desde %s. Respetando ventana de cuota (%ds)...", url, retry_after)
        time.sleep(retry_after)
        raise RateLimitError(f"Rate limit excedido en {url}")
    response.raise_for_status()
    return response.json()


def fetch_open_meteo_batch(
    base_url: str,
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    hourly_vars: List[str],
    timezone: str = "auto",
    sleep_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Consulta el clima histórico de Open-Meteo para un lote de aeropuertos en una sola llamada.
    Admite múltiples coordenadas separadas por coma según la documentación de la API.
    """
    latitudes = ",".join(str(loc["LATITUDE"]) for loc in locations)
    longitudes = ",".join(str(loc["LONGITUDE"]) for loc in locations)

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    raw_response = fetch_json_with_retry(base_url, params=params)
    time.sleep(sleep_seconds)

    # Si se envía 1 sola ubicación, Open-Meteo devuelve un dict; si son varias, devuelve una lista de dicts
    responses = raw_response if isinstance(raw_response, list) else [raw_response]

    enriched_records = []
    for loc, weather_data in zip(locations, responses):
        enriched_records.append({
            "iata_code": loc["IATA_CODE"],
            "query_latitude": loc["LATITUDE"],
            "query_longitude": loc["LONGITUDE"],
            "raw_payload": weather_data,
        })

    return enriched_records