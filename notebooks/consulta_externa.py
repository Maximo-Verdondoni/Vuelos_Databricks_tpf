"""Prueba de acceso externo a la capa Gold con un service principal de solo lectura.

Requisitos: pip install databricks-sql-connector databricks-sdk python-dotenv
"""
import os
from databricks import sql
from databricks.sdk.core import Config, oauth_service_principal

from dotenv import load_dotenv
load_dotenv()

SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]  # ej: adb-123456789.10.azuredatabricks.net
HTTP_PATH = os.environ["DATABRICKS_HTTP_PATH"]              # ej: /sql/1.0/warehouses/abc123
CLIENT_ID = os.environ["DATABRICKS_CLIENT_ID"]
CLIENT_SECRET = os.environ["DATABRICKS_CLIENT_SECRET"]


def credential_provider():
    config = Config(
        host=f"https://{SERVER_HOSTNAME}",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    return oauth_service_principal(config)


QUERY = """
SELECT a.airline_name, COUNT(*) AS vuelos, ROUND(AVG(f.is_delayed_15) * 100, 2) AS pct_retraso
FROM vuelos_dev.gold.fact_flights f
JOIN vuelos_dev.gold.dim_airline a ON f.airline_sk = a.airline_sk
WHERE NOT f.is_cancelled AND NOT f.is_diverted
GROUP BY a.airline_name
ORDER BY pct_retraso DESC
"""

with sql.connect(
    server_hostname=SERVER_HOSTNAME,
    http_path=HTTP_PATH,
    credentials_provider=credential_provider,
) as conn:
    with conn.cursor() as cursor:
        cursor.execute(QUERY)
        for row in cursor.fetchall():
            print(row)