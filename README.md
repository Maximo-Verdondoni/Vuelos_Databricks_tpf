# ✈️ Vuelos Databricks TPF — Pipeline Medallion en Azure Databricks

Pipeline de datos de punta a punta sobre los **vuelos domésticos de EE.UU. 2015**, enriquecidos con **clima horario** (Open-Meteo) y **feriados federales** (Nager.Date). Implementa la arquitectura **Medallion (Bronze → Silver → Gold)** con **Delta Lake** y **Unity Catalog**, orquestada con un **Databricks Job** sobre cómputo serverless.

> TP final integrador — Procesamiento de Big Data · Licenciatura en Ciencia de Datos · Universidad Austral

---

## 📌 Objetivos

- Ingestar tres fuentes heterogéneas (CSV + 2 APIs REST) con trazabilidad y registro de errores.
- Limpiar, tipificar y unificar los datos en una **visión empresarial única** (`flights_silver`).
- Publicar dos productos Gold:
  - **Gold A — Modelo estrella** para analítica SQL (`fact_flights` + 4 dimensiones).
  - **Gold B — Dataset de ML** para predecir retraso de llegada ≥ 15 min, **sin fuga de información**.
- Aplicar **gobernanza** en Unity Catalog (diccionario de datos y permisos de lectura).
- Que el mismo código corra **en local** (PySpark + delta-spark) y **en Databricks** sin cambios.

---

## 🗂️ Fuentes de datos

| ID | Fuente | Formato | Contenido |
|---|---|---|---|
| F1 | Kaggle — *2015 Flight Delays and Cancellations* | CSV | `flights.csv` (~5,8 M filas), `airlines.csv`, `airports.csv` |
| F2 | Open-Meteo (Historical Weather API) | JSON | Clima horario 2015 de los **Top 50 aeropuertos de origen** |
| F3 | Nager.Date (Public Holidays API) | JSON | Feriados federales de EE.UU. 2015 |

Variables horarias de F2: `temperature_2m`, `precipitation`, `snowfall`, `wind_speed_10m`, `wind_gusts_10m`, `cloud_cover`.

---

## 🏗️ Arquitectura

```
                 ┌──────────────── BRONZE (copia fiel) ────────────────┐
F1 CSV (Volume) ─┤ flights_raw · airlines_raw · airports_raw            │
F2 Open-Meteo  ──┤ weather_raw (JSON crudo por aeropuerto)              │
F3 Nager.Date  ──┤ holidays_raw (JSON crudo)                            │
                 │ ingestion_errors_f1 · ingestion_errors_f2            │
                 └──────────────────────────┬──────────────────────────┘
                                            ▼
                 ┌──────────────── SILVER (limpio y conformado) ────────┐
                 │ airlines_silver · airports_silver · holidays_silver  │
                 │ dot_airport_mapping · weather_silver                 │
                 │ flights_clean_silver ──► flights_silver (unificada)  │
                 │ silver_quality_metrics                               │
                 └──────────────────────────┬──────────────────────────┘
                                            ▼
                 ┌──────────────── GOLD (consumo) ──────────────────────┐
                 │ A) dim_date · dim_airline · dim_airport ·            │
                 │    dim_weather_condition · fact_flights              │
                 │ B) ml_flight_delay_training · gold_b_audit_log       │
                 └──────────────────────────────────────────────────────┘
```

Catálogo Unity Catalog: **`vuelos_dev`** con esquemas `bronze`, `silver`, `gold`.
Los CSV crudos se suben al Volume **`/Volumes/vuelos_dev/bronze/raw_landing`**.

---

## 📁 Estructura del repositorio

```
Vuelos_Databricks_tpf/
├── conf/
│   └── parameters.yaml          # Configuración (entorno, rutas, APIs, splits ML)
├── notebooks/
│   └── _job_runner.py           # Punto de entrada único del Job (local y Databricks)
└── src/
    ├── common/
    │   ├── api_client.py        # GET con reintentos (tenacity) y manejo de HTTP 429
    │   ├── config_loader.py     # Carga YAML + overrides
    │   └── spark_session.py     # SparkSession local con Delta Lake
    ├── bronze/
    │   ├── 01a_ingest_f1_kaggle.py
    │   ├── 01b_ingest_f2_weather.py
    │   └── 01c_ingest_f3_holidays.py
    ├── silver/
    │   ├── 02a_clean_dimensions.py
    │   ├── 02b_clean_weather.py
    │   ├── 02c_clean_flights.py
    │   └── 03_build_silver_unified.py
    └── gold/
        ├── 04a_build_gold_star.py
        ├── 04b_build_gold_ml.py
        └── 05_apply_uc_governance.py
```

> Cada script de capa **solo define funciones** (`run_...`). Quien las ejecuta es `_job_runner.py`.

---

## ▶️ Cómo ejecutar

### En Databricks (Job)

El Job **`job_vuelos_medallion_tpi`** tiene 10 tareas de tipo **Script de Python** que apuntan todas a `notebooks/_job_runner.py`. Cada tarea recibe dos parámetros: el **módulo** y la **función** a ejecutar.

| Tarea | Depende de | Parámetros |
|---|---|---|
| `01a_bronze_f1` | — | `["src.bronze.01a_ingest_f1_kaggle", "run_ingest_f1"]` |
| `01b_bronze_f2` | — | `["src.bronze.01b_ingest_f2_weather", "run_ingest_f2_weather"]` |
| `01c_bronze_f3` | — | `["src.bronze.01c_ingest_f3_holidays", "run_ingest_f3_holidays"]` |
| `02a_silver_dims` | 01a, 01c | `["src.silver.02a_clean_dimensions", "run_clean_dimensions"]` |
| `02b_silver_weather` | 01b | `["src.silver.02b_clean_weather", "run_clean_weather"]` |
| `02c_silver_flights` | 01a | `["src.silver.02c_clean_flights", "run_clean_flights"]` |
| `03_silver_unified` | 02a, 02b, 02c | `["src.silver.03_build_silver_unified", "run_build_silver_unified"]` |
| `04a_gold_star` | 03 | `["src.gold.04a_build_gold_star", "run_build_gold_star"]` |
| `04b_gold_ml` | 03 | `["src.gold.04b_build_gold_ml", "run_build_gold_ml"]` |
| `05_gold_governance` | 04a, 04b | `["src.gold.05_apply_uc_governance", "run_apply_uc_governance"]` |

Requisitos previos en Databricks:
1. Catálogo `vuelos_dev` con esquemas `bronze`, `silver`, `gold`.
2. Volume `vuelos_dev.bronze.raw_landing` con `flights.csv`, `airlines.csv` y `airports.csv`.
3. Carpeta Git del repo sincronizada (pull) con la rama `main`.

Duración de referencia de una corrida completa: **~5 min 20 s** (serverless).

### En local

Requisitos: Python 3.10+, Java 11/17, `pyspark`, `delta-spark`, `pandas`, `pyyaml`, `requests`, `tenacity`.
Los CSV van en la ruta `local.raw_path` del YAML (por defecto `data/raw/`).

Desde la raíz del repo, una etapa por vez:

```bash
python notebooks/_job_runner.py src.bronze.01a_ingest_f1_kaggle run_ingest_f1
python notebooks/_job_runner.py src.bronze.01b_ingest_f2_weather run_ingest_f2_weather
python notebooks/_job_runner.py src.bronze.01c_ingest_f3_holidays run_ingest_f3_holidays
python notebooks/_job_runner.py src.silver.02a_clean_dimensions run_clean_dimensions
python notebooks/_job_runner.py src.silver.02b_clean_weather run_clean_weather
python notebooks/_job_runner.py src.silver.02c_clean_flights run_clean_flights
python notebooks/_job_runner.py src.silver.03_build_silver_unified run_build_silver_unified
python notebooks/_job_runner.py src.gold.04a_build_gold_star run_build_gold_star
python notebooks/_job_runner.py src.gold.04b_build_gold_ml run_build_gold_ml
```

> El paso 05 (gobernanza) se omite solo en local: no hay Unity Catalog.

---

## ⚙️ Cómo funciona `_job_runner.py`

1. **Busca la raíz del repo** (la carpeta que contiene `conf/parameters.yaml` y `src/`). En Databricks serverless `__file__` no existe, así que prueba con `__file__`, `sys.argv[0]` y la carpeta actual.
2. **Detecta el entorno** con la variable `DATABRICKS_RUNTIME_VERSION`:
   - Databricks → usa la SparkSession activa y fuerza `environment = databricks` (lee del Volume y escribe en Unity Catalog).
   - Local → crea una SparkSession con Delta Lake y escribe en rutas locales.
3. **Lee los widgets del Job** si existen (`catalog`, `start_date`, `end_date`); si no, usa el YAML.
4. **Importa el módulo** con `importlib` y ejecuta la función `run_...(spark, config)`.

Si una etapa falla, la excepción se propaga y la tarea queda **en rojo** con el error real.

---

## 🧹 Decisiones de calidad de datos (resumen)

- **Bronze**: todo se lee como `string` (copia fiel), modo `PERMISSIVE` con `_corrupt_record`, y columnas de trazabilidad `_ingestion_ts`, `_batch_id`, `_source_file` / `_source_url`.
- **Aeropuertos sin coordenadas** (ECP, PBG, UST): imputación por promedio del estado.
- **Octubre 2015**: los aeropuertos vienen como IDs numéricos DOT de 5 dígitos. Se traducen a IATA con la **moda** del origen para cada `(aerolínea, nro. de vuelo)` en los otros meses, más la excepción manual `10666 → BLI`.
- **Clima**: filtros de rangos físicos (temperatura entre −60 y 60 °C, precipitación ≥ 0, nubosidad entre 0 y 100 %) y deduplicación por aeropuerto-fecha-hora.
- **Variable objetivo** `is_delayed_15`: 1 si `arrival_delay_min ≥ 15`, nula para vuelos cancelados o desviados.

---

## 🤖 Dataset de ML sin data leakage

- Solo variables **conocidas antes de la salida programada** (calendario, ruta, programación, clima en origen).
- Variables históricas (tasa de retraso y retraso medio de salida por aerolínea y por aeropuerto) calculadas con ventanas **hasta el día anterior** (`rowsBetween(unboundedPreceding, -1)`).
- Partición **temporal**: train = meses 1–8 · validation = 9–10 · test = 11–12.
- Auditoría de población en `gold_b_audit_log` (filas que entran, excluidas por criterio y tamaño de cada partición).

---

## 🔐 Gobernanza

`05_apply_uc_governance.py` agrega comentarios de tabla y de columna (diccionario de datos de `ml_flight_delay_training`) y otorga `USE CATALOG`, `USE SCHEMA` y `SELECT` sobre `vuelos_dev.gold` al usuario de la cátedra.

Verificación:

```sql
SHOW TABLES IN vuelos_dev.silver;
SHOW TABLES IN vuelos_dev.gold;
SELECT * FROM vuelos_dev.silver.silver_quality_metrics;
SELECT * FROM vuelos_dev.gold.gold_b_audit_log;
SHOW GRANTS ON SCHEMA vuelos_dev.gold;
```

---

## 👥 Integrantes

- Castillo, Juan
- Leoni, Juan Martín
- Rubiolo, Facundo
- Máximo, Verdondoni
