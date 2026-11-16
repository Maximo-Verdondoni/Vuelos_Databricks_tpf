import importlib
from datetime import date
import pytest
from pyspark.sql import functions as F
from src.common.spark_session import get_spark_session


@pytest.fixture(scope="module")
def spark():
    return get_spark_session(app_name="TestVuelosTPI", is_local=True)


def test_weather_categorization_handles_missing_and_extremes(spark):
    """Verifica que los vuelos sin clima reciban 'Sin dato' y los extremos se clasifiquen bien."""
    gold_star = importlib.import_module("src.gold.04a_build_gold_star")
    precip_expr, wind_expr = gold_star._categorize_weather_expr()

    data = [
        (False, None, None, None),     # Aeropuerto fuera del Top 50
        (True, 0.0, 0.0, 10.0),        # Seco y viento leve
        (True, 0.0, 2.5, 40.0),        # Nieve y viento fuerte
        (True, 5.0, 0.0, 25.0),        # Lluvia moderada/fuerte y viento moderado
    ]
    df = spark.createDataFrame(data, ["has_weather_data", "precipitation", "snowfall", "wind_speed_10m"])
    res = df.select(precip_expr.alias("p"), wind_expr.alias("w")).collect()

    assert res[0]["p"] == "Sin dato" and res[0]["w"] == "Sin dato"
    assert res[1]["p"] == "Seco" and res[1]["w"] == "Viento leve (<20 km/h)"
    assert res[2]["p"] == "Nieve" and res[2]["w"] == "Viento fuerte (>=35 km/h)"
    assert res[3]["p"] == "Lluvia moderada/fuerte" and res[3]["w"] == "Viento moderado (20-34 km/h)"


def test_historical_window_has_no_data_leakage(spark):
    """Verifica que el cálculo histórico use estrictamente días previos (sin leakage intra-día)."""
    from pyspark.sql import Window

    data = [
        ("AA", date(2015, 1, 1), 1),  # Día 1: 1 retraso -> para el día 1 no hay historial previo (0.0)
        ("AA", date(2015, 1, 2), 0),  # Día 2: debe ver 1/1 = 1.0 del día 1, pero NO su propio 0
        ("AA", date(2015, 1, 3), 0),  # Día 3: debe ver (1+0)/2 = 0.5 de los días 1 y 2
    ]
    df = spark.createDataFrame(data, ["airline_code", "flight_date", "is_delayed_15"])

    w_hist = Window.partitionBy("airline_code").orderBy("flight_date").rowsBetween(Window.unboundedPreceding, -1)
    daily = (
        df.groupBy("airline_code", "flight_date")
        .agg(F.count("*").alias("n"), F.sum("is_delayed_15").alias("d"))
        .withColumn("hist_rate", F.coalesce(F.sum("d").over(w_hist) / F.sum("n").over(w_hist), F.lit(0.0)))
        .orderBy("flight_date")
        .collect()
    )

    assert daily[0]["hist_rate"] == pytest.approx(0.0)
    assert daily[1]["hist_rate"] == pytest.approx(1.0)
    assert daily[2]["hist_rate"] == pytest.approx(0.5)