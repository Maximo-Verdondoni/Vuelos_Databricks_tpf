import os
import sys
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "VuelosDatabricksTPI", is_local: bool = True) -> SparkSession:
    """Obtiene la SparkSession activa en Databricks o crea una local con Delta Lake."""
    active_spark = SparkSession.getActiveSession()
    if active_spark is not None:
        return active_spark

    if is_local:
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        if os.name == "nt" and os.path.exists("C:\\hadoop\\bin\\winutils.exe"):
            os.environ["HADOOP_HOME"] = "C:\\hadoop"
            os.environ["PATH"] = f"C:\\hadoop\\bin;{os.environ.get('PATH', '')}"

    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )

    if is_local:
        builder = (
            builder
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
        )
        return configure_spark_with_delta_pip(builder).getOrCreate()

    return builder.getOrCreate()