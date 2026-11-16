from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "VuelosDatabricksTPI", is_local: bool = True) -> SparkSession:
    """Obtiene la SparkSession activa en Databricks o crea una local con Delta Lake."""
    active_spark = SparkSession.getActiveSession()
    if active_spark is not None:
        return active_spark

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

    return builder.getOrCreate()