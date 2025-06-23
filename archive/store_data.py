from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, TimestampType


class DataStore:
    def __init__(self):
        self.data = {}

    # initialize the spark session
        spark = SparkSession.builder \
            .appName("bwt") \
            .config("spark.driver.extraClassPath", "/path/to/spark-sql-connector.jar") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .config("spark.databricks.service.token", "dapi5758ebff5e6937faf1915834a6b0cce2") \
            .config("spark.databricks.service.url", "https://adb-490086633420161.1") \
            .getOrCreate()

        # Define the schema for the DataFrame
        schema = StructType([
            StructField("timestamp", TimestampType(), True),
        ])

    def add_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data.get(key)

    def remove_data(self, key):
        del self.data[key]

    def get_all_data(self):
        return self.data

    def clear_data(self):
        self.data.clear()

    def __str__(self):
        return str(self.data)