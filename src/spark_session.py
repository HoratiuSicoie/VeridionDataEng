import os
os.environ["PYSPARK_PYTHON"] = "python"
from pyspark.sql import SparkSession

def get_spark_session(app_name="VeridionDeduplication"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.jars.repositories", "https://repos.spark-packages.org/") \
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .config("spark.sql.parquet.output.committer.class", "org.apache.parquet.hadoop.ParquetOutputCommitter") \
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1") \
        .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
        .config("spark.sql.shuffle.partitions", "8")\
        .getOrCreate()

    return spark

