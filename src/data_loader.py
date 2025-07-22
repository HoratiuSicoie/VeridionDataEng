from pyspark.sql import SparkSession, DataFrame

def load_data(spark: SparkSession, path: str) -> DataFrame:
    """
    Load a dataset from a Parquet file into a Spark DataFrame.

    Parameters:
    ----------
    spark : SparkSession
        An active Spark session used to read the data.
    path : str
        Path to the input Parquet file.

    Returns:
    -------
    DataFrame
        A Spark DataFrame containing the data from the specified Parquet file.
    """
    return spark.read.parquet(path)
