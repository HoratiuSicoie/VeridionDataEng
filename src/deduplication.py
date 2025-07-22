from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import DoubleType
import numpy as np


# ---------------------------------------------
# UDF to compute cosine similarity between two vectors
# ---------------------------------------------
def cosine_sim(v1, v2):
    """
    Computes cosine similarity between two dense vectors.
    Returns 0.0 if either vector has zero norm.

    Parameters
    ----------
    v1 : DenseVector
    v2 : DenseVector

    Returns
    -------
    float
        Cosine similarity value in [0, 1]
    """
    v1, v2 = np.array(v1.toArray()), np.array(v2.toArray())
    dot = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(dot / norm_product) if norm_product != 0 else 0.0

cosine_udf = udf(cosine_sim, DoubleType())


# ---------------------------------------------
# TF-IDF Vectorization Pipeline
# ---------------------------------------------
def tfidf_pipeline(df, input_col='product_name', output_col='tfidf'):
    """
    Transforms a text column into TF-IDF features.

    Parameters
    ----------
    df : DataFrame
        Input Spark DataFrame containing a text column.
    input_col : str
        Name of the column containing text data.
    output_col : str
        Name of the output column to store the TF-IDF vectors.

    Returns
    -------
    DataFrame
        DataFrame with new column `output_col` containing TF-IDF vectors.
    """
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    tokens = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol="tokens", outputCol="rawFeatures", numFeatures=1000)
    featurized = hashingTF.transform(tokens)

    idf = IDF(inputCol="rawFeatures", outputCol=output_col)
    idf_model = idf.fit(featurized)
    tfidf_df = idf_model.transform(featurized)

    return tfidf_df


# ---------------------------------------------
# Pairwise Duplicate Finder using Cosine Similarity
# ---------------------------------------------
def find_duplicates(df, block_col='brand', text_col='product_name', threshold=0.8):
    """
    Identifies near-duplicate products by computing cosine similarity between TF-IDF vectors.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing product data.
    block_col : str
        Column to block on (e.g., 'brand') to limit comparisons to similar groups.
    text_col : str
        Column to compute TF-IDF features from (e.g., 'product_name').
    threshold : float
        Similarity threshold above which two items are considered duplicates.

    Returns
    -------
    tuple (DataFrame, DataFrame)
        - A DataFrame with columns ['id_a', 'id_b', 'similarity'] representing matched pairs.
        - A TF-IDF enhanced version of the input DataFrame (includes 'row_id' and 'tfidf').
    """
    # Add unique row ID for matching and join filtering
    df = df.withColumn("row_id", monotonically_increasing_id())

    # Generate TF-IDF features
    tfidf_df = tfidf_pipeline(df, input_col=text_col, output_col="tfidf")

    # Self-join on the blocking key and row_id < row_id to avoid symmetric duplicates
    df_a = tfidf_df.alias("a")
    df_b = tfidf_df.alias("b")
    joined = df_a.join(
        df_b,
        (col("a." + block_col) == col("b." + block_col)) &  # block within the same group
        (col("a.row_id") < col("b.row_id"))                 # avoid self-joins and duplicates
    )

    # Apply cosine similarity and filter based on the threshold
    result = joined.withColumn(
        "similarity",
        cosine_udf(col("a.tfidf"), col("b.tfidf"))
    ).filter(col("similarity") >= threshold)

    # Output the relevant ID pairs and similarity
    return result.select(
        col("a.row_id").alias("id_a"),
        col("b.row_id").alias("id_b"),
        col("similarity")
    ), tfidf_df
