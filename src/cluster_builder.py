from pyspark.sql import DataFrame
from pyspark.sql.functions import col, min as spark_min

def assign_clusters(duplicates_df: DataFrame, original_df: DataFrame) -> DataFrame:
    """
    Assigns cluster IDs to duplicate entries using a union-find approximation implemented in PySpark.
    This is an alternative to GraphFrames' connected components algorithm, and is suitable for local Spark use.

    Parameters
    ----------
    duplicates_df : DataFrame
        A DataFrame containing pairwise duplicate relationships. Must include columns: 'id_a', 'id_b'.
    original_df : DataFrame
        The original DataFrame containing a 'row_id' column used to match against cluster assignments.

    Returns
    -------
    DataFrame
        The original DataFrame enriched with a new column 'deduplication_id', representing the cluster ID each row belongs to.
        If no duplicates are found, 'row_id' is used as the default cluster ID.
    """

    # If no duplicate pairs exist, each record is its own cluster
    if duplicates_df.rdd.isEmpty():
        return original_df.withColumn("deduplication_id", col("row_id"))

    # Create initial vertices from duplicate pairs, with each ID initially its own cluster
    vertices = duplicates_df.select("id_a").union(duplicates_df.select("id_b")).distinct()
    vertices = vertices.withColumnRenamed("id_a", "id")
    vertices = vertices.withColumn("cluster", col("id"))

    # Define directed edges between the duplicate pairs
    edges = duplicates_df.select(
        col("id_a").alias("src"),
        col("id_b").alias("dst")
    )

    # Iteratively propagate the smallest cluster ID through the graph
    converged = False
    while not converged:
        # Propagate cluster ID from src to dst
        updates = edges.join(vertices, edges.src == vertices.id, "left").select(
            col("dst").alias("id"),
            col("cluster")
        )

        # For each node, select the smallest cluster ID from all incoming updates
        new_vertices = vertices.union(updates).groupBy("id").agg(
            spark_min("cluster").alias("cluster")
        )

        # Check for convergence (i.e., no more cluster ID changes)
        converged = new_vertices.join(vertices, "id") \
                                .filter(new_vertices.cluster != vertices.cluster) \
                                .rdd.isEmpty()

        # Update the cluster assignments
        vertices = new_vertices

    # Join the deduplicated cluster info back with the original DataFrame
    return original_df.join(vertices, original_df["row_id"] == vertices["id"], how="left") \
                      .drop("id") \
                      .withColumnRenamed("cluster", "deduplication_id")
