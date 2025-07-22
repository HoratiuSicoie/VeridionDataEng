
from pyspark.sql import functions as F

def consolidate_clusters(df_with_clusters):
    # Choose representative row_id for each cluster
    rep_ids = df_with_clusters.groupBy("deduplication_id") \
        .agg(F.min("row_id").alias("representative_id"))

    # Join back to original rows to get full info from representative
    consolidated = rep_ids.join(
        df_with_clusters,
        rep_ids["representative_id"] == df_with_clusters["row_id"],
        how="left"
    ).drop(df_with_clusters["deduplication_id"])  # avoid duplication

    return consolidated.select("deduplication_id", "representative_id", *[
        c for c in df_with_clusters.columns if c not in ["deduplication_id", "row_id"]
    ])