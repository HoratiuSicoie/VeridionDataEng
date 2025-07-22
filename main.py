from src.spark_session import get_spark_session
from src.data_loader import load_data
from src.deduplication import find_duplicates
from src.cluster_builder import assign_clusters
from src.consolidation import consolidate_clusters

def main():
    spark = get_spark_session("VeridionDeduplication")

    spark._jsc.hadoopConfiguration().set("hadoop.native.lib", "false")


    df = load_data(spark, "Data/cleaned_products.parquet")

    duplicates_df, tfidf_df = find_duplicates(df, block_col="brand", text_col="product_name", threshold=0.6)

    df_with_clusters = assign_clusters(duplicates_df, tfidf_df)
    deduplicated_df = consolidate_clusters(df_with_clusters)

    deduplicated_df_clean = deduplicated_df.drop("tokens", "rawFeatures", "tfidf")
    deduplicated_df_clean.show(truncate=False)
    #deduplicated_df_clean.write.mode("overwrite").parquet("../Data/deduplicated_output.parquet")


if __name__ == "__main__":
    main()