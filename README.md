# Veridion Product Deduplication

This project addresses the **deduplication of product records** that contain noise, inconsistencies, and redundancy. It includes:

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data cleaning
- 🧠 Local deduplication experiments using notebooks
- ⚡ Scalable deduplication with Apache Spark (in progress)

---

## 📦 Installation

Set up your environment by installing the required dependencies:

```bash
pip install -r requirements.txt


VeridionDataEng/
├── Data/
│   ├── cleaned_products.parquet     # Cleaned product dataset
│   └── veridion_challange_data.parquet  # Initial Dataset
│
├── EDA/
│   ├── eda.ipynb                    # EDA and data cleaning
│   └── deduplication_local.ipynb    # Local deduplication implementation
│
├── src/
│   ├── spark_session.py             # Spark session configuration
│   ├── data_loader.py               # Data reading logic
│   ├── deduplication.py             # TF-IDF and cosine similarity based matching
│   ├── cluster_builder.py           # Clustering logic for assigning deduplication groups
│   └── consolidation.py             # Cluster-level consolidation
│
├── main.py                          # Entry point for Spark deduplication pipeline
├── requirements.txt                 # Required Python packages
└── README.md                        # Project documentation
```
---

## Notes
The implementation inside src/ is designed to scale to large datasets, but still requires further optimization:

Caching intermediate results between stages (especially after TF-IDF)

For now, the Spark implementation runs locally and may need tuning for performance on large datasets.

## 🔧 Future Improvements
Add evaluation metrics for deduplication performance (precision, recall)

Additional similarity metrics (e.g. Jaccard, Levenshtein)

