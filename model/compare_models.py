import gcsfs
import pandas as pd
from google.cloud import storage
from tabulate import tabulate

# Filepaths in GCS
model_1_metrics_filepath = "gs://amazon_sentiment_analysis/processed_data/compare_model_metrics/compare_model_metrics_roberta.csv"
model_2_metrics_filepath = "gs://amazon_sentiment_analysis/processed_data/compare_model_metrics/compare_model_metrics_distilbert.csv"
model_3_metrics_filepath = "gs://amazon_sentiment_analysis/processed_data/compare_model_metrics/compare_model_metrics_albert.csv"

# Load CSV files from GCP
fs = gcsfs.GCSFileSystem()
with fs.open(model_1_metrics_filepath) as f:
    model_1_metrics_df = pd.read_csv(f, nrows=7)
with fs.open(model_2_metrics_filepath) as f:
    model_2_metrics_df = pd.read_csv(f, nrows=7)
with fs.open(model_3_metrics_filepath) as f:
    model_3_metrics_df = pd.read_csv(f, nrows=7)

# Combine all DataFrames
combined_df = pd.concat([model_1_metrics_df, model_2_metrics_df, model_3_metrics_df], ignore_index=True)

# Check for duplicates and handle them
duplicates = combined_df[combined_df.duplicated(subset=["Model_Name", "Metric"], keep=False)]
if not duplicates.empty:
    print("Duplicate rows causing issues:")
    print(duplicates)

    # Resolve duplicates (aggregate using mean as an example)
    combined_df = combined_df.groupby(["Model_Name", "Metric"], as_index=False).agg({"Value": "mean"})

# Ensure required columns are present
if not {"Model_Name", "Metric", "Value"}.issubset(combined_df.columns):
    raise ValueError("Required columns (Model_Name, Metric, Value) are missing in the input files.")

# Pivot DataFrame to reshape it
pivot_df = combined_df.pivot(index="Model_Name", columns="Metric", values="Value")
pivot_df.reset_index(inplace=True)

# Handle missing values
pivot_df.fillna(0, inplace=True)

# Define weights for metrics
weights = {
    "accuracy": 0.3,
    "f1": 0.3,
    "precision": 0.2,
    "recall": 0.2,
    "auc": 0.05,
    "mcc": 0.05,
    "log_loss": -0.05  # Lower log_loss is better
}

# Normalize weights
total_weight = sum(weights.values())
weights = {metric: weight / total_weight for metric, weight in weights.items()}

# Calculate scores for each model
def calculate_score(row):
    return sum(row.get(metric, 0) * weight for metric, weight in weights.items())

pivot_df["Score"] = pivot_df.apply(calculate_score, axis=1)

# Rank models based on scores
pivot_df = pivot_df.sort_values(by="Score", ascending=False)

# Display the final DataFrame
print(tabulate(pivot_df, headers="keys", tablefmt="pretty"))

# # GCS save logic
# output_filepath = "gs://amazon_sentiment_analysis/combined_metrics.csv"
# try:
#     # Local saving
#     # pivot_df.to_csv(output_filepath, index=False)
    
#     # GCS saving
#     with fs.open(output_filepath, 'w') as f:
#         pivot_df.to_csv(f, index=False)
    
#     print(f"Output saved to {output_filepath}")
# except Exception as e:
#     print(f"Failed to save output: {e}")
