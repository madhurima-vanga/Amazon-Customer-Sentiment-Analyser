import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame
import re
import matplotlib.pyplot as plt
import gcsfs
import mlflow
import seaborn as sns
import csv, os
from transformers import RobertaTokenizer

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_JSON_Key"

# Set MLflow tracking URI and experiment name
mlflow.set_tracking_uri("http://35.193.202.32:5000")  # Replace with your MLflow URI
mlflow.set_experiment("Roberta Sentiment Analysis Bias Detection")

# Define the GCS path to your file
gcs_test_data_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/test_bias_amazon_reviews.csv"

# Load only 500 rows from the dataset
fs = gcsfs.GCSFileSystem()
with fs.open(gcs_test_data_path) as f:
    sampled_data = pd.read_csv(f, nrows=500)

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the entire model
local_model_path = "mlruns/790700365954151778/9266e0d978184b53a1483ebf98a7213f/artifacts/model/data/model.pth"
model = torch.load(local_model_path, map_location=device)
model.to(device)

# Define the label mapping
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

# Map text labels to numerical labels
sampled_data["mapped_labels"] = sampled_data["final_sentiment"].map(label_mapping)

# Initialize results storage
predictions = []

# Define batch size
batch_size = 16  # Adjust batch size for better memory efficiency

# Process data in batches
for i in range(0, len(sampled_data), batch_size):
    batch = sampled_data.iloc[i:i + batch_size]
    # Tokenize batch
    inputs = tokenizer(
        list(batch["cleaned_text"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512  # Limit sequence length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(batch_predictions)

# Add predictions to the DataFrame
sampled_data["predictions"] = predictions

# Get true labels
labels = sampled_data["mapped_labels"].values

# Define the output CSV file for metrics
csv_file = "compare_model_metrics.csv"

# Start an MLflow run
with mlflow.start_run(run_name="Roberta Model Bias Detection") as run:
    # Use MetricFrame to evaluate metrics across `main_category`
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
        },
        y_true=labels,
        y_pred=predictions,
        sensitive_features=sampled_data["main_category"],  # Group by main_category
    )

    # Log metrics by main_category to MLflow
    print("Metrics by Main Category:")
    print(metric_frame.by_group)

    # Prepare CSV entry for main_category metrics
    model_name = "Roberta"

    csv_rows = []
    for category, metrics in metric_frame.by_group.iterrows():
        for metric_name, value in metrics.items():
            # Clean the category name for compatibility
            clean_category = re.sub(r"[^a-zA-Z0-9_:/-]", "_", category)

            # Add to CSV rows
            csv_rows.append(
                {
                    "Model_Name": model_name,
                    "Category": clean_category,
                    "Metric": metric_name,
                    "Value": value,
                }
            )
            mlflow.log_metric(f"{clean_category}_{metric_name}", value)

    # Write metrics to CSV file
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model_Name", "Category", "Metric", "Value"])
        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerows(csv_rows)

    # Generate visualization
    metrics_df = metric_frame.by_group.reset_index().melt(
        id_vars="main_category", var_name="Metric", value_name="Value"
    )

    # Plot metrics by category
    plt.figure(figsize=(12, 8))
    sns.barplot(data=metrics_df, x="main_category", y="Value", hue="Metric")
    plt.xticks(rotation=90)
    plt.title("Metrics by Main Category")
    plt.tight_layout()
    plt.savefig("bias_metrics_by_category.png")
    plt.close()
    mlflow.log_artifact("bias_metrics_by_category.png", artifact_path="plots")
