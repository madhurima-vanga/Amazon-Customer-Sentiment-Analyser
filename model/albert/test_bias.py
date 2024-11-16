import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame
import re
import matplotlib.pyplot as plt
import gcsfs
import mlflow
import seaborn as sns
import csv,os


mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("ALBERT Sentiment Analysis Bias Detection")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the GCS path to your file
gcs_test_data_path = "gs://amazon_sentiment_analysis/albert/processed_data/final_cleaned_data/test_albert_bias_amazon_reviews.csv"

# Load the CSV file directly from GCS into a DataFrame
fs = gcsfs.GCSFileSystem()
with fs.open(gcs_test_data_path) as f:
    sampled_data = pd.read_csv(f, nrows=500)

# Use environment variables or arguments for model and tokenizer paths
model_dir = os.getenv("MODEL_DIR", "./albert_sentiment_model")
tokenizer_dir = os.getenv("TOKENIZER_DIR", model_dir)

# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# Define the label mapping
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

# Map text labels to numerical labels
sampled_data["mapped_labels"] = sampled_data["final_sentiment"].map(label_mapping)

# Tokenize all texts in the sampled dataset
inputs = tokenizer(
    list(sampled_data["cleaned_text"]), padding=True, truncation=True, return_tensors="pt"
).to(device)

batch_size = 32  # Define the batch size
all_predictions = []

# Loop through data in batches
for i in range(0, len(inputs["input_ids"]), batch_size):
    # Create batch inputs
    batch_inputs = {key: val[i:i + batch_size].to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        # Perform inference on the batch
        batch_outputs = model(**batch_inputs)
    
    # Collect predictions for the batch
    batch_predictions = torch.argmax(batch_outputs.logits, dim=1).cpu().numpy()
    all_predictions.extend(batch_predictions)

# Combine all predictions
predictions = np.array(all_predictions)

# Get true labels
labels = sampled_data["mapped_labels"].values

csv_file = "compare_model_metrics_albert.csv"


# Start an MLflow run
with mlflow.start_run(run_name="Albert Model Bias Detection") as run:
    # Use MetricFrame to evaluate metrics across `main_category`
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
            "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
            "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted")
        },
        y_true=labels,
        y_pred=predictions,
        sensitive_features=sampled_data["main_category"]  # Group by main_category
    )

    # Log metrics by main_category to MLflow
    print("Metrics by Main Category:")
    print(metric_frame.by_group)

    # Prepare CSV entry for main_category metrics
    model_name = "ALBERT"


    csv_rows = []
    for category, metrics in metric_frame.by_group.iterrows():
        for metric_name, value in metrics.items():
            # Clean the category name for compatibility
            clean_category = re.sub(r'[^a-zA-Z0-9_:/-]', '_', category)
            #print(f"{clean_category}: {metric_name} = {value:.4f}")

            # Add to CSV rows
            csv_rows.append({
                "Model_Name": model_name,
                "Category": clean_category,
                "Metric": metric_name,
                "Value": value
            })
            mlflow.log_metric(f"{clean_category}_{metric_name}", value)

    # Write metrics to CSV file
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model_Name", "Category", "Metric", "Value"])
        if not file_exists:
            writer.writeheader()  
        writer.writerows(csv_rows)

    # Generate visualization
    metrics_df = metric_frame.by_group.reset_index().melt(
        id_vars="main_category", 
        var_name="Metric", 
        value_name="Value"
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