import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame
import re
import gcsfs
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow URI
mlflow.set_experiment("Sentiment Analysis Bias Detection")

# Define the GCS path to your file
gcs_test_data_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/test_bias_amazon_reviews.csv"

# Load the CSV file directly from GCS into a DataFrame
fs = gcsfs.GCSFileSystem()
with fs.open(gcs_test_data_path) as f:
    sampled_data = pd.read_csv(f, nrows=500)

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./distilbert_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./distilbert_sentiment_model")

# Define the label mapping
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

# Map text labels to numerical labels
sampled_data["mapped_labels"] = sampled_data["final_sentiment"].map(label_mapping)

# Tokenize all texts in the sampled dataset
inputs = tokenizer(
    list(sampled_data["cleaned_text"]), padding=True, truncation=True, return_tensors="pt"
)

# Perform inference with no_grad
with torch.no_grad():
    outputs = model(**inputs)

# Get predicted class labels
predictions = torch.argmax(outputs.logits, dim=1).numpy()

# Get true labels
labels = sampled_data["mapped_labels"].values

# Start an MLflow run
with mlflow.start_run(run_name="Distilbert Model Bias Detection"):

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
    for category, metrics in metric_frame.by_group.iterrows():
        # Clean the category name for MLflow compatibility
        clean_category = re.sub(r'[^a-zA-Z0-9_:/-]', '_', category)
        #print(f"{category}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, "
         #     f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

        for metric_name, value in metrics.items():
            # Log each metric for each cleaned category name
            mlflow.log_metric(f"{clean_category}_{metric_name}", value)


    mlflow.log_artifact("metrics_by_main_category.csv")



'''# Use MetricFrame to evaluate metrics across `rating`
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted")
    },
    y_true=labels,
    y_pred=predictions,
    sensitive_features=sampled_data["rating"]  # Group by rating
)

# Display metrics by rating
print("Metrics by Rating:")
print(metric_frame.by_group)'''