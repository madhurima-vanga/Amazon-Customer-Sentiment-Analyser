from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from evaluate import load
import numpy as np
from datasets import DatasetDict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, matthews_corrcoef, log_loss
import mlflow
from scipy.special import softmax
import csv
import os

# Set MLflow tracking URI to the local instance
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DistilBERT Sentiment Analysis Evaluation")

# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("./distilbert_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./distilbert_sentiment_model")

from datasets import load_dataset

# Specify the GCS path to your test dataset
gcs_test_data_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/test_amazon_reviews.csv"

# Load and preprocess test dataset from GCS
dataset = load_dataset("csv", data_files=gcs_test_data_path)

dataset = DatasetDict({"test": dataset["train"]})

label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

# Map text labels to numerical labels in the `final_sentiment` column
def map_labels(example):
    example["labels"] = label_mapping[example["final_sentiment"]]
    return example

# Apply label mapping to the dataset and rename the column
dataset = dataset.map(map_labels)
dataset = dataset.remove_columns(["final_sentiment"])

def preprocess_function(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, padding="max_length")

# Tokenize the test dataset
tokenized_test_dataset = dataset.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply softmax to get probabilities for AUC and log loss
    probabilities = softmax(logits, axis=1)
    # Get class predictions for accuracy, precision, recall, and F1
    predictions = np.argmax(logits, axis=-1)

    # Calculate standard classification metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    # Calculate AUC (One-vs-Rest for multi-class classification) using probabilities
    try:
        auc = roc_auc_score(labels, probabilities, multi_class="ovr", average="weighted")
    except ValueError:
        auc = None  # Handle case where AUC can't be computed

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, predictions)
    
    # Calculate Log Loss using probabilities
    logloss = log_loss(labels, probabilities)
    
    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc if auc is not None else 0.0,
        "mcc": mcc,
        "log_loss": logloss
    }





training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_output",
    per_device_eval_batch_size=8,
)

# Initialize Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset, 
    compute_metrics=compute_metrics
)

# Start an MLflow run to log metrics
with mlflow.start_run(run_name="Model Evaluation"):
    # Use predict to get logits and labels
    predictions = trainer.predict(tokenized_test_dataset["test"])
    
    # Log each metric to MLflow
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")
        mlflow.log_metric(metric_name, metric_value)

    # Compute confusion matrix and save as artifact
    conf_matrix = confusion_matrix(predictions.label_ids, np.argmax(predictions.predictions, axis=-1))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save and log the confusion matrix plot to MLflow
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt_path = "plots/confusion_matrix.png"
    plt.savefig(plt_path)
    plt.close()
    mlflow.log_artifact(plt_path, artifact_path="plots")

    # Prepare CSV entry
    model_name = "DistilBERT"  


    # Create or append to the CSV file
    csv_file = "compare_model_metrics.csv"

    csv_rows = []
    for metric_name, metric_value in metrics.items():
        csv_rows.append({
            "Model_Name": model_name,
            "Metric": metric_name,
            "Value": metric_value
        })

    # Write to CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model_Name", "Metric", "Value"])
        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerows(csv_rows)