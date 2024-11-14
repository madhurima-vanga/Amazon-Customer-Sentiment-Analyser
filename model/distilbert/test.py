from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from evaluate import load
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load the model and tokenizer from the saved directory
model = AutoModelForSequenceClassification.from_pretrained("./distilbert_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./distilbert_sentiment_model")

from datasets import load_dataset

# Specify the GCS path to your test dataset
gcs_test_data_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/test_amazon_reviews.csv"

# Load and preprocess test dataset from GCS
dataset = load_dataset("csv", data_files=gcs_test_data_path)

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
    # Unpack model outputs
    logits, labels = eval_pred
    # Get the predicted class by choosing the max logit for each sample
    predictions = np.argmax(logits, axis=-1)

    # Calculate each metric using scikit-learn functions
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")

    # Return a dictionary of metrics
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
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

# Perform evaluation
eval_results = trainer.evaluate()
print("Evaluation Metrics:")
for metric_name, metric_value in eval_results.items():
    print(f"{metric_name}: {metric_value}")
