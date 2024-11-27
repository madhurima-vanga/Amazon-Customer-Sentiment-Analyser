from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DistilBertTokenizer
import mlflow, os
import argparse


DEFAULT_DATASETS = {
    "balanced": "gs://amazon_sentiment_analysis/distilbert/processed_data/balanced_tokenized_amazon_reviews",
    "tokenized": "gs://amazon_sentiment_analysis/distilbert/processed_data/tokenized_amazon_reviews"
}

# Set up argument parser
parser = argparse.ArgumentParser(description="DistilBERT Sentiment Analysis Training")
parser.add_argument(
    "--dataset_type",
    type=str,
    choices=["balanced", "tokenized"],
    default="balanced",
    help="Type of dataset to use: 'balanced' (default) or 'tokenized'."
)
args = parser.parse_args()

# Select dataset path based on CLI argument
dataset_path = DEFAULT_DATASETS[args.dataset_type]
print(f"Loading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)


mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("Distilbert-sentiment-analysis-model")

# Load DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./distilbert_sentiment_model")
tokenizer.save_pretrained("./distilbert_sentiment_model")

