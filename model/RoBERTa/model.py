from tqdm import tqdm  # Import tqdm for progress bars
import os
import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from google.cloud import storage
from io import StringIO
from datetime import datetime
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_JSON_Key"

log_dir = os.path.join("logs", "model")
os.makedirs(log_dir, exist_ok=True)

# Create a new log file for each run
log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("RoBERTa_Fine_Tuning_Experiment")

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load RoBERTa tokenizer and model for three-class classification
logger.info("Loading tokenizer and model...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3).to(device)
logger.info("Tokenizer and model loaded.")

# Load train and test datasets from Google Cloud Storage
bucket_name = "amazon_sentiment_analysis"

def load_data_from_gcs(bucket_name, file_path):
    """Load CSV data from Google Cloud Storage."""
    logger.info(f"Loading data from {file_path} in bucket {bucket_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text()
    logger.info(f"Data from {file_path} loaded successfully.")
    return pd.read_csv(StringIO(data))

# Paths to train and test files in GCS
train_file_path = "model/train/train.csv"
test_file_path = "model/test/test.csv"

# Load datasets
logger.info("Loading train and test datasets...")
train_df = load_data_from_gcs(bucket_name, train_file_path)
test_df = load_data_from_gcs(bucket_name, test_file_path)

logger.info("Sampling datasets...")
train_df = train_df.sample(n=10000, random_state=42)
test_df = test_df.sample(n=10000, random_state=42)

logger.info("Verifying label columns...")
assert set(train_df['label'].unique()) == {0, 1, 2}, "Label column should contain values 0, 1, and 2"
assert set(test_df['label'].unique()) == {0, 1, 2}, "Label column in test set should contain values 0, 1, and 2"

# Convert DataFrames to Hugging Face Datasets
logger.info("Converting datasets to Hugging Face format...")
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(test_df)

# Preprocess and tokenize the datasets
def preprocess_data(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, padding="max_length", max_length=512)

logger.info("Tokenizing train dataset...")
train_dataset = train_dataset.map(preprocess_data, batched=True)
logger.info("Tokenizing eval dataset...")
eval_dataset = eval_dataset.map(preprocess_data, batched=True)

# Define a custom compute_metrics function for three labels
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize Trainer
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Start MLflow run
with mlflow.start_run():
    logger.info("Logging parameters to MLflow...")
    mlflow.log_params({
        "model_name": "roberta-base",
        "learning_rate": training_args.learning_rate,
        "batch_size": training_args.per_device_train_batch_size,
        "epochs": training_args.num_train_epochs,
        "weight_decay": training_args.weight_decay,
        "num_labels": 3
    })
    
    # Train the model
    logger.info("Starting training...")
    for _ in tqdm(range(training_args.num_train_epochs), desc="Training Progress"):
        trainer.train()
    logger.info("Training completed.")
    
    # Evaluate the model
    logger.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Log evaluation metrics to MLflow
    logger.info("Logging evaluation metrics to MLflow...")
    mlflow.log_metrics(eval_results)
    
    # Save the fine-tuned model and tokenizer
    logger.info("Saving the fine-tuned model and tokenizer...")
    model.save_pretrained("./fine_tuned_roberta")
    tokenizer.save_pretrained("./fine_tuned_roberta")
    
    # Log model and tokenizer as artifacts in MLflow
    logger.info("Logging model to MLflow...")
    mlflow.pytorch.log_model(model, "model")
    logger.info("Model and tokenizer saved and logged to MLflow.")
