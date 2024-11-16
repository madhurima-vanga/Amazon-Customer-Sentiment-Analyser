import logging
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from google.cloud import storage
from io import StringIO
import os
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from tqdm import tqdm
from datetime import datetime
import io

log_dir = os.path.join("logs", "bias")
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name using a timestamp
log_filename = os.path.join(log_dir, f"bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=log_filename,
    filemode="w"  # Append mode
)

# Add console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

# Get the root logger and add the console handler
logging.getLogger().addHandler(console_handler)
# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_JSON_Key"

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the tokenizer for encoding text
logging.info("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_SEQ_LENGTH = 512  # Define a fixed sequence length for padding

# Load the trained model from the .pth file
'''def load_model(filepath):
    try:
        logging.info(f"Loading model from {filepath}...")
        model = torch.load(filepath, map_location=device)
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise'''

# Custom dataset class for test data with padding
class CustomTestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_SEQ_LENGTH):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_text = self.dataframe.iloc[idx]["cleaned_text"]
        label = self.dataframe.iloc[idx]["label"]

        # Tokenize and pad the input text
        encoded_input = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded_input["input_ids"].squeeze()  # Remove extra dimension
        attention_mask = encoded_input["attention_mask"].squeeze()  # Remove extra dimension
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label_tensor

# Function to make predictions
def get_predictions(model, data_loader):
    logging.info("Starting predictions...")
    model.to(device)
    predictions = []
    labels = []
    try:
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                input_ids, attention_mask, label = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                output = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(output.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                labels.extend(label.cpu().numpy())
        logging.info("Prediction complete.")
        return predictions, labels
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# Define evaluation function using Fairlearn
def evaluate_model(y_true, y_pred, sensitive_features):
    logging.info("Evaluating model performance...")
    metrics = {
        "accuracy": accuracy_score,
        "f1_score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro", zero_division=0)
    }
    try:
        metric_frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
        logging.info(f"Metrics by group:\n{metric_frame.by_group}")
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
        logging.info(f"Demographic Parity Difference: {dp_diff}")
        logging.info("Evaluation complete.")
        return metric_frame.by_group
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

# Function to load data from Google Cloud Storage
def load_data_from_gcs(bucket_name, file_path):
    logging.info(f"Loading data from GCS bucket '{bucket_name}', file '{file_path}'...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        data = blob.download_as_text()
        logging.info("Data loaded successfully.")
        return pd.read_csv(StringIO(data))
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        raise

# Function to get a balanced sample of up to 30 samples per category from the dataset
def get_balanced_subset(data, target_column="main_category", max_samples_per_category=30):
    logging.info("Creating balanced subset...")
    try:
        categories = data[target_column].unique()
        balanced_data = pd.concat([
            data[data[target_column] == category].sample(
                min(max_samples_per_category, len(data[data[target_column] == category])),
                random_state=42,
                replace=False
            )
            for category in categories
        ])
        logging.info(f"Balanced subset created with {len(balanced_data)} samples.")
        return balanced_data
    except Exception as e:
        logging.error(f"Error creating balanced subset: {e}")
        raise



def load_model_from_gcs(bucket_name, model_filepath):
    # Initialize GCS client
    client = storage.Client()
    
    # Get the bucket and blob
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(model_filepath)
    
    # Download the blob as bytes
    model_data = blob.download_as_bytes()
    
    # Load the model from the bytes
    model = torch.load(io.BytesIO(model_data))
    return model


# Main code
if __name__ == "__main__":
    try:
        # Load model
        #model_filepath = "mlruns/790700365954151778/9266e0d978184b53a1483ebf98a7213f/artifacts/model/data/model.pth"
        #model = load_model(model_filepath)

        # Load test data from GCS
        bucket_name = "amazon_sentiment_analysis"
        test_file_path = "model/test/test.csv"
        model_filepath = "saved_models/model.pth"
        model = load_model_from_gcs(bucket_name, model_filepath)
        test_data = load_data_from_gcs(bucket_name, test_file_path)
        logging.info("loaded model and test data from GCP")

        # Get a balanced subset
        balanced_test_data = get_balanced_subset(test_data, target_column="main_category", max_samples_per_category=30)

        # Prepare test dataset and DataLoader
        logging.info("Preparing DataLoader...")
        test_dataset = CustomTestDataset(dataframe=balanced_test_data, tokenizer=tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Get sensitive features
        sensitive_column = balanced_test_data["main_category"].fillna("Blank")
        logging.info("Sensitive features loaded.")

        # Get predictions
        predictions, labels = get_predictions(model, test_loader)

        # Evaluate across slices
        metrics_by_slice = evaluate_model(y_true=labels, y_pred=predictions, sensitive_features=sensitive_column)
        logging.info("Model evaluation completed.")
    except Exception as e:
        logging.error(f"Unexpected error in main execution: {e}")
