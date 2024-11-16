import pandas as pd
from sklearn.model_selection import train_test_split
from google.cloud import storage
from io import StringIO
import os
from datetime import datetime 
import logging

log_dir = os.path.join("logs", "data_split")
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name using a timestamp
log_filename = os.path.join(log_dir, f"data_split_and_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")



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

# Example log message
logging.info("Logging setup complete. Logs will be written to both console and 'logs/data_split_and_upload.log'.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wired-glider-441301-e1-11f52bf3c41e.json"

logging.info("Loading the CSV file from Google Cloud Storage...")
# Step 1: Load the CSV file from GCS
gcs_path = 'gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/final_amazon_reviews.csv'
df = pd.read_csv(gcs_path, storage_options={"token": "your_JSON_Key"})
logging.info("Data successfully fetched.")

# Step 2: Map the sentiment labels to numeric labels
logging.info("Mapping sentiment labels to numeric labels...")
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
df['label'] = df['final_sentiment'].map(label_mapping)
logging.info("Label mapping completed.")

# Step 3: Define the label column and ensure we have three classes
label_column = 'label'
unique_labels = [0, 1, 2]
logging.info(f"Unique labels: {unique_labels}")

# Step 4: Sample a balanced train set
logging.info("Sampling a balanced train set...")
train_dfs = []
for label in unique_labels:
    label_df = df[df[label_column] == label]
    train_label_df = label_df.sample(n=20000 // len(unique_labels), random_state=42)
    train_dfs.append(train_label_df)

train_df = pd.concat(train_dfs)
logging.info("Balanced train set created.")

# Step 5: Remove train samples from the original dataset for test and inference split
logging.info("Removing train samples from the original dataset...")
remaining_df = df.drop(train_df.index)

# Step 6: Calculate the size of the balanced test set
test_size_per_class = int(0.2 * len(remaining_df) / len(unique_labels))
logging.info(f"Test size per class: {test_size_per_class}")

# Step 7: Create a balanced test set
logging.info("Creating a balanced test set...")
test_dfs = []
for label in unique_labels:
    label_df = remaining_df[remaining_df[label_column] == label]
    test_label_df = label_df.sample(n=test_size_per_class, random_state=42)
    test_dfs.append(test_label_df)

balanced_test_df = pd.concat(test_dfs)
logging.info("Balanced test set created.")

# Step 8: Remove test samples from remaining data for the inference set
logging.info("Creating an inference set...")
inference_df = remaining_df.drop(balanced_test_df.index)

# Function to upload DataFrame to GCS directly
def upload_dataframe_to_gcs(df, bucket_name, destination_blob_name):
    """Uploads a DataFrame as a CSV to GCS without saving locally."""
    logging.info(f"Uploading data to {destination_blob_name} in bucket {bucket_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    logging.info(f"Data successfully uploaded to {destination_blob_name}.")

# Upload DataFrames directly to GCS
bucket_name = "amazon_sentiment_analysis"
upload_dataframe_to_gcs(train_df, bucket_name, "model/train/train.csv")
upload_dataframe_to_gcs(balanced_test_df, bucket_name, "model/test/test.csv")
upload_dataframe_to_gcs(inference_df, bucket_name, "model/validate/inference.csv")

logging.info("Data has been split and uploaded into train, test, and inference sets in Google Cloud Storage.")
