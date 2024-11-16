import logging
import pandas as pd
from sklearn.utils import resample
from google.cloud import storage
from io import StringIO
import os
from datetime import datetime

log_dir = os.path.join("logs", "bias_mitigation")
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name using a timestamp
log_filename = os.path.join(log_dir, f"bias_mitigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")



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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "wired-glider-441301-e1-11f52bf3c41e.json"
logging.info("Google Cloud credentials set.")

# Step 1: Fetch data from GCS
gcs_path = 'gs://amazon_sentiment_analysis/model/train/train.csv'
logging.info(f"Fetching data from {gcs_path}.")
data = pd.read_csv(gcs_path, storage_options={"token": "your_JSON_Key"})
logging.info("Data fetched successfully.")

# Step 2: Map the sentiment labels to numeric labels
logging.info("Mapping sentiment labels to numeric labels.")
label_mapping = {'positive': 0, 'negative': 1, 'neutral': 2}
data['label'] = data['final_sentiment'].map(label_mapping)
logging.info("Sentiment labels mapped successfully.")

# Step 3: Resample to balance classes within each main category
logging.info("Balancing classes within each main category.")
balanced_data = pd.DataFrame()

for main_category in data['main_category'].unique():
    logging.info(f"Processing main category: {main_category}.")
    category_data = data[data['main_category'] == main_category]
    
    max_label_size = category_data['label'].value_counts().max()
    category_balanced_data = pd.DataFrame()

    for label in category_data['label'].unique():
        label_data = category_data[category_data['label'] == label]
        if len(label_data) < max_label_size:
            logging.info(f"Oversampling label {label} in category {main_category}.")
            label_data = resample(label_data, replace=True, n_samples=max_label_size, random_state=42)
        category_balanced_data = pd.concat([category_balanced_data, label_data])

    balanced_data = pd.concat([balanced_data, category_balanced_data])

logging.info("Classes balanced successfully.")

# Step 4: Calculate sample weights for reweighting
logging.info("Calculating sample weights for reweighting.")
category_counts = balanced_data['main_category'].value_counts()
total_samples = len(balanced_data)

weights = {category: total_samples / count for category, count in category_counts.items()}
balanced_data['sample_weight'] = balanced_data['main_category'].map(weights)
logging.info("Sample weights calculated successfully.")

# Step 5: Upload the revised dataset to GCS
def upload_dataframe_to_gcs(df, bucket_name, destination_blob_name):
    """Uploads a DataFrame as a CSV to GCS without saving locally."""
    logging.info(f"Uploading DataFrame to GCS bucket: {bucket_name}, destination: {destination_blob_name}.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    logging.info(f"Data uploaded to {destination_blob_name} in bucket {bucket_name}.")

bucket_name = "amazon_sentiment_analysis"
upload_dataframe_to_gcs(balanced_data, bucket_name, "bias_mitigation/train_bias_mitigation.csv")

logging.info("Revised dataset created with balanced classes and sample weights, and uploaded to GCP.")
