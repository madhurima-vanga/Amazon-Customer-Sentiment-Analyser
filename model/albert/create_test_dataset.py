import pandas as pd
from google.cloud import storage
from io import StringIO

# Initialize Google Cloud Storage client
client = storage.Client()

# Define GCS bucket and file paths
bucket_name = "amazon_sentiment_analysis"
source_blob_path = "processed_data/final_cleaned_data/final_amazon_reviews.csv"
test_blob_path = "processed_data/final_cleaned_data/test_amazon_reviews.csv"

# Download the file content from GCS
bucket = client.bucket(bucket_name)
source_blob = bucket.blob(source_blob_path)
data_string = source_blob.download_as_text()

# Load the CSV content into a pandas DataFrame
data = pd.read_csv(StringIO(data_string))

# Extract rows 10,000 to 11,000 for the test dataset
test_df = data.iloc[10000:11000]

# Save the extracted rows to a new CSV in-memory
test_csv_buffer = StringIO()
test_df.to_csv(test_csv_buffer, index=False)

# Upload the new test CSV file back to GCS
test_blob = bucket.blob(test_blob_path)
test_blob.upload_from_string(test_csv_buffer.getvalue(), content_type="text/csv")

print(f"Test dataset saved to gs://{bucket_name}/{test_blob_path}")
