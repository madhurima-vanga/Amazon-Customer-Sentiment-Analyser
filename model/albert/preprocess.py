from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer
import pandas as pd
from google.cloud import storage
from io import StringIO

import os


# Initialize the GCS client
client = storage.Client()

# Define your bucket and file path
bucket_name = "amazon_sentiment_analysis"
blob_path = "processed_data/final_cleaned_data/final_amazon_reviews.csv"

# Download the file content
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)
data_string = blob.download_as_text()

# Load the CSV content into a pandas DataFrame
data = pd.read_csv(StringIO(data_string))

# Load CSV directly from GCS

# Take the first 5000 rows for training and next 1000 rows for testing
train_df = data.iloc[:5000]
test_df = data.iloc[5000:6000]

# Map to Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a single DatasetDict
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Map sentiment labels to numerical values
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
dataset = dataset.map(lambda x: {"label": label_mapping[x["final_sentiment"]]})

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Save tokenized data to GCS
tokenized_dataset.save_to_disk("gs://amazon_sentiment_analysis/distilbert/processed_data/tokenized_amazon_reviews")
