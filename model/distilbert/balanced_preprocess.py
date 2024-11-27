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

# Map sentiment labels to numerical values
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}
data["label"] = data["final_sentiment"].map(label_mapping)

# Create subsets for each sentiment class
positive_rows = data[data["label"] == 0].sample(n=100, random_state=42)
negative_rows = data[data["label"] == 1].sample(n=100, random_state=42)
neutral_rows = data[data["label"] == 2].sample(n=100, random_state=42)

# Concatenate subsets to create a balanced dataset
balanced_train_df = pd.concat([positive_rows, negative_rows, neutral_rows])

# Shuffle the dataset to mix the rows
balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert the DataFrame to a Hugging Face Dataset
train_dataset = Dataset.from_pandas(balanced_train_df)

# Use the remaining rows for testing (optional)
test_df = data.iloc[300:600]
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Save tokenized data to GCS
tokenized_dataset.save_to_disk("gs://amazon_sentiment_analysis/distilbert/processed_data/balanced_tokenized_amazon_reviews")
