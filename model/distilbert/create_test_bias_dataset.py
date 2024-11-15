

import pandas as pd
from google.cloud import storage

# Initialize the Google Cloud Storage client
client = storage.Client()
bucket_name = "amazon_sentiment_analysis"
blob_path = "processed_data/final_cleaned_data/final_amazon_reviews.csv"
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_path)

# Define the target sample size per category and chunksize
sample_size_per_category = 50
chunksize = 100000  # Adjust based on memory capacity
samples = []

# Open blob and read in chunks for efficient processing
with blob.open("rt") as f:
    for chunk in pd.read_csv(f, chunksize=chunksize):
        unique_main_categories = chunk["main_category"].unique()
        unique_stores = chunk["store"].unique()

        # Sample from each unique category in the chunk
        for category in unique_main_categories:
            category_data = chunk[chunk["main_category"] == category]
            # Sample as many rows as available if less than sample_size_per_category
            category_sample = category_data.sample(
                n=min(sample_size_per_category, len(category_data)), replace=False, random_state=42)
            samples.append(category_sample)

        # Sample from each unique store in the chunk
        for store in unique_stores:
            store_data = chunk[chunk["store"] == store]
            # Sample as many rows as available if less than sample_size_per_category
            store_sample = store_data.sample(
                n=min(sample_size_per_category, len(store_data)), replace=False, random_state=42)
            samples.append(store_sample)

# Concatenate all sampled data
sampled_data = pd.concat(samples).drop_duplicates()
print("Sampled dataset shape:", sampled_data.shape)
print("Unique main categories in sampled data:", sampled_data["main_category"].unique())
print("Unique stores in sampled data:", sampled_data["store"].unique())

# Save sampled data to a new file for further analysis
sampled_data.to_csv("test_bias_amazon_reviews.csv", index=False)

