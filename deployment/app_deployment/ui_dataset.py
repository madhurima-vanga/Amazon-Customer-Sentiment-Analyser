import pandas as pd
from google.cloud import storage

# Load the dataset from Google Cloud Storage
gcs_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/final_amazon_reviews.csv"
client = storage.Client()
bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

with blob.open("r") as f:
    df = pd.read_csv(f)

# Filter dataset to ensure unique categories and products
filtered_rows = []
categories = df["main_category"].unique()

for category in categories:
    category_df = df[df["main_category"] == category]
    products = category_df["product_title"].unique()[:5]  # Select first 5 products for each category
    
    for product in products:
        product_df = category_df[category_df["product_title"] == product].head(10)  # Select first 10 reviews for each product
        filtered_rows.append(product_df)

    # Stop if we have reached 500 rows
    if len(pd.concat(filtered_rows)) >= 500:
        break

# Combine the filtered rows into a single DataFrame
filtered_df = pd.concat(filtered_rows).head(500)

# Save the filtered dataset to Google Cloud Storage
filtered_dataset_path = "gs://amazon_sentiment_analysis/ui_data/filtered_dataset.csv"
bucket_name, blob_name = filtered_dataset_path.replace("gs://", "").split("/", 1)
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

with blob.open("w") as f:
    filtered_df.to_csv(f, index=False)

print(f"Filtered dataset saved to {filtered_dataset_path}")
