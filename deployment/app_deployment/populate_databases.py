import pandas as pd
import psycopg2
from google.cloud import storage
from datetime import datetime

# Load the dataset from Google Cloud Storage
gcs_path = "gs://amazon_sentiment_analysis/ui_data/filtered_dataset.csv"
client = storage.Client()
bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_name)

with blob.open("r") as f:
    df = pd.read_csv(f)

# Convert timestamp from milliseconds to datetime format
if 'timestamp' in df.columns:
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d %H:%M:%S'))
    print("Converted Timestamps:")
    print(df['timestamp'])
else:
    print("The 'timestamp' column was not found in the dataset.")

# Connect to PostgreSQL
db_connection = psycopg2.connect(
    host="34.56.246.97",
    port="5432",
    dbname="sentiment-analysis",
    user="postgres",
    password=""
)
cursor = db_connection.cursor()

# Insert data into categories table
categories = df["main_category"].unique()
category_id_map = {}

for category in categories:
    cursor.execute("""
        INSERT INTO categories (category_name)
        VALUES (%s)
        RETURNING category_id
    """, (category,))
    category_id = cursor.fetchone()[0]
    category_id_map[category] = category_id

# Insert data into products table
products_inserted = {}

for _, row in df.iterrows():
    category_id = category_id_map[row['main_category']]
    product_key = (row['product_title'], category_id)
    
    if product_key not in products_inserted:
        cursor.execute("""
            INSERT INTO products (category_id, product_title, price, parent_asin)
            VALUES (%s, %s, %s, %s)
            RETURNING product_id
        """, (category_id, row['product_title'], row['price'], row['parent_asin']))
        product_id = cursor.fetchone()[0]
        products_inserted[product_key] = product_id
    else:
        product_id = products_inserted[product_key]

    # Insert data into reviews table
    cursor.execute("""
        INSERT INTO reviews (product_id, cleaned_text, rating, helpful_vote, timestamp, final_sentiment)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (product_id, row['cleaned_text'], row['rating'], row['helpful_vote'], row['timestamp'], row['final_sentiment']))

# Commit the changes and close the connection
db_connection.commit()
cursor.close()
db_connection.close()

print("Data successfully inserted into PostgreSQL database.")
