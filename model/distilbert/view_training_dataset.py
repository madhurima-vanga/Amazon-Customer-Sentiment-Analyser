from datasets import load_from_disk

# Load the dataset directly from GCS
dataset_path = "gs://amazon_sentiment_analysis/distilbert/processed_data/balanced_tokenized_amazon_reviews/train"
dataset = load_from_disk(dataset_path)

# View the dataset
print("Dataset loaded successfully!")
print(dataset)

# Count the rows for each sentiment
sentiment_counts = dataset.to_pandas()["final_sentiment"].value_counts()

# Print the counts
print("Sentiment Counts:")
print(f"Positive: {sentiment_counts.get('positive', 0)}")
print(f"Negative: {sentiment_counts.get('negative', 0)}")
print(f"Neutral: {sentiment_counts.get('neutral', 0)}")
