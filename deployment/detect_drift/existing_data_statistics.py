import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json

# Path to the training data and output file
training_data_path = "gs://amazon_sentiment_analysis/processed_data/final_cleaned_data/final_amazon_reviews.csv"
output_stats_path = "/Users/madhurima/Documents/GitHub/Amazon-Customer-Sentiment-Analyser/detect_drift/training_data_statistics.csv"

# Load the training data
try:
    training_data = pd.read_csv(training_data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    raise

print("Data load done")

# Ensure 'cleaned_text' column exists
if "cleaned_text" not in training_data.columns:
    raise ValueError("The 'cleaned_text' column is missing in training data.")
else:
    print("cleaned_text column is present")

# Extract and clean review text
training_reviews = training_data["cleaned_text"].dropna()

# Tokenize and compute word frequencies
vectorizer = CountVectorizer()
training_matrix = vectorizer.fit_transform(training_reviews)
print("line 31")
training_freq = training_matrix.sum(axis=0).A1
print("line 32")

print("-------------Training Frequencies-----------\n", training_freq)

# Get feature names and store statistics
feature_names = vectorizer.get_feature_names_out()
training_stats = dict(zip(feature_names, training_freq))

# Convert the statistics to a pandas DataFrame
training_stats_df = pd.DataFrame(list(training_stats.items()), columns=["Feature", "Frequency"])

# Write to CSV, replace existing file if it exists
training_stats_df.to_csv(output_stats_path, index=False)

print(f"Training data statistics saved to {output_stats_path}")