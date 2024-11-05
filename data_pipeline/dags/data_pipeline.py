import os
import pandas as pd
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = "./data"
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")

MERGED_CSV = os.path.join(DATA_DIR, "merged_data.csv")
HTML_CLEANED_CSV = os.path.join(DATA_DIR, "html_cleaned_data.csv")
FILTERED_CSV = os.path.join(DATA_DIR, "filtered_data.csv")
PREPROCESSED_CSV = os.path.join(DATA_DIR, "preprocessed_reviews.csv")

FINAL_OUTPUT_CSV = os.path.join(DATA_DIR, "final_amazon_reviews.csv")

# Initialize sentiment models


# Data Loading Functions
def load_reviews_dataset():
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                           "raw_review_Appliances", split="full", trust_remote_code=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset.to_pandas().to_csv(REVIEWS_CSV, index=False)
    logger.info("Reviews dataset downloaded and saved to reviews.csv")


def load_metadata_dataset():
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)

    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                           "raw_meta_Appliances", split="full", trust_remote_code=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset.to_pandas().to_csv(METADATA_CSV, index=False)
    logger.info("Metadata dataset downloaded and saved to metadata.csv")


def merge_data():
    """Merge reviews and metadata, drop unnecessary columns, and save."""
    reviews = pd.read_csv(REVIEWS_CSV, low_memory=False)
    metadata = pd.read_csv(METADATA_CSV, low_memory=False)
    reviews = reviews.rename(columns={'title': 'review_title'})
    metadata = metadata.rename(columns={'title': 'product_title'})
    merged_data = pd.merge(reviews, metadata, on='parent_asin', how='inner')
    existing_columns = ['text', 'rating',
                        'product_title', 'store', 'price', 'helpful_vote']
    filtered_data = merged_data.dropna(subset=['text', 'rating'])[
        existing_columns]
    filtered_data.to_csv(MERGED_CSV, index=False)
    logger.info("Data merged and saved to merged_data.csv")

# Data Cleaning and Preprocessing Functions


def remove_html_tags():
    """Clean text by removing HTML tags and save to CSV."""

    from bs4 import BeautifulSoup

    def clean_text(text):
        """Remove HTML tags from text."""
        if isinstance(text, str) and text.strip() != "":
            text = BeautifulSoup(text, "html.parser").get_text()
        return text.lower()

    df = pd.read_csv(MERGED_CSV).head(10000)
    df['cleaned_text'] = df['text'].apply(clean_text)
    df.to_csv(HTML_CLEANED_CSV, index=False)
    logger.info("HTML tags removed and data saved to html_cleaned_data.csv")


def remove_non_string_rows(df, column_name):
    """Remove rows where specified column is not a string."""
    return df[df[column_name].apply(lambda x: isinstance(x, str))]


def remove_non_string_reviews():
    """Load, clean, and save the preprocessed data by removing non-string rows."""
    df = pd.read_csv(PREPROCESSED_CSV)
    df = remove_non_string_rows(df, 'cleaned_text')
    df.to_csv(PREPROCESSED_CSV, index=False)
    logger.info("Data cleaning completed and saved to preprocessed_reviews.csv")


def filter_urls_paths():
    """Filter out rows containing URLs or file paths and save the final cleaned data."""
    url_pattern = r'^https?://'
    file_path_pattern = r'^[\w,\s-]+\.[A-Za-z]{3}$'
    df = pd.read_csv(HTML_CLEANED_CSV)
    filtered_df = df[~df['cleaned_text'].str.contains(url_pattern, na=False)]
    filtered_df = filtered_df[~filtered_df['cleaned_text'].str.contains(
        file_path_pattern, na=False)]
    # Remove the original 'text' column
    filtered_df.drop(columns=['text'], inplace=True)
    filtered_df.to_csv(PREPROCESSED_CSV, index=False)
    logger.info(
        "URLs and file paths filtered, final data saved to preprocessed_reviews.csv")


# Individual Sentiment Analysis Functions
def apply_heuristic_sentiment():
    """Apply heuristic sentiment based on rating and save results."""
    df = pd.read_csv(PREPROCESSED_CSV)
    df['heuristic_sentiment'] = df.apply(
        lambda row: heuristic_sentiment(row), axis=1)
    df.to_csv('./data/heuristic_sentiment.csv', index=False)
    logger.info(
        "Heuristic sentiment analysis applied and saved to preprocessed_reviews.csv")


def heuristic_sentiment(row):
    """Determine sentiment based on rating, verified purchase, and helpful votes."""
    rating = row['rating']
    verified_purchase = row.get('verified_purchase', False)
    helpful_votes = row.get('helpful_vote', 0)
    sentiment_score = 0
    if rating >= 4:
        sentiment_score += 1
    elif rating <= 2:
        sentiment_score -= 1
    if verified_purchase:
        sentiment_score += 0.5
    if helpful_votes >= 5:
        sentiment_score += 0.5
    return 'positive' if sentiment_score > 1 else 'negative' if sentiment_score < 0 else 'neutral'


def apply_vader_sentiment():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()

    """Apply VADER sentiment analysis and save results."""
    df = pd.read_csv(PREPROCESSED_CSV)
    df['vader_sentiment'] = df['cleaned_text'].apply(
        lambda text: vader_sentiment(text))
    df.to_csv('./data/vader_sentiment.csv', index=False)
    logger.info(
        "VADER sentiment analysis applied and saved to preprocessed_reviews.csv")


def vader_sentiment(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    vader_analyzer = SentimentIntensityAnalyzer()

    if isinstance(text, str):
        score = vader_analyzer.polarity_scores(text)
        return 'positive' if score['compound'] >= 0.05 else 'negative' if score['compound'] <= -0.05 else 'neutral'


def apply_textblob_sentiment():
    from textblob import TextBlob

    """Apply TextBlob sentiment analysis and save results."""
    df = pd.read_csv(PREPROCESSED_CSV)
    df['textblob_sentiment'] = df['cleaned_text'].apply(
        lambda text: textblob_sentiment(text))
    df.to_csv('./data/textblob_sentiment.csv', index=False)
    logger.info(
        "TextBlob sentiment analysis applied and saved to preprocessed_reviews.csv")


def textblob_sentiment(text):
    from textblob import TextBlob

    if isinstance(text, str):
        analysis = TextBlob(text)
        return 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'


# Majority Voting Function
def apply_majority_vote():
    """Apply majority voting based on all sentiment analysis results and save."""
    df = pd.read_csv(PREPROCESSED_CSV)
    heuristic_df = pd.read_csv('./data/heuristic_sentiment.csv')
    vader_df = pd.read_csv('./data/vader_sentiment.csv')
    textblob_df = pd.read_csv('./data/textblob_sentiment.csv')

    # Merge the files on a common column
    df = pd.concat([df, heuristic_df[['cleaned_text', 'heuristic_sentiment']],
                vader_df[['cleaned_text', 'vader_sentiment']],
                textblob_df[['cleaned_text', 'textblob_sentiment']]], axis=1)

    df['final_sentiment'] = df.apply(lambda row: majority_vote(
        row['heuristic_sentiment'], row['vader_sentiment'],
        row['textblob_sentiment']
    ), axis=1)
    df[['cleaned_text', 'rating', 'final_sentiment']].to_csv(
        FINAL_OUTPUT_CSV, index=False)
    logger.info(
        "Final sentiment labeling completed and saved to final_amazon_reviews.csv")


def majority_vote(heuristic, vader, textblob):
    votes = [heuristic, vader, textblob]
    pos_count = votes.count('positive')
    neg_count = votes.count('negative')
    neu_count = votes.count('neutral')
    return 'positive' if pos_count > neg_count and pos_count >= neu_count else 'negative' if neg_count > pos_count and neg_count > neu_count else 'neutral'
