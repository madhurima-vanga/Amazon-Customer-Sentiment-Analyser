from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from tqdm import tqdm  # Import tqdm for progress bars
import datetime
import pandas as pd
import logging

logger = logging.getLogger()

# Initialize tqdm for Pandas
tqdm.pandas()

# Initialize sentiment models
vader_analyzer = SentimentIntensityAnalyzer()
flair_classifier = TextClassifier.load('sentiment-fast')

# Heuristic sentiment function
def heuristic_sentiment(row):
    rating = row['rating']
    verified_purchase = row['verified_purchase']
    helpful_votes = row['helpful_vote']

    # Determine the timestamp format
    if isinstance(row['timestamp'], (int, float)):
        review_time = datetime.datetime.fromtimestamp(row['timestamp'])
    elif isinstance(row['timestamp'], pd.Timestamp):
        review_time = row['timestamp'].to_pydatetime()
    else:
        raise ValueError(f"Unknown timestamp format: {type(row['timestamp'])}")

    # Start with neutral sentiment score
    sentiment_score = 0

    # Rating-based sentiment
    if rating >= 4:
        sentiment_score += 1
    elif rating <= 2:
        sentiment_score -= 1

    # Add weight if the purchase is verified
    if verified_purchase:
        sentiment_score += 0.5

    # Add weight if there are a significant number of helpful votes
    if helpful_votes >= 5:
        sentiment_score += 0.5

    # Add weight if the review is recent (within the last 6 months)
    time_diff = (datetime.datetime.now() - review_time).days
    if time_diff < 180:
        sentiment_score += 0.5

    # Determine sentiment based on the final score
    if sentiment_score > 1:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# VADER sentiment function
def vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)
    return 'positive' if score['compound'] >= 0.05 else 'negative' if score['compound'] <= -0.05 else 'neutral'

# TextBlob sentiment function
def textblob_sentiment(text):
    analysis = TextBlob(text)
    return 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'

# Flair sentiment function
def flair_sentiment(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    label = sentence.labels[0].value if sentence.labels else 'neutral'
    return 'positive' if label == 'POSITIVE' else 'negative' if label == 'NEGATIVE' else 'neutral'

# Apply all sentiment analysis functions
def apply_sentiment_analysis(df):
    logger.info("Sentiment Analysis initialized")

    print("Running heuristic sentiment analysis...")
    logger.info("Running Heuristic Sentiment Labler")
    df['heuristic_sentiment'] = df.progress_apply(heuristic_sentiment, axis=1)
    logger.info("Finished Heuristic Sentiment Labeling")

    logger.info("Running VADER Sentiment Labler")
    print("Running VADER sentiment analysis...")
    df['vader_sentiment'] = df['cleaned_text'].progress_apply(vader_sentiment)
    logger.info("Finished VADER Sentiment Labeling")

    logger.info("Running TextBlob Sentiment Labler")
    print("Running TextBlob sentiment analysis...")
    df['textblob_sentiment'] = df['cleaned_text'].progress_apply(textblob_sentiment)
    logger.info("Finished TextBlob Sentiment Labeling")

    logger.info("Running Flair Sentiment Labler")
    print("Running Flair sentiment analysis...")
    df['flair_sentiment'] = df['cleaned_text'].progress_apply(flair_sentiment)
    logger.info("Finished Flair Sentiment Labeling")

    return df
