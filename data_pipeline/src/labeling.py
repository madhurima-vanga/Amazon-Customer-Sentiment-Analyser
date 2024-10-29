import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import datetime

# Define text cleaning function
def clean_text(text):
    # Step 1: Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Optional Step 2: Convert to lowercase (only if using an uncased BERT model)
    text = text.lower()

    return text

# Load your data (replace 'your_reviews_data.csv' with your actual data file)
df = pd.read_json('Appliances.jsonl', lines=True)

# Print a message after the file is successfully loaded
print("File has been successfully loaded. Number of rows read:", len(df))

# Initialize the sentiment models
vader_analyzer = SentimentIntensityAnalyzer()
flair_classifier = TextClassifier.load('sentiment-fast')

# Enable tqdm progress bar for pandas
tqdm.pandas()

# Heuristic function using metadata
def heuristic_sentiment(row):
    rating = row['rating']  # Assuming rating column is available
    verified_purchase = row['verified_purchase']
    helpful_votes = row['helpful_vote']

    if isinstance(row['timestamp'], (int, float)):
        review_time = datetime.datetime.fromtimestamp(row['timestamp'])
    elif isinstance(row['timestamp'], pd.Timestamp):
        review_time = row['timestamp'].to_pydatetime()
    else:
        raise ValueError(f"Unknown timestamp format: {type(row['timestamp'])}")

    sentiment_score = 0

    if rating >= 4:
        sentiment_score += 1
    elif rating <= 2:
        sentiment_score -= 1

    if verified_purchase:
        sentiment_score += 0.5

    if helpful_votes >= 5:
        sentiment_score += 0.5

    time_diff = (datetime.datetime.now() - review_time).days
    if time_diff < 180:
        sentiment_score += 0.5

    if sentiment_score > 1:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

# VADER Sentiment function
def vader_sentiment(text):
    score = vader_analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# TextBlob Sentiment function
def textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Flair Sentiment function
##def flair_sentiment(text):
    #sentence = Sentence(text)
    #flair_classifier.predict(sentence)
    #label = sentence.labels[0]
    #return label.value.lower()

def flair_sentiment(text):
    sentence = Sentence(text)
    flair_classifier.predict(sentence)
    
    if len(sentence.labels) > 0:
        label = sentence.labels[0].value  # Get the predicted sentiment
        if label == 'POSITIVE':
            return 'positive'
        elif label == 'NEGATIVE':
            return 'negative'
        else:
            return 'neutral'
    else:
        # If no sentiment label is found, return 'neutral' or handle appropriately
        return 'neutral'

# Function to aggregate and take majority vote
def majority_vote(heuristic, vader, textblob, flair):
    votes = [heuristic, vader, textblob, flair]
    pos_count = votes.count('positive')
    neg_count = votes.count('negative')
    neu_count = votes.count('neutral')

    if pos_count > neg_count and pos_count > neu_count:
        return 'positive'
    elif neg_count > pos_count and neg_count > neu_count:
        return 'negative'
    else:
        return 'neutral'
df=df.head(600)
# Apply cleaning before passing to sentiment models
print("Cleaning text...")
df['cleaned_text'] = df['text'].progress_apply(clean_text)

# Apply heuristic sentiment analysis
print("Processing heuristic sentiment...")
df['heuristic_sentiment'] = df.progress_apply(heuristic_sentiment, axis=1)

# Apply VADER sentiment analysis
print("Processing VADER sentiment...")
df['vader_sentiment'] = df['cleaned_text'].progress_apply(vader_sentiment)

# Apply TextBlob sentiment analysis
print("Processing TextBlob sentiment...")
df['textblob_sentiment'] = df['cleaned_text'].progress_apply(textblob_sentiment)

# Apply Flair sentiment analysis
print("Processing Flair sentiment...")
df['flair_sentiment'] = df['cleaned_text'].progress_apply(flair_sentiment)

# Final label based on majority vote
df['final_sentiment'] = df.apply(lambda row: majority_vote(row['heuristic_sentiment'], row['vader_sentiment'], row['textblob_sentiment'], row['flair_sentiment']), axis=1)

# Select the desired columns and save them to a CSV file
df_selected = df[['text', 'rating', 'timestamp', 'helpful_vote', 'parent_asin', 'final_sentiment']]
df_selected.to_csv('final_amazon_reviews.csv', index=False)

print("CSV file with selected columns has been saved as 'final_amazon_reviews.csv'.")
