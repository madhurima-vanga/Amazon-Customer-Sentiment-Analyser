import os
import pandas as pd
from dags.data_pipeline import (
    merge_data,
    remove_html_tags,
    filter_urls_paths,
    heuristic_sentiment,
    vader_sentiment,
    textblob_sentiment,
    majority_vote
)

# Mock paths for tests
TEST_DATA_DIR = "./data"
TEST_REVIEWS_CSV = os.path.join(TEST_DATA_DIR, "reviews.csv")
TEST_METADATA_CSV = os.path.join(TEST_DATA_DIR, "metadata.csv")
TEST_MERGED_CSV = os.path.join(TEST_DATA_DIR, "merged_data.csv")
TEST_HTML_CLEANED_CSV = os.path.join(TEST_DATA_DIR, 'html_cleaned_data.csv')
TEST_PREPROCESSED_CSV = os.path.join(TEST_DATA_DIR, 'preprocessed_reviews.csv')
TEST_FINAL_OUTPUT_CSV = os.path.join(TEST_DATA_DIR, 'final_amazon_reviews.csv')

# Helper function to create sample CSV data


def create_sample_reviews():
    data = {
        'parent_asin': ['B001', 'B002', 'B003'],
        'title': ['Review A', 'Review B', 'Review C'],
        'text': ['Great product!', 'Not bad', 'Could be better'],
        'rating': [5, 3, 1],
        'verified_purchase': [True, False, True],
        'helpful_vote': [10, 2, 1]  # Include 'helpful_vote'
    }
    pd.DataFrame(data).to_csv(TEST_REVIEWS_CSV, index=False)


def create_sample_metadata():
    data = {
        'parent_asin': ['B001', 'B002', 'B003'],
        'title': ['Product A', 'Product B', 'Product C'],
        'store': ['Store A', 'Store B', 'Store C'],  # Include 'store'
        'price': [29.99, 19.99, 9.99]
    }
    pd.DataFrame(data).to_csv(TEST_METADATA_CSV, index=False)


def test_merge_data():
    create_sample_metadata()
    create_sample_reviews()
    merge_data()
    assert os.path.exists(
        TEST_MERGED_CSV), "Merged file should exist after merging data"
    df = pd.read_csv(TEST_MERGED_CSV)
    assert 'product_title' in df.columns
    assert df.shape[0] == 3


def test_remove_html_tags():
    create_sample_metadata()
    create_sample_reviews()
    df = pd.DataFrame(
        {"text": ["<p>Great product!</p>", "<div>Not good</div>"]})
    df.to_csv(TEST_MERGED_CSV, index=False)
    remove_html_tags()
    df = pd.read_csv(TEST_HTML_CLEANED_CSV)
    assert df["cleaned_text"].iloc[0] == "great product!"
    assert df["cleaned_text"].iloc[1] == "not good"


def test_filter_urls_paths():
    """Test that rows containing URLs or file paths are filtered out correctly based on current filter function behavior."""

    # Create data that doesn't rely strictly on URLs or file paths, based on current filtering patterns
    df = pd.DataFrame({
        'text': ['Sample text', 'Sample text', 'Sample text'],
        'cleaned_text': [
            'Review with content',
            'https://www.example.com',
            'Some other simple review'
        ]
    })

    df.to_csv(TEST_HTML_CLEANED_CSV, index=False)
    filter_urls_paths()

    filtered_df = pd.read_csv(TEST_PREPROCESSED_CSV)

    assert len(
        filtered_df) == 2, "Should retain 2 rows based on current filter pattern"
    assert 'Review with content' in filtered_df['cleaned_text'].values
    assert 'Some other simple review' in filtered_df['cleaned_text'].values


def test_heuristic_sentiment():
    row = {"rating": 5, "verified_purchase": True, "helpful_vote": 6}
    result = heuristic_sentiment(row)
    assert result == "positive"


def test_vader_sentiment():
    text = "I love this!"
    result = vader_sentiment(text)
    assert result == "positive"


def test_textblob_sentiment():
    text = "I hate this!"
    result = textblob_sentiment(text)
    assert result == "negative"


def test_majority_vote():
    result = majority_vote("positive", "positive", "neutral")
    assert result == "positive"
