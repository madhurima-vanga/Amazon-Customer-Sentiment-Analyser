import pandas as pd
from data_cleaning import clean_dataframe
from data_preprocessing import apply_sentiment_analysis
from data_labelling import apply_majority_vote
import os
from datetime import datetime
import logging

# Step 1: Define the new log folder path
log_folder = '../system_logs/data_pipeline'

# Step 2: Create the directory if it doesn't exist
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Step 3: Create a unique log file name with a timestamp
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f'{log_folder}/pipeline_{current_time}.log'

# Step 4: Configure logging with file and console handlers
logging.basicConfig(
    filename=log_file,  # Log file path with unique name
    level=logging.DEBUG,  # Set to DEBUG level to capture all logs
    format='%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | Line: %(lineno)d | %(message)s'
)

# Adding a StreamHandler to print logs to the console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Log level for console output
console_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
console_handler.setFormatter(console_format)

logger = logging.getLogger()
logger.addHandler(console_handler)  # Add the handler to the logger

# Example pipeline functions (same as before)
def load_data(file_path):
    logger.info(f"Attempting to load data from {file_path}")
    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try reading the JSONL file
    try:
        df = pd.read_json(file_path, lines=True)
        logger.info("Data loaded successfully")
    except ValueError as e:
        logger.error(f"Error reading the JSON file: {e}")
        raise ValueError(f"Error reading the JSON file: {e}")
    
    print(f"File has been successfully loaded. Number of rows read: {len(df)}")
    logger.info(f"Data Loading completed for {len(df)} records")
    return df

def save_data(df, output_path):
    # Save the DataFrame as a CSV file
    df.to_csv(output_path, index=False)
    print(f"CSV file saved at: {output_path}")

def run_pipeline(file_path, output_path):
    logger.info("Pipeline started")

    # Load the data
    logger.info(f"Loading data from {file_path}")
    try:
        df = load_data(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    df = df.head(600)  # Limit to 600 rows for this example

    # Step 1: Clean the text data
    logger.info("Starting data cleaning")
    try:
        df = clean_dataframe(df)
        logger.info("Data cleaning completed")
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise

    # Step 2: Perform sentiment analysis
    logger.info("Starting sentiment analysis")
    try:
        df = apply_sentiment_analysis(df)
        logger.info("Sentiment analysis completed")
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise

    # Step 3: Apply majority vote for final sentiment
    logger.info("Applying majority vote for sentiment labeling")
    try:
        df = apply_majority_vote(df)
        logger.info("Sentiment labeling completed")
    except Exception as e:
        logger.error(f"Sentiment labeling failed: {e}")
        raise

    # Save the final result
    logger.info(f"Saving data to {output_path}")
    try:
        save_data(df, output_path)
        logger.info("Data saved successfully")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    # Specify the input file and output file paths
    file_path = '../data/raw/Appliances.jsonl'  # The JSONL file path
    output_path = '../data/processed/final_amazon_reviews.csv'  # Where to save the CSV

    # Run the pipeline and save the results
    run_pipeline(file_path, output_path)
