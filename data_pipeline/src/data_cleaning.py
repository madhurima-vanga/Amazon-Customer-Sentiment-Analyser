from bs4 import BeautifulSoup
from tqdm import tqdm  # Import tqdm for progress bars
import logging

logger = logging.getLogger()

# Initialize tqdm for Pandas
tqdm.pandas()

def clean_text(text):
    # Ensure the text is not empty and not a potential file path
    if isinstance(text, str) and text.strip() != "":
        try:
            # Parse the text using BeautifulSoup only if it contains HTML-like content
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception as e:
            print(f"Error parsing text with BeautifulSoup: {e}")
            return text  # Return the original text if parsing fails
    return text.lower()  # Convert to lowercase after parsing



def clean_dataframe(df):
    logger.info("Cleaning dataframe initiated")
    print(f"Processing text for cleaning....")
    try:
        df['cleaned_text'] = df['text'].progress_apply(clean_text)
        logger.info(f"Data cleaning completed for {len(df)} records")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise
    return df
