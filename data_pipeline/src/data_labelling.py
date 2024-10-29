from tqdm import tqdm  # Import tqdm for progress bars
import logging

logger = logging.getLogger()

# Initialize tqdm for Pandas
tqdm.pandas()

def majority_vote(heuristic, vader, textblob, flair):
    votes = [heuristic, vader, textblob, flair]
    pos_count = votes.count('positive')
    neg_count = votes.count('negative')
    neu_count = votes.count('neutral')

    if pos_count > neg_count and pos_count > neu_count:
        return 'positive'
    elif neg_count > pos_count and neg_count > neu_count:
        return 'negative'
    return 'neutral'

def apply_majority_vote(df):
    logger.info("Majority Voting")
    print('Majority Voting Status...')
    df['final_sentiment'] = df.progress_apply(lambda row: majority_vote(
        row['heuristic_sentiment'], row['vader_sentiment'], 
        row['textblob_sentiment'], row['flair_sentiment']
    ), axis=1)  # Use progress_apply for progress bar
    return df
