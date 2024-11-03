from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.email import EmailOperator

from data_pipeline import (
    load_reviews_dataset,
    load_metadata_dataset,
    merge_data,
    filter_urls_paths,
    remove_html_tags,
    remove_non_string_reviews,
    apply_heuristic_sentiment,
    apply_vader_sentiment,
    apply_textblob_sentiment,
    apply_majority_vote,
    failure_email,
    perform_schema_check
)

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'sentiment_analysis_pipeline',
    default_args=default_args,
    description='A Data pipeline for sentiment analysis with Airflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    load_reviews_task = PythonOperator(
        task_id='load_reviews_dataset',
        python_callable=load_reviews_dataset,
        on_failure_callback=failure_email
    )

    load_metadata_task = PythonOperator(
        task_id='load_metadata_dataset',
        python_callable=load_metadata_dataset,
        on_failure_callback=failure_email
    )

    # Define schema check task to validate loaded data
    schema_check_task = PythonOperator(
        task_id='schema_validation',
        python_callable=perform_schema_check
    )

    merge_data_task = PythonOperator(
        task_id='merge_data',
        python_callable=merge_data,
        on_failure_callback=failure_email
    )

    remove_html_tags_task = PythonOperator(
        task_id='remove_html_tags',
        python_callable=remove_html_tags,
        on_failure_callback=failure_email
    )

    filter_urls_paths_task = PythonOperator(
        task_id='filter_urls_paths',
        python_callable=filter_urls_paths,
        on_failure_callback=failure_email
    )

    remove_non_string_reviews_task = PythonOperator(
        task_id='remove_non_string_reviews',
        python_callable=remove_non_string_reviews,
        on_failure_callback=failure_email
    )

    heuristic_sentiment_task = PythonOperator(
        task_id='heuristic_sentiment_analysis',
        python_callable=apply_heuristic_sentiment,
        on_failure_callback=failure_email
    )

    vader_sentiment_task = PythonOperator(
        task_id='vader_sentiment_analysis',
        python_callable=apply_vader_sentiment,
        on_failure_callback=failure_email
    )

    textblob_sentiment_task = PythonOperator(
        task_id='textblob_sentiment_analysis',
        python_callable=apply_textblob_sentiment,
        on_failure_callback=failure_email
    )

    majority_vote_task = PythonOperator(
        task_id='apply_majority_vote',
        python_callable=apply_majority_vote,
        on_failure_callback=failure_email
    )

    # Email operator for sending success notification
    send_email_success = EmailOperator(
        task_id='send_email_success',
        to='yesitsmefolks@gmail.com',
        subject='Data Pipeline Success: Sentiment Analysis Data Pipeline Completed',
        html_content="<p>The sentiment analysis data extraction and labeling pipeline has completed successfully.</p>",
        trigger_rule='all_success',
        dag=dag
    )

    # Set up dependencies
    [load_reviews_task, load_metadata_task] >> schema_check_task >> merge_data_task >> remove_html_tags_task >> filter_urls_paths_task >> remove_non_string_reviews_task
    remove_non_string_reviews_task >> [
        heuristic_sentiment_task, vader_sentiment_task, textblob_sentiment_task]
    [heuristic_sentiment_task, vader_sentiment_task,
        textblob_sentiment_task] >> majority_vote_task >> send_email_success