from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import pandas as pd
from src.data_cleaning import clean_dataframe
from src.data_preprocessing import apply_sentiment_analysis
from src.data_labelling import apply_majority_vote

# Set default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'data_pipeline_dag',  # DAG name
    default_args=default_args,
    description='A simple data pipeline DAG for sentiment analysis',
    schedule_interval=timedelta(days=1),  # Runs every day
)

# Define your functions (you can reuse the same functions you have)
def load_data(file_path):
    df = pd.read_json(file_path, lines=True)
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def run_cleaning(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='load_data_task')
    df = clean_dataframe(df)
    ti.xcom_push(key='cleaned_data', value=df)

def run_sentiment_analysis(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='clean_data_task')
    df = apply_sentiment_analysis(df)
    ti.xcom_push(key='sentiment_data', value=df)

def run_majority_vote(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='sentiment_analysis_task')
    df = apply_majority_vote(df)
    ti.xcom_push(key='final_data', value=df)

def save_final_data(**kwargs):
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='majority_vote_task')
    save_data(df, '../data/processed/final_amazon_reviews.csv')

# Define tasks using PythonOperator
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    op_kwargs={'file_path': '../data/raw/Appliances.jsonl'},
    dag=dag,
)

clean_data_task = PythonOperator(
    task_id='clean_data_task',
    python_callable=run_cleaning,
    provide_context=True,  # Allows passing data between tasks via XCom
    dag=dag,
)

sentiment_analysis_task = PythonOperator(
    task_id='sentiment_analysis_task',
    python_callable=run_sentiment_analysis,
    provide_context=True,
    dag=dag,
)

majority_vote_task = PythonOperator(
    task_id='majority_vote_task',
    python_callable=run_majority_vote,
    provide_context=True,
    dag=dag,
)

save_data_task = PythonOperator(
    task_id='save_data_task',
    python_callable=save_final_data,
    provide_context=True,
    dag=dag,
)

# Set up task dependencies
load_data_task >> clean_data_task >> sentiment_analysis_task >> majority_vote_task >> save_data_task
