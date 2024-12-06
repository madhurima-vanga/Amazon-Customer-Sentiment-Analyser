import functions_framework
import os
import pandas as pd
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import *
import psycopg2
import json
import logging
import smtplib
from email.mime.text import MIMEText
import requests
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# DB Connection parameters 
connection_params = {
    'dbname': os.environ['DB_NAME'],
    'user': os.environ['DB_USER'],
    'password': os.environ['DB_PASSWORD'],
    'host': os.environ['DB_HOST'],
    'port': os.environ['DB_PORT']
}

#SMTP parameters 
smtp_server = os.environ['SMTP_SERVER']
port = os.environ['SMTP_PORT']
login = os.environ['SMTP_LOGIN']
password = os.environ['SMTP_PASSWORD'] 
sender_email = os.environ['SENDER_EMAIL']
receiver_email = os.environ['RECEIVER_EMAIL']

# Jenkins server details
JENKINS_URL = os.environ['JENKINS_URL']
JENKINS_USERNAME = os.environ['JENKINS_USER']
JENKINS_API_TOKEN = os.environ['JENKINS_API_TOKEN']

# Name of the job to trigger
JOB_NAME = os.environ['JOB_NAME']

@app.route('/drift-detection', methods=['POST'])
def drift_detection(request):
    
    def fetch_cleaned_texts():
        try:
            # Establish a connection to the database
            connection = psycopg2.connect(**connection_params)
            cursor = connection.cursor()       
            reference_text_query = """SELECT cleaned_text FROM public.reviews WHERE review_id between 1 and 500;"""
            current_text_query = """SELECT cleaned_text FROM public.reviews ORDER BY review_id DESC LIMIT 5;;"""
            cursor.execute(reference_text_query)       
            reference_text_results = cursor.fetchall()
            cursor.execute(current_text_query)       
            current_text_results = cursor.fetchall()
            reference_text_list = [row[0] for row in reference_text_results]
            current_text_list = [row[0] for row in current_text_results]
            return reference_text_list,current_text_list
        except Exception as e:
            print(f"An error occurred: {e}")  
        finally:
            # Ensure that the cursor and connection are closed
            if cursor:
                cursor.close()
            if connection:
                connection.close()    

    def send_email(subject, body):
        message = MIMEText(body, "html")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email
        # Send the email using SMTP
        try:
            
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()  
            server.login(login, password)
            # Send the email
            server.sendmail(sender_email, receiver_email, message.as_string())
            logging.warning("Email sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")
        finally:
            # Ensure the connection is closed after sending the email
            server.quit()

    def trigger_jenkins_job():
        try:
            # Trigger URL
            trigger_url = f"{JENKINS_URL}/job/{JOB_NAME}/build"

            # Trigger the job with basic authentication
            response = requests.post(trigger_url, auth=(JENKINS_USERNAME, JENKINS_API_TOKEN))

            # Check the response status
            if response.status_code == 201:
                 logging.warning("Jenkins job triggered successfully!")
            elif response.status_code == 403:
                 logging.warning("Authentication failed. Check your credentials.")
            else:
                logging.warning(f"Failed to trigger Jenkins job. Status code: {response.status_code}")
                logging.warning(f"Response: {response.text}")
    
        except Exception as e:
            logging.warning(f"An error occurred: {e}")

    def update_drift_counter_and_trigger_actions():
        try:
        
            # Establish a connection to the database
            connection = psycopg2.connect(**connection_params)
            cursor = connection.cursor()

            # Fetch the current drift counter value
            cursor.execute("SELECT drift_counter FROM drift_table LIMIT 1;")
            result = cursor.fetchone()

            if result is None:
                # If no row exists in the table, initialize drift_counter to 0
                logging.warning("No row found in drift_table. Initializing drift_counter to 0.")
                return

            current_count = result[0]
            logging.warning(f"Current drift_counter value: {current_count}")

            if current_count < 9:
                # Increment the drift counter
                new_count = current_count + 1
                # update_query = "UPDATE drift_table SET drift_counter = %s;"
                # cursor.execute(update_query, (new_count,))
                cursor.execute("UPDATE drift_table SET drift_counter = %s;", (new_count,))
                connection.commit()
                logging.warning(f"Drift counter incremented to {new_count}.")
            else:
                # Reset the drift counter to 0 and trigger email and job
                cursor.execute("UPDATE drift_table SET drift_counter = 0;")
                connection.commit()
                logging.warning("Drift counter reached 10. Resetting to 0 and triggering actions.")
            
                # Trigger email and Jenkins job
                subject = "Alert: Drift Detected in Dataset"
                body = "Drift has been detected in the dataset. Please review the changes. And Model Retrain is triggered"
                # Send an alert email
                logging.warning("Triggering an Drift Detected email alert.")
                send_email(subject, body)

                logging.warning("Triggering Retrain Pipeline")
                trigger_jenkins_job()

        except Exception as e:
            logging.error(f"An error occurred while updating the drift counter: {e}")

    request_json = request.get_json(silent=True)
    #Fetching review data from db
    logging.info("Fetching review data from the database")
    reference_text,current_text=fetch_cleaned_texts()
    reviews_ref = pd.DataFrame(reference_text, columns=['review_text'])
    reviews_cur = pd.DataFrame(current_text, columns=['review_text'])

    column_mapping = ColumnMapping(
        text_features=['review_text']
    )

    #Initiating metrics report
    logging.info("Initiating text-specific metrics report.")
    text_specific_metrics_report = Report(metrics=[
        TextDescriptorsDriftMetric(column_name="review_text",stattest_threshold=0.05),
    ])

    text_specific_metrics_report.run(reference_data=reviews_ref, current_data=reviews_cur, column_mapping=column_mapping)

    #Reading the report json to check dataset_drift
    logging.info("Reading the report JSON to check for dataset drift.")
    text_metrics_report_json_str = text_specific_metrics_report.json()
    text_metrics_report_json = json.loads(text_metrics_report_json_str)
    logging.info(text_metrics_report_json_str)

    # Check for drifted columns
    number_of_drifted_columns = text_metrics_report_json['metrics'][0]['result']['number_of_drifted_columns']
    logging.warning(f'number_of_drifted_columns : {number_of_drifted_columns}')
    are_columns_drifted = number_of_drifted_columns > 0 
    

    if are_columns_drifted:
        logging.warning("Drift detected in the dataset.")
        update_drift_counter_and_trigger_actions()
    else:
        logging.warning("No drift detected.")
    return jsonify({"message": "Drift detection completed successfully."}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)