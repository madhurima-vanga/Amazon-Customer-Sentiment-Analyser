import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
from scipy.stats import ks_2samp

# Path to email login credential file
credentials_path = "/Users/madhurima/Documents/GitHub/Amazon-Customer-Sentiment-Analyser/detect_drift/credentials.csv"

# Path to the new live data and saved training data statistics CSV file
new_data_path = "/Users/madhurima/Documents/GitHub/Amazon-Customer-Sentiment-Analyser/detect_drift/live_data_with_drift.csv"
saved_stats_path = "/Users/madhurima/Documents/GitHub/Amazon-Customer-Sentiment-Analyser/detect_drift/statistics_new_data.csv.csv"

# p_value_threshold
p_value_threshold = 0.05

# Function to read email and password from CSV
def read_credentials_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        email = df['email'][0]
        password = df['password'][0]
        return email, password
    except Exception as e:
        print(f"Error reading credentials from CSV: {e}")
        return None, None

# Function to send email
def send_email(subject, body, to_email):
    from_email, password = read_credentials_from_csv(credentials_path)  # Read email and password from CSV
    
    if not from_email or not password:
        print("Email or password not found. Exiting.")
        return

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Using iCloud's SMTP server with TLS
        server = smtplib.SMTP('smtp.mail.me.com', 587)  # iCloud SMTP server
        server.starttls()  # Use TLS (Transport Layer Security)
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Function to check drift using Kolmogorov-Smirnov Test
def check_drift(saved_data_stats, new_data_stats):
    saved_values = np.array(list(saved_data_stats.values()))
    new_values = np.array(list(new_data_stats.values()))
    
    # Kolmogorov-Smirnov Test
    ks_stat, p_value = ks_2samp(saved_values, new_values)
    
    # Drift detected if p-value is below threshold (e.g., 0.05)
    drift_detected = p_value > p_value_threshold
    
    return drift_detected, ks_stat, p_value

# Load the new data
try:
    new_data = pd.read_csv(new_data_path)
except Exception as e:
    print(f"Error loading new data: {e}")
    raise

# Ensure 'cleaned_text' column exists in the new data
if "cleaned_text" not in new_data.columns:
    raise ValueError("The 'cleaned_text' column is missing in new data.")

# Extract and clean review text from new data
new_reviews = new_data["cleaned_text"].dropna()

# Tokenize and compute word frequencies for the new data
vectorizer = CountVectorizer()
new_matrix = vectorizer.fit_transform(new_reviews)
new_freq = new_matrix.sum(axis=0).A1

# Get feature names from the new data
new_feature_names = vectorizer.get_feature_names_out()
new_data_stats = dict(zip(new_feature_names, new_freq))

# Compare new data statistics with saved statistics (if the saved file exists)
try:
    saved_stats_df = pd.read_csv(saved_stats_path)
    saved_data_stats = dict(zip(saved_stats_df['Feature'], saved_stats_df['Frequency']))
except FileNotFoundError:
    print(f"No saved statistics found at {saved_stats_path}. Creating a new file.")
    saved_data_stats = {}

# Check if there is drift
drift_detected, ks_stat, p_value = check_drift(saved_data_stats, new_data_stats)

if drift_detected:
    text = "Further investigation required"
else:
    text = "No action required"

# Create the body for the statistics email
statistics_email_body = f"""
New data statistics have been processed.

Kolmogorov-Smirnov Test:
KS Stat: {ks_stat}
P-Value: {p_value}

Drift Detected: {'Yes' if drift_detected else 'No'}

{text}

"""

# Send a general statistics email for every run
send_email("Live Data Drift Detection - Statistics", statistics_email_body, "sentimentanalysisreviewproject@gmail.com")

# If drift is detected, trigger an alert email
if drift_detected:
    drift_alert_body = f"""
    DRIFT ALERT: Significant drift detected between training and live data.

    Kolmogorov-Smirnov Test:
    KS Stat: {ks_stat}
    P-Value: {p_value}

    Drift has been detected, which might indicate that the model is performing differently.
    Please review the changes in the data and retrain the model if necessary.
    """
    send_email("Live Data Drift Alert", drift_alert_body, "sentimentanalysisreviewproject@gmail.com")

# Write the new data statistics to the CSV file, overwriting the existing one
new_data_stats_df = pd.DataFrame(list(new_data_stats.items()), columns=["Feature", "Frequency"])
new_data_stats_df.to_csv(saved_stats_path, index=False)

print(f"New data statistics saved to {saved_stats_path}")
