import streamlit as st
import json
import subprocess
from transformers import DistilBertTokenizer
import psycopg2
import os
import requests

st.set_page_config(
    page_title="MLOps Application",
    page_icon="🤖",
    layout="wide"
)

# Database connection
try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    cursor = conn.cursor()
    #st.write("Database connection established successfully.")
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def get_cursor():
    global conn
    if conn.closed:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
    return conn.cursor()

# Load categories
@st.cache_data
def load_categories():
    cursor = get_cursor()
    cursor.execute("SELECT * FROM categories")
    categories = cursor.fetchall()
    return categories

# Load products for a specific category
@st.cache_data
def load_products(category_id):
    cursor = get_cursor()
    cursor.execute("SELECT * FROM products WHERE category_id = %s", (category_id,))
    products = cursor.fetchall()
    return products

# Load product details and reviews
#@st.cache_data
def load_product_details(product_id):
    cursor = get_cursor()
    cursor.execute("SELECT * FROM products WHERE product_id = %s", (product_id,))
    product = cursor.fetchone()
    cursor.execute("SELECT * FROM reviews WHERE product_id = %s", (product_id,))
    reviews = cursor.fetchall()
    return product, reviews



def calculate_avg_rating(product_id):
    try:
        cursor = get_cursor()
        cursor.execute("SELECT AVG(rating) FROM reviews WHERE product_id = %s", (product_id,))
        avg_rating = cursor.fetchone()[0]
        return round(avg_rating, 2) if avg_rating else 0.0
    except Exception as e:
        st.error(f"Failed to calculate average rating: {e}")
        return 0.0


def call_drift_detection_function():
    SERVICE_URL = "https://us-west1-wired-glider-441301-e1.cloudfunctions.net/drift-detect"  

    payload = {}

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(SERVICE_URL, json=payload, headers=headers)
        if response.status_code == 200:
            print("Request was successful!")
            print("Response:", response.json())
        else:
            print(f"Request failed with status code: {response.status_code}")
            print("Response:", response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

# Tokenization function
def tokenize_texts(texts, max_length=256):
    tokens = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"].tolist()
    attention_mask = tokens["attention_mask"].tolist()
    instances = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(input_ids, attention_mask)
    ]
    #st.write(f"Tokenized instances: {instances}")
    return instances

def activate_service_account():
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # If no key file is set, assume gcloud is already authenticated (e.g., on local machine)
    if not key_file:
        #st.write("No key file found. Assuming gcloud is already configured on this machine.")
        return True

    # If key file exists, activate the service account
    if os.path.exists(key_file):
        try:
            subprocess.run(
                ["gcloud", "auth", "activate-service-account", f"--key-file={key_file}"],
                check=True,
            )
            #st.write("Service account activated successfully.")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to activate service account: {e}")
            return False
    else:
        st.error(f"Google Cloud key file not found at {key_file}.")
        return False

# Vertex AI Prediction function
def query_vertex_ai(instances):

    if not activate_service_account():
        return None
    try:
        input_data = {"instances": instances}
        input_file = "input.json"
        with open(input_file, "w") as f:
            json.dump(input_data, f)

        access_token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        token = access_token.stdout.strip()
        if not token:
            st.error("Failed to retrieve a valid access token.")
            return None

        endpoint_id = "6768545202077433856"
        project_id = "317356856351"
        url = f"https://us-east1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-east1/endpoints/{endpoint_id}:predict"
        command = [
            "curl", "-X", "POST",
            "-H", f"Authorization: Bearer {token}",
            "-H", "Content-Type: application/json",
            url,
            "-d", f"@{input_file}"
        ]

        response = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        #st.write(f"Raw response from Vertex AI: {response.stdout}")
        response_json = json.loads(response.stdout)
        return response_json

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def classify_sentiment(scores):
    if scores[0] > scores[1] and scores[0] > scores[2]:
        return "Positive"
    elif scores[1] > scores[0] and scores[1] > scores[2]:
        return "Negative"
    else:
        return "Neutral"

# Insert a new review into the database
def insert_review(product_id, review_text, rating, sentiment):
    try:
        cursor = get_cursor()
        #st.write(f"Inserting review: Product ID: {product_id}, Review: {review_text}, Rating: {rating}, Sentiment: {sentiment}")
        cursor.execute(
            """
            INSERT INTO reviews (product_id, cleaned_text, rating, timestamp, final_sentiment, helpful_vote)
            VALUES (%s, %s, %s, NOW(), %s, 0)
            """,
            (product_id, review_text, rating, sentiment)
        )
        conn.commit()
        #st.write("Review inserted successfully.")
    except Exception as e:
        st.error(f"Failed to insert review into the database: {e}")

def main_app():
    # Streamlit UI
    st.title("E-commerce Sentiment Analysis Application")

    # Step 1: Select a Category
    st.header("Select a Product Category")
    categories = load_categories()
    category_dict = {category[0]: category[1] for category in categories}
    category_id = st.selectbox("Choose a category", options=list(category_dict.keys()), format_func=lambda x: category_dict[x])

    if category_id:
        # Step 2: Select a Product
        st.header("Select a Product")
        products = load_products(category_id)
        product_dict = {product[0]: product[2] for product in products}
        product_id = st.selectbox("Choose a product", options=list(product_dict.keys()), format_func=lambda x: product_dict[x])

        if product_id:
            # Step 3: View Product Details
            st.header("Product Details")
            product, reviews = load_product_details(product_id)
            avg_rating = calculate_avg_rating(product_id)

            st.write(f"**Product Name:** {product[2]}")
            st.write(f"**Price:** ${product[3]}")
            st.write(f"**Product ID:** {product[0]}")
            st.write(f"**Average Rating:** {avg_rating}/5")

            # Ensure reviews are stored and refreshed dynamically
            if f"reviews_{product_id}" not in st.session_state:
                _, st.session_state[f"reviews_{product_id}"] = load_product_details(product_id)

            # Display past reviews in an expander
            with st.expander("Show Past Reviews"):
                reviews = st.session_state[f"reviews_{product_id}"]  # Load reviews from session state
                if reviews:
                    for review in reviews:
                        st.write(f"- {review[2]} (Rating: {review[3]}/5, Helpful votes: {review[4]})")
                else:
                    st.write("No reviews yet.")


            # Initialize session state for the text box if not already set
            if "new_review" not in st.session_state:
                st.session_state.new_review = ""
            # Step 4: Sentiment Analysis and Review Submission
            st.subheader("Submit a New Review")
            new_review = st.text_area("Write your review here:", value=st.session_state.new_review, key="review_input")
            rating = st.slider("Rate the product", min_value=1, max_value=5, value=5)

            

            if st.button("Submit Review"):
                if new_review:
                    # Analyze the review
                    tokenized_instances = tokenize_texts([new_review])
                    response = query_vertex_ai(tokenized_instances)

                    if response:
                        prediction = response.get("predictions", [])[0]
                        sentiment = classify_sentiment(prediction)

                        # Insert review into the database
                        insert_review(product_id, new_review, rating, sentiment)
                        st.success(f"Your review has been submitted! Sentiment: {sentiment}")
                        call_drift_detection_function()


                        # Refresh session state to include the new review
                        _, updated_reviews = load_product_details(product_id)  # Reload reviews
                        st.session_state[f"reviews_{product_id}"] = updated_reviews  # Update session state
                        # Clear the text box by resetting session state
                        st.session_state.new_review = ""
                    else:
                        st.error("Failed to analyze sentiment. Please try again.")
                else:
                    st.error("Please write a review before submitting.")



# Close the connection when done
try:
    conn.close()
    #st.write("Database connection closed.")
except Exception as e:
    st.error(f"Failed to close the database connection: {e}")
