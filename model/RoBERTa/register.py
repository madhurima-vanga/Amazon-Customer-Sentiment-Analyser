import os
import torch
import pickle
from google.cloud import storage, aiplatform
import logging
from datetime import datetime
import io

# Configure logging
def configure_logging():
    log_dir = os.path.join("logs", "register")
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"register_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_filename,
        filemode="w",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)

configure_logging()

# Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_JSON_Key"
project_id = "wired-glider-441301-e1"
bucket_name = "amazon_sentiment_analysis"
region = "us-central1"
gcs_model_directory = "saved_models/"
model_display_name = "roberta"
container_image_uri = "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-9:latest"

# Load model from GCS
def load_model_from_gcs(bucket_name, model_filepath):
    client = storage.Client()
    model_data = client.bucket(bucket_name).blob(model_filepath).download_as_bytes()
    return torch.load(io.BytesIO(model_data))

# Upload file to GCS
def upload_to_gcs(local_file_path, bucket_name, destination_blob_name):
    storage.Client(project=project_id).bucket(bucket_name).blob(destination_blob_name).upload_from_filename(local_file_path)
    logging.info("Uploaded to GCS: gs://%s/%s", bucket_name, destination_blob_name)

# Register model in Vertex AI
def register_model_in_vertex_ai():
    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=f"gs://{bucket_name}/{gcs_model_directory}",
        serving_container_image_uri=container_image_uri,
    )
    logging.info("Registered model in Vertex AI: %s", model.resource_name)

# Main workflow
try:
    logging.info("Loading model from GCS")
    model = load_model_from_gcs(bucket_name, "saved_models/model.pth")
    
    pkl_model_path = "model.pkl"
    with open(pkl_model_path, "wb") as f:
        pickle.dump(model, f)
    logging.info("Converted and saved model as: %s", pkl_model_path)

    gcs_model_path = f"{gcs_model_directory}model.pkl"
    upload_to_gcs(pkl_model_path, bucket_name, gcs_model_path)
    register_model_in_vertex_ai()
except Exception as e:
    logging.error("Error occurred: %s", e)
