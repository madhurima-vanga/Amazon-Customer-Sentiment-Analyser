import google.cloud.aiplatform as aiplatform

def deploy_tensorflow_model_to_endpoint(
    project_id: str,
    region: str,
    model_name: str,
    model_artifact_uri: str,
    machine_type: str = "n1-standard-2",
):
    """
    Deploy a TensorFlow model from GCP bucket to Vertex AI endpoint.

    Args:
        project_id (str): Google Cloud project ID.
        region (str): Google Cloud region (e.g., 'us-central1').
        model_name (str): Name of the model in the registry.
        model_artifact_uri (str): URI of the model artifact in Google Cloud Storage.
        machine_type (str): Machine type for deployment. Default is 'n1-standard-2'.

    Returns:
        endpoint (google.cloud.aiplatform.Endpoint): Deployed endpoint object.
    """
    # Initialize the AI Platform client
    aiplatform.init(project=project_id, location=region)

    # Register the TensorFlow model in the Model Registry
    print("Uploading TensorFlow model to Model Registry...")
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest",  # TensorFlow 2.11 CPU container
    )
    print(f"Model registered with ID: {model.resource_name}")

    # Create an endpoint for the model
    print("Creating an endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name=f"{model_name}-endpoint")
    print(f"Endpoint created with ID: {endpoint.resource_name}")

    # Deploy the model to the endpoint
    print("Deploying model to the endpoint...")
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{model_name}-deployment",
        machine_type=machine_type,
    )
    print(f"Model deployed to endpoint: {endpoint.resource_name}")

    return endpoint


if __name__ == "__main__":
    # Replace these variables with your configuration
    PROJECT_ID = "wired-glider-441301-e1"
    REGION = "us-east4"
    MODEL_NAME = "amazon-sentiment-model"
    MODEL_ARTIFACT_URI = "gs://amazon_sentiment_analysis/distillbert-model-artifacts-final/tensorflow"  # Path to folder containing saved_model.pb

    deploy_tensorflow_model_to_endpoint(
        project_id=PROJECT_ID,
        region=REGION,
        model_name=MODEL_NAME,
        model_artifact_uri=MODEL_ARTIFACT_URI,
    )
