# Amazon-Product-Review-Sentiment-Analyser

## Project Overview
This project aims to use Large language models and automate sentiment analysis for Amazon customer reviews using an MLOps pipeline that handles data ingestion, training, and deployment in a scalable and efficient way. The system is designed for continuous monitoring and retraining based on data drift.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation Instructions](#installation-instructions)
- [Usage Guidelines](#usage-guidelines)
- [Data Pipeline Setup](#data-pipeline-setup)
- [Model Details](#model-details)
- [Deployment](#deployment)
- [Monitoring and Drift Detection](#monitoring-and-drift-detection)


## Proposed Directory Structure
```
.
├── .dvc/                       # DVC configuration files
├── .github/workflows/          # GitHub Actions for CI/CD
├── Logging/                    # Logging setup for tracking model and data pipeline execution
├── Scraping/                   # Data scraping scripts
├── assets/                     # Assets like architecture diagrams, images
├── backend/                    # Backend services (e.g., API endpoints)
├── config/                     # Configuration files (e.g., environment variables)
├── dags/                       # Airflow DAGs for orchestrating tasks
├── data_drift/                 # Data drift detection scripts
├── data_pipeline/              # Scripts for data preprocessing and pipeline management
├── docs/                       # Documentation and architecture diagrams
├── frontend/                   # Frontend files for UI (if applicable)
├── mlflow/                     # MLflow experiment tracking
├── model/                      # Model training and evaluation scripts
├── model_endpoint/             # Model deployment endpoint (e.g., FastAPI or Flask)
├── model_training/             # Scripts for managing model training
├── docker-compose.yaml         # Docker Compose setup for development
└── README.md                   # Project README
```

## Installation Instructions
To install the required dependencies for this project, follow the steps below:

#### The installation instructions for data pipeline can be found in the readme for the data pipeline folder. [Instructions link](data_pipeline/README.md)


1. **Clone the repository:**
   ```bash
   git clone https://github.com/madhurima-vanga/Amazon-Customer-Sentiment-Analyser.git
   cd your-repo
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Docker:**
   Ensure Docker is installed and run the following command:
   ```bash
   docker-compose up -d
   ```

## Usage Guidelines
1. To run the data pipeline, refer to the readme in the data pipeline folder.

2. To train the model: to be implemented.

3. To deploy the model: to be implemented.

## Data Pipeline Setup
The data pipeline is set up using **Apache Airflow**. It is responsible for scraping, cleaning, and processing Amazon reviews data for sentiment analysis.

## Model Details
We are using pre-trained models like BERT, RoBERTa, or LLaMA for sentiment analysis, fine-tuned on Amazon reviews data. Model details and training scripts can be found in the `model/` directory.

## Deployment
The model is deployed using Docker and served via FastAPI. Run the following to deploy:
```bash
docker-compose -f docker-compose.yaml up -d
```

## Monitoring and Drift Detection
We use Prometheus and Grafana for monitoring the system and MLFlow for tracking experiments. Data drift is detected through regular checks using the scripts in `data_drift/`.


