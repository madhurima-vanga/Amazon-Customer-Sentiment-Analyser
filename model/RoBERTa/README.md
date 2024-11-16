# Bias Mitigation and Model Evaluation Workflow

This repository contains a comprehensive pipeline for data splitting, bias detection, mitigation, model training, and deployment using Google Cloud services and state-of-the-art machine learning frameworks.

## Project Structure

```plaintext
├── data_split.py           # Script for splitting data into train, test, and inference sets
├── bias.py                 # Script for bias detection and evaluation
├── bias_mitigation.py      # Script for mitigating bias in data
├── model.py                # Script for model training and evaluation
├── register.py             # Script for model registration and deployment
├── logs/                   # Directory for storing log files
```

## Prerequisites

- **Python Version**: 3.8 or above
- **Google Cloud SDK**: Installed and authenticated
- **Environment Variable**: `GOOGLE_APPLICATION_CREDENTIALS` set to a valid GCP service account key
- **Dependencies**: Install via `requirements.txt`:
  - `torch`, `transformers`, `mlflow`, `google-cloud-storage`, `pandas`, `sklearn`, `datasets`

## Workflow Overview

### 1. Data Splitting (`data_split.py`)

This script:
- Loads a dataset from Google Cloud Storage (GCS).
- Maps sentiment labels (`positive`, `negative`, `neutral`) to numeric labels (`0`, `1`, `2`).
- Splits the data into balanced train, test, and inference sets.
- Uploads the split datasets back to GCS.

#### Key Outputs:
- **Train set**: `gs://<bucket_name>/model/train/train.csv`
- **Test set**: `gs://<bucket_name>/model/test/test.csv`
- **Inference set**: `gs://<bucket_name>/model/validate/inference.csv`

---

### 2. Bias Detection (`bias.py`)

This script:
- Evaluates model performance across slices of sensitive features (e.g., categories).
- Uses metrics such as fairness disparity to detect potential bias.

#### Key Metrics:
- **Accuracy per slice**
- **Fairness-related metrics** (e.g., parity, disparate impact)

---

### 3. Bias Mitigation (`bias_mitigation.py`)

This script:
- Balances classes within each sensitive feature (e.g., product category).
- Resamples data to ensure equal representation of all classes.
- Calculates sample weights for reweighting.
- Uploads the revised dataset to GCS.

#### Key Outputs:
- **Revised dataset**: `gs://<bucket_name>/bias_mitigation/train_bias_mitigation.csv`

---

### 4. Model Training (`model.py`)

This script:
- Loads and tokenizes data using Hugging Face `transformers`.
- Fine-tunes a RoBERTa model on the training dataset.
- Evaluates the model on the test dataset.
- Logs metrics and artifacts to MLflow.

#### Key Features:
- Integrated with **MLflow** for experiment tracking.
- Supports **GPU acceleration** for training.

#### Key Outputs:
- **Fine-tuned model** and tokenizer: `./fine_tuned_roberta`
- **MLflow logs and artifacts**.

---

### 5. Model Registration and Deployment (`register.py`)

This script:
- Converts the trained model to a format compatible with Vertex AI.
- Uploads the model to GCS.
- Registers the model in Vertex AI for serving.

#### Key Outputs:
- **Serialized model**: `gs://<bucket_name>/saved_models/model.pkl`
- **Registered model** in Vertex AI.

---

## How to Run

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up Google Cloud credentials:

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-service-account-key.json"
   ```

4. Execute the scripts in the following order:

   **Step 1**: Data Splitting
   ```bash
   python data_split.py
   ```

   **Step 2**: Bias Detection
   ```bash
   python bias.py
   ```

   **Step 3**: Bias Mitigation
   ```bash
   python bias_mitigation.py
   ```

   **Step 4**: Model Training
   ```bash
   python model.py
   ```

   **Step 5**: Model Registration and Deployment
   ```bash
   python register.py
   ```

---

## Logging

- All scripts log information to the `logs/` directory.
- Logs include timestamps, script-specific details, and error handling information.

---

## Notes

- Ensure all bucket names, file paths, and project IDs are updated according to your GCP configuration.
- Modify training parameters in `model.py` as needed for your dataset and computational resources.
- export GOOGLE_APPLICATION_CREDENTIALS="../wired-glider-441301-e1-b6a080d0f979.json" this file need to be present for GCP auth.

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.
