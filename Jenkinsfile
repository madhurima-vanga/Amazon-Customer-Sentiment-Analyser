pipeline {
    agent any

    environment {
        // Define environment variables
        MLFLOW_TRACKING_URI = "http://35.193.202.32:5000"
        NUM_EPOCHS = "1"
        EMAIL_ADDRESS = "sentimentanalysisreviewproject@gmail.com"
        MODEL_DIR = "./model_output/distilbert_sentiment_model"
        DOCKER_IMAGE = "us-east1-docker.pkg.dev/wired-glider-441301-e1/sentiment-analysis-model/distilbert-sentiment-analysis:latest"
        DOCKER_REGISTRY = "us-east1-docker.pkg.dev/wired-glider-441301-e1/sentiment-analysis-model"  // Artifact registry URL

    }

    stages {
        stage('Setup Python Environment') {
            steps {
                script {
                    // Set up Python virtual environment and install dependencies
                    sh '''
                    python3 -m venv .venv
                    bash -c "source .venv/bin/activate && pip install --upgrade pip && pip install -r model/distilbert/requirements.txt"
                    '''
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    // Train the model
                    sh '''
                    bash -c "source .venv/bin/activate && python3 model/distilbert/train.py"
                    mkdir -p model_output
                    cp -r ./distilbert_sentiment_model model_output/
                    '''
                }
            }
        }

        stage('Validate Model') {
            steps {
                script {
                    // Validate the model
                    sh '''
                    bash -c "source .venv/bin/activate && python3 model/distilbert/test.py"
                    '''
                }
            }
        }

        stage('Perform Bias Detection') {
            steps {
                script {
                    // Perform bias detection
                    sh '''
                    bash -c "source .venv/bin/activate && python3 model/distilbert/test_bias.py"
                    '''
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image
                    echo "Building Docker image..."
                    sh '''
                    cp -r model_output/ model/distilbert/

                    docker build -t $DOCKER_IMAGE -f model/distilbert/Dockerfile model/distilbert
                    '''
                }
            }
        }

        stage('Authenticate to Google Cloud') {
            steps {
                withCredentials([file(credentialsId: '71764a53-ad36-4dae-b44c-856d3c7adb3d', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh """
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud auth configure-docker
                    """
                }
            }
        }


        stage('Push Docker Image') {
            steps {
                script {
                    // Push the Docker image to Artifact Registry
                    echo "Pushing Docker image to Artifact Registry..."
                    sh '''
                    docker push $DOCKER_IMAGE
                    '''
                }
            }
        }

        stage('Send Email Notification') {
            steps {
                script {
                    // Send email notification (use a plugin or external script)
                    echo "Email notification to $EMAIL_ADDRESS"
                }
            }
        }
    }

    post {
        always {
            script {
                echo "Pipeline completed!"
            }
        }
    }
}
