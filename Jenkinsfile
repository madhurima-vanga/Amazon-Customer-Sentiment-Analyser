pipeline {
    agent any

    environment {
        // Define environment variables
        MLFLOW_TRACKING_URI = "http://35.193.202.32:5000"
        NUM_EPOCHS = "1"
        EMAIL_ADDRESS = "sentimentanalysisreviewproject@gmail.com"
        MODEL_DIR = "./model_output/distilbert_sentiment_model"
        DOCKER_IMAGE = "sentimentanalysis/mlops-sentiment-analysis:latest"  // Docker Hub image URL
        DOCKER_REGISTRY = "https://index.docker.io/v1/"  // Docker Hub registry
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

        stage('Authenticate to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                    sh """
                        echo $DOCKER_PASSWORD | docker login --username $DOCKER_USERNAME --password-stdin
                    """
                }
            }
        }

        stage('Push Docker Image to Docker Hub') {
            steps {
                script {
                    // Push the Docker image to Docker Hub
                    echo "Pushing Docker image to Docker Hub..."
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
        success {
            script {
                // Send email for success
                emailext(
                    subject: "Pipeline Success: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                    body: "The Jenkins pipeline has completed successfully.\n\nJob: ${env.JOB_NAME}\nBuild: ${env.BUILD_NUMBER}\nStatus: SUCCESS",
                    to: "${EMAIL_ADDRESS}"
                )
            }
        }
        failure {
            script {
                // Send email for failure
                emailext(
                    subject: "Pipeline Failure: ${env.JOB_NAME} [${env.BUILD_NUMBER}]",
                    body: "The Jenkins pipeline has failed.\n\nJob: ${env.JOB_NAME}\nBuild: ${env.BUILD_NUMBER}\nStatus: FAILURE\nPlease check the logs for details.",
                    to: "${EMAIL_ADDRESS}"
                )
            }
        }
        always {
            script {
                echo "Pipeline completed!"
            }
        }
    }

}
