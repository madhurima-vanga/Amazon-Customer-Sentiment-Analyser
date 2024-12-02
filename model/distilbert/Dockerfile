# Start with a base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
# Copy model files into the container
COPY ./model_output/distilbert_sentiment_model /app/model

# Install necessary libraries
RUN  pip install transformers

# Run a simple API or serve the model
CMD ["python3", "-c", "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model = AutoModelForSequenceClassification.from_pretrained('/app/model'); tokenizer = AutoTokenizer.from_pretrained('/app/model'); print('Model loaded successfully!')"]
