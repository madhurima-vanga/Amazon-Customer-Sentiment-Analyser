# Start with a base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
# Copy model files into the container
COPY ./distilbert_sentiment_model /app/model

# Install necessary libraries
RUN  pip install torch transformers

# Run a simple API or serve the model
CMD ["python", "-c", "from transformers import RobertaForSequenceClassification, RobertaTokenizer; model = RobertaForSequenceClassification.from_pretrained('/app/model'); tokenizer = RobertaTokenizer.from_pretrained('/app/model'); print('Model loaded successfully!')"]