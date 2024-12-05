import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification

# Load the saved model and tokenizer
MODEL_PATH = "./distillbert_sentiment_model"
new_model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Sentiment labels
LABELS = {0: "Positive", 1: "Negative", 2: "Neutral"}

def classify_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=256)
    
    # Perform prediction
    outputs = new_model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=1)
    predicted_class = tf.argmax(probabilities, axis=1).numpy()[0]
    
    # Map the predicted class to sentiment label
    sentiment = LABELS[predicted_class]
    return sentiment

if __name__ == "__main__":
    print("Enter a sentence to classify sentiment (type 'exit' to quit):")
    while True:
        user_input = input(">> ")
        if user_input.lower() == "exit":
            print("Exiting sentiment analysis.")
            break
        sentiment = classify_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")
