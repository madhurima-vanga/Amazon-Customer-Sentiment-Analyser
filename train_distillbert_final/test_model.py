import tensorflow as tf
from transformers import DistilBertTokenizer
import pandas as pd
from sklearn.metrics import classification_report

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model
#model = tf.saved_model.load("C:\Amazon-Customer-Sentiment-Analyser\train_distilbert\distillbert_sentiment_model\saved_model.pb")
model = tf.saved_model.load("./distillbert_sentiment_model")


# Load the test data
data = pd.read_csv("model_test.csv")

# Ensure column names match the CSV structure
text_column = "cleaned_text"
label_column = "label"

# Tokenize the data
max_seq_length = 256
def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="tf"
    )

tokenized_inputs = tokenize_texts(data[text_column])

# Get predictions
predictions = model(tokenized_inputs)["logits"]
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Evaluate the results
true_labels = data[label_column].values
print(classification_report(true_labels, predicted_labels, target_names=["Positive", "Negative", "Neutral"]))
