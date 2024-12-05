from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load dataset
file_path = 'model_train.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Ensure dataset contains only relevant columns and drop NaNs
data = data[['cleaned_text', 'label']].dropna()

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['cleaned_text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Check class distribution
from collections import Counter
print("Class distribution in training set:", Counter(train_labels))

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize datasets
max_length = 256  # Desired max length for tokens
def tokenize_texts(texts, labels, tokenizer, max_length):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings, tf.convert_to_tensor(labels)

train_encodings, train_labels_tensor = tokenize_texts(train_texts, train_labels, tokenizer, max_length)
val_encodings, val_labels_tensor = tokenize_texts(val_texts, val_labels, tokenizer, max_length)

# Load pre-trained DistilBERT model
model = TFDistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Define optimizer with gradient clipping and warm-up steps
from transformers import create_optimizer

batch_size = 8
epochs = 1
num_train_steps = len(train_texts) // batch_size * epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_warmup_steps=100,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01
)

# Compile the model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [
    tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    x={
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask']
    },
    y=train_labels_tensor,
    validation_data=(
        {
            'input_ids': val_encodings['input_ids'],
            'attention_mask': val_encodings['attention_mask']
        },
        val_labels_tensor
    ),
    epochs=epochs,
    batch_size=batch_size,
    class_weight=class_weights,  # Apply class weights
    callbacks=[early_stopping]
)

# Save the model in Hugging Face format
model.save_pretrained('./distillbert_sentiment_model')
tokenizer.save_pretrained('./distillbert_sentiment_model')
print("Model and tokenizer saved to Hugging Face format.")

# Save the model in TensorFlow SavedModel format
saved_model_dir = './distillbert_sentiment_model'

# Define the serving function with input signature
@tf.function(input_signature=[
    {
        'input_ids': tf.TensorSpec(shape=(None, max_length), dtype=tf.int32, name='input_ids'),
        'attention_mask': tf.TensorSpec(shape=(None, max_length), dtype=tf.int32, name='attention_mask')
    }
])
def serving_fn(inputs):
    return model(inputs)

# Save to TensorFlow SavedModel format
tf.saved_model.save(model, saved_model_dir, signatures={'serving_default': serving_fn})
print(f"Model saved in TensorFlow SavedModel format at: {os.path.abspath(saved_model_dir)}")

# Evaluate model and generate metrics
val_preds = model.predict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask']
})['logits']
val_preds_labels = tf.argmax(tf.nn.softmax(val_preds, axis=1), axis=1).numpy()

# Classification report
print("Classification Report:")
print(classification_report(val_labels, val_preds_labels, target_names=["Positive", "Negative", "Neutral"]))

# Confusion matrix
conf_matrix = confusion_matrix(val_labels, val_preds_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative', 'Neutral'], yticklabels=['Positive', 'Negative', 'Neutral'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
