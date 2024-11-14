from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DistilBertTokenizer

# Load preprocessed tokenized dataset from GCS
dataset = load_from_disk("gs://amazon_sentiment_analysis/distilbert/processed_data/tokenized_amazon_reviews")

# Load DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./distilbert_sentiment_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./distilbert_sentiment_model")
tokenizer.save_pretrained("./distilbert_sentiment_model")

