import torch
from transformers import AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import AlbertTokenizer

# Check if CUDA (GPU) is available, change the device to gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load preprocessed tokenized dataset from GCS
dataset = load_from_disk("gs://amazon_sentiment_analysis/albert/processed_data/tokenized_amazon_reviews")

# Load ALBERT model for sequence classification
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3).to(device)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./albert_sentiment_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True
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
trainer.save_model("./albert_sentiment_model")
tokenizer.save_pretrained("./albert_sentiment_model")

