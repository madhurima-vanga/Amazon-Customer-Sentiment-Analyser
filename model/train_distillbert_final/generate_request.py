from transformers import DistilBertTokenizer
import json

# Load a pre-trained tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Input text
text = "This is a bad movie, highly dissapointed!"

# Maximum sequence length for the model
max_seq_length = 256

# Tokenize the input text with padding and truncation
tokenized_data = tokenizer(
    text,
    padding="max_length",  # Pads the sequence to max_seq_length
    truncation=True,       # Truncates if the sequence exceeds max_seq_length
    max_length=max_seq_length,
    return_tensors="tf"    # Outputs TensorFlow tensors
)

# Convert tensors to lists for JSON serialization
input_ids = tokenized_data["input_ids"].numpy().squeeze().tolist()
attention_mask = tokenized_data["attention_mask"].numpy().squeeze().tolist()

# Create the JSON structure
json_data = {
    "instances": [
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    ]
}

# Write the JSON data to a file
output_file = "input_data.json"
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSON data saved to {output_file}")
