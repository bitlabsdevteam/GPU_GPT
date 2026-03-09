import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer

# Check if multiple GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Move the model to GPU if available
model.to(device)

# Use DataParallel for multi-GPU data parallelism
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
    model = nn.DataParallel(model)

# Sample input text
texts = ["Hello, how are you?", "Data parallelism with GPT-2 model."]

# Tokenize input texts
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Move inputs to the same device as the model
inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass
outputs = model(**inputs)

# The outputs are a tuple, where outputs[0] is last_hidden_state
last_hidden_state = outputs.last_hidden_state

print("Last hidden states shape:", last_hidden_state.shape)

# Example usage: Mean pooling of the last hidden state
pooled_output = last_hidden_state.mean(dim=1)
print("Pooled output shape:", pooled_output.shape)

# End of example script
