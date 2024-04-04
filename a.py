import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Authenticate with the Hugging Face model hub
from transformers import logging
logging.set_verbosity_error()  # Suppress logging messages

checkpoint = "bigcode/starcoder2-3b"
device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available, otherwise use CPU

# Load tokenizer with authentication
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# Set pad_token_id to eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# Create inputs tensor as requested
input_text = "provide code of factorial of n in python language "
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate outputs with attention_mask set
attention_mask = torch.ones_like(inputs)
attention_mask[:, 0] = 0  # Set attention mask to 0 for the first token (CLS token)
outputs = model.generate(inputs, attention_mask=attention_mask,max_length=100)

# Decode the generated output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
