import os
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
from datasets import load_dataset
import torch

hf_token = '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'

# Set CUDA memory management environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize model
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", use_auth_token=hf_token)
model.gradient_checkpointing_enable()

# Initialize pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer="meta-llama/Llama-2-7b-hf",
    device_map="auto",
    token=hf_token
)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    model_max_length=500,  # Reduce input length
    max_length=500,  # Reduce input length
    use_fast=False, 
    trust_remote_code=True,
    token=hf_token
)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_combined = dataset["text"]

# Prepare data
batch_size = 1  # Reduce batch size to save memory
sentence_group = [str(text_combined[0:500]) for _ in range(batch_size)]  # Reduce input length
inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,  # Explicitly set truncation to True
)
inputs = inputs.to("cuda:0")

# Clear CUDA cache
torch.cuda.empty_cache()

# Generate text using pipeline
print("begin")
outputs = pipe(sentence_group, max_new_tokens=100)  # Reduce max_new_tokens

# Display results
for output in outputs:
    print(output['generated_text'])
