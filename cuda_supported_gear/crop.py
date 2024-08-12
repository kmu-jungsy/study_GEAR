import torch

def crop_prompt(prompt: str, tokenizer, max_length: int = 1000):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.decode(tokens)

def crop(prompt: str, tokenizer, max_length: int = 1000):
    cropped_prompt = crop_prompt(prompt, tokenizer, max_length)
    if len(tokenizer.encode(cropped_prompt)) == len(tokenizer.encode(prompt)):
        return prompt  # No change after cropping
    return cropped_prompt
