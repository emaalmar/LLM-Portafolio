"""
Next-Token Prediction Demo
===========================
Shows how an LLM predicts the next token given a sequence.

Run: python next_token.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
MODEL_NAME = "gpt2"
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print(f"  Model loaded! ({model.num_parameters() / 1e6:.0f}M parameters)\n")

# Input
text = "The cat sleeps on the"
input_ids = tokenizer.encode(text, return_tensors="pt")

print(f"Input:          '{text}'")
print(f"Token IDs:      {input_ids[0].tolist()}")
print(f"Tokens:         {tokenizer.convert_ids_to_tokens(input_ids[0])}\n")

# Predict next token
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

next_token_logits = logits[0, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)

# Show top 5
TOP_K = 5
top_probs, top_indices = torch.topk(probs, TOP_K)

print(f"Top {TOP_K} predicted next tokens:")
print("-" * 40)
for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
    token_text = tokenizer.decode([idx.item()])
    bar = "█" * int(prob.item() * 40)
    print(f"  {rank}. '{token_text}' — {prob.item():.1%}  {bar}")

print(f"\nBest prediction: '{text} {tokenizer.decode([top_indices[0].item()])}'")
