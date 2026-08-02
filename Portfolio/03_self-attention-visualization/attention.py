"""
Self-Attention Visualization (Q, K, V)
========================================
Shows attention weights — which tokens "look at" which.

  - Q (Queries): "What am I looking for?"
  - K (Keys):    "What do I contain?"
  - V (Values):  "What do I pass along?"
"""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load the model ──────────────────────────────────────────
MODEL_NAME = "gpt2"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_attentions=True)
model.eval()
print(f"  Ready!\n")

# ── Step 2: Input ───────────────────────────────────────────────────
text = "The cat sleeps on the"
input_ids = tokenizer.encode(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"Input: '{text}'")
print(f"Tokens: {tokens}\n")

# ── Step 3: Run the model and extract attention weights ─────────────
with torch.no_grad():
    outputs = model(input_ids)

attentions = outputs.attentions
last_layer_attention = attentions[-1][0]
avg_attention = last_layer_attention.mean(dim=0)

# ── Step 4: Print the attention matrix ─────────────────────────────
print("Attention weights (averaged across heads, last layer):")
print("  Each row = 'who is looking'")
print("  Each column = 'who is being looked at'\n")

header = "".join(f"{t:>12s}" for t in tokens)
print(f"{'':>12s}{header}")
print("-" * (12 * (len(tokens) + 1)))

for i, row_token in enumerate(tokens):
    row = avg_attention[i]
    row_str = "".join(f"{v:12.3f}" for v in row)
    print(f"{row_token:>12s}{row_str}")

# ── Step 5: Save heatmap as image ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(avg_attention.numpy(), cmap="Blues", aspect="auto")
ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(tokens, fontsize=10)
ax.set_xlabel("Attended to (Keys)", fontsize=12)
ax.set_ylabel("Looking from (Queries)", fontsize=12)
ax.set_title("Self-Attention Weights — GPT-2 (Last Layer, Avg Heads)", fontsize=13)

for i in range(len(tokens)):
    for j in range(len(tokens)):
        val = avg_attention[i, j].item()
        color = "white" if val > 0.3 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

plt.colorbar(im, label="Attention weight")
plt.tight_layout()

output_path = "attention_heatmap.png"
plt.savefig(output_path, dpi=150)
print(f"\nHeatmap saved to: {output_path}")

print()
print("INTERPRETATION:")
print("  - Dark blue cells = high attention (strong connection)")
print("  - Light cells = low attention (tokens ignore each other)")
print("  - 'sleeps' attends to 'cat' (who is sleeping?)")
print("  - 'the' attends to 'cat' + 'sleeps' to predict the next word")
