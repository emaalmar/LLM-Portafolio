"""
Autoregressive Text Generation
================================
Shows how an LLM builds text token by token, feeding its own
output back as input — "the predicted variable regresses on itself."

This is the core loop behind ChatGPT, Claude, Gemini, etc.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load the model ──────────────────────────────────────────
MODEL_NAME = "gpt2"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print(f"  Ready! ({model.num_parameters() / 1e6:.0f}M parameters)\n")

# ── Step 2: Define input + generation settings ─────────────────────
prompt = "The future of healthcare is"
max_tokens = 30

print(f"Prompt: '{prompt}'\n")
print("=" * 60)
print("  AUTOREGRESSIVE LOOP — watch text build word by word")
print("=" * 60)

input_ids = tokenizer.encode(prompt, return_tensors="pt")
generated = input_ids.clone()

# ── Step 3: The autoregressive loop ────────────────────────────────
for step in range(max_tokens):
    with torch.no_grad():
        outputs = model(generated)
        logits = outputs.logits

    next_token_logits = logits[0, -1, :]

    temperature = 0.8
    next_token_logits = next_token_logits / temperature

    probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.multinomial(probs, 1)

    generated = torch.cat([generated, next_token_id.unsqueeze(0)], dim=1)

    current_text = tokenizer.decode(generated[0])
    new_token = tokenizer.decode([next_token_id.item()])
    print(f"  Step {step + 1:2d}: {new_token:>6s}  ->  {current_text}")

print()
print("=" * 60)
print("  FINAL OUTPUT:")
print("=" * 60)
print(f"  {tokenizer.decode(generated[0])}")
print()
print("WHAT JUST HAPPENED:")
print("  1. The model read 'The future of healthcare is'")
print("  2. It predicted one token -> fed it back in")
print("  3. Repeated 30 times -> text grew token by token")
print("  4. Each prediction was based on ALL previous tokens")
