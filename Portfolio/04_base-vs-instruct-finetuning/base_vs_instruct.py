"""
Base vs Instruct — Why Fine-Tuning Matters
============================================
The same model, same prompt, different behavior:
  - GPT-2 (base) = trained on raw text → monologue style
  - Qwen2.5-0.5B-Instruct = SFT + aligned → conversational

This demonstrates the CORE insight from Class 4 (5 Stages of an LLM):
  - Stage 3 (pre-training) gives you a base model
  - Stage 4 (fine-tuning / alignment) gives you an instruct model
  - Without fine-tuning, the model just continues text like autocomplete

Concepts:
  - Tokenization (BPE) — how text becomes numbers the model processes
  - Pre-training — next-token prediction on raw web text
  - Fine-tuning — SFT (Supervised Fine Tuning) + RLHF/DPO alignment
  - Base vs Instruct — the difference between "continues text" and "answers questions"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════
# Load both models
# ══════════════════════════════════════════════════════════════════════

print("Loading models...")

BASE_MODEL = "gpt2"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# GPT-2: base model (pre-trained only, no fine-tuning)
base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
base_model.eval()

# Qwen2.5-0.5B-Instruct: fine-tuned with SFT + RLHF
instruct_tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
instruct_model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL)
instruct_model.eval()

print(f"  GPT-2: {base_model.num_parameters() / 1e6:.0f}M params (base model)")
print(f"  Qwen:  {instruct_model.num_parameters() / 1e6:.0f}M params (instruct model)")
print()

# ══════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def generate_from_base(prompt, max_new_tokens=80):
    """Generate from the base model (raw text continuation)."""
    inputs = base_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=base_tokenizer.eos_token_id,
        )
    return base_tokenizer.decode(output[0], skip_special_tokens=True)

def generate_from_instruct(prompt, max_new_tokens=80):
    """Generate from the instruct model (with chat template)."""
    messages = [{"role": "user", "content": prompt}]
    text = instruct_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = instruct_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = instruct_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=instruct_tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return instruct_tokenizer.decode(new_tokens, skip_special_tokens=True)

# ══════════════════════════════════════════════════════════════════════
# The same prompts, two very different models
# ══════════════════════════════════════════════════════════════════════

prompts = [
    "What is the capital of Japan?",
    "Summarize the benefits of exercise in one sentence.",
    "Explain what an LLM is to a 5-year-old.",
]

for i, prompt in enumerate(prompts, 1):
    print(f"{'=' * 65}")
    print(f"  PROMPT {i}: '{prompt}'")
    print(f"{'=' * 65}")
    print()

    base_response = generate_from_base(prompt)
    instruct_response = generate_from_instruct(prompt)

    print(f"  GPT-2 (base model, 124M):")
    print(f"  {base_response}")
    print()
    print(f"  Qwen2.5 (instruct model, 500M):")
    print(f"  {instruct_response}")
    print()

# ══════════════════════════════════════════════════════════════════════
# Why does this happen?
# ══════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  WHY IS GPT-2 ACTING WEIRD?")
print("=" * 65)
print()
print("  GPT-2 was trained to PREDICT THE NEXT TOKEN in web text.")
print("  It doesn't 'understand' questions — it just continues patterns.")
print()
print("  When you ask 'What is the capital of Japan?':")
print("    - GPT-2 sees this as a QUESTION (pattern from web text)")
print("    - It continues with more questions/answers (like Wikipedia)")
print("    - It doesn't know it should answer YOUR question")
print()
print("  Qwen2.5-Instruct was:")
print("    1. Pre-trained (like GPT-2) — learns language patterns")
print("    2. Fine-tuned on instruction-response pairs (SFT)")
print("    3. Aligned with RLHF — learned to answer, not to monologue")
print()

# ══════════════════════════════════════════════════════════════════════
# The 5 stages of an LLM — where we are
# ══════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  THE 5 STAGES OF AN LLM")
print("=" * 65)
print()
print("  Stage 1: Get a huge dataset")
print("    - Common Crawl, books, code, papers, social media")
print()
print("  Stage 2: Tokenize the dataset")
print("    - BPE (Byte Pair Encoding) — words → numbers")
print()
print("  Stage 3: Pre-train the model")
print("    - Embeddings + Attention + Transformer = base model")
print()
print("  Stage 4: Fine-tune the model")
print("    - SFT (Supervised Fine Tuning) + RLHF/DPO")
print()
print("  Stage 5: Evaluate the model")
print("    - Benchmarks (MMLU, GPQA), human eval, evals tools")
print()
print("  THIS DEMO SHOWS STAGE 4 IN ACTION!")
print("  GPT-2 = Stage 3 only (base model)")
print("  Qwen  = Stage 3 + Stage 4 (instruct model)")
print()
print("=" * 65)
print("  KEY TAKEAWAYS")
print("=" * 65)
print()
print("  1. Base models autocomplete; instruct models answer.")
print("  2. Fine-tuning turns a text predictor into a conversation partner.")
print("  3. SFT = teach with examples; RLHF/DPO = teach with preferences.")
print("  4. GPT-2 isn't 'stupid' — it's just not aligned for dialogue.")
print("  5. In 2026: all chatbots are instruct models.")
print("     Base models are only used for training/research.")
