"""
Embeddings & Polysemy: "One embedding is not enough"
======================================================
From Class 4 (Stage 3 — how the model represents meaning):

  A static embedding table gives each token ONE fixed vector.
  But words are polysemous: "bat" is an animal OR a baseball tool.

  This script PROVES that GPT-2 does NOT rely on one static vector:
  it builds a CONTEXTUAL embedding that depends on the sentence.

  How:
    1. Take "bat" in animal sentences and in sports sentences.
    2. Extract the hidden state at the position of "bat".
    3. Measure L2 distances:
         - same meaning (animal vs animal)   -> close
         - different meaning (animal vs sports) -> far

  Design note: "bat" is placed LATE in each sentence because GPT-2
  uses causal (left-to-right) attention — only context BEFORE the
  word can disambiguate it.
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load the model ──────────────────────────────────────────
MODEL_NAME = "gpt2"

print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print(f"  Ready! Embedding dimension: {model.transformer.wte.weight.shape[-1]}\n")

# ── Step 2: Two meanings of "bat" ───────────────────────────────────
TARGET = "bat"

ANIMAL = [
    "The animal that flew at night was a bat.",
    "The small creature in the cave was a bat.",
    "The flying mammal that squeaked was a bat.",
]
SPORTS = [
    "The tool that he hit the ball with was a bat.",
    "The wooden stick he gripped was a bat.",
    "What he swung at the plate was a bat.",
]

# ── Step 3: Extract embeddings ──────────────────────────────────────
def find_token_pos(sentence):
    """Position of the token containing the target word."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(sentence))
    for i, tok in enumerate(tokens):
        if TARGET in tok.lower():
            return i
    raise ValueError(f"'{TARGET}' not found in: {sentence}")

def static_embedding():
    """Dictionary lookup — the SAME vector for every context."""
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(TARGET))
    tok_id = next(i for i, t in enumerate(tokens) if TARGET in t.lower())
    token_id = tokenizer.convert_tokens_to_ids(tokens[tok_id])
    return model.transformer.wte.weight[token_id].detach()

def contextual_embedding(sentence):
    """Hidden state at the target position — depends on the sentence."""
    ids = tokenizer.encode(sentence, return_tensors="pt")
    pos = find_token_pos(sentence)
    with torch.no_grad():
        hidden = model(ids, output_hidden_states=True).hidden_states[-1]
    return hidden[0, pos].detach()

def l2_dist(a, b):
    return torch.norm(a - b).item()

# ── Step 4: Show the static embedding is identical ───────────────────
stat = static_embedding()
print("=" * 65)
print("  Static embedding of 'bat' (dictionary lookup)")
print("=" * 65)
print(f"  A vector of {len(stat)} numbers, e.g. first 5: "
      f"{[round(x.item(), 3) for x in stat[:5]]}")
print("  This vector is IDENTICAL in every sentence.")
print("  It cannot tell 'animal bat' from 'baseball bat'.\n")

# ── Step 5: Extract contextual embeddings ───────────────────────────
print("=" * 65)
print("  Contextual embeddings — the proof")
print("=" * 65)

animal_ctx = [contextual_embedding(s) for s in ANIMAL]
sports_ctx = [contextual_embedding(s) for s in SPORTS]
all_ctx = animal_ctx + sports_ctx
labels = [f"{c}{i}" for c in ("A", "S") for i in (1, 2, 3)]

# ── Step 6: Distance matrix ─────────────────────────────────────────
print("  L2 distance matrix of 'bat' embeddings (smaller = closer):\n")
header = "        " + "".join(f"{l:>8}" for l in labels)
print(header)
print("        " + "-" * (8 * len(labels) + 1))
for i, (lab, vec) in enumerate(zip(labels, all_ctx)):
    row = f"  {lab:<5} "
    for other in all_ctx:
        row += f"{l2_dist(vec, other):8.1f}"
    print(row)

within_animal = np.mean([l2_dist(animal_ctx[i], animal_ctx[j])
                         for i in range(3) for j in range(3) if i != j])
within_sports = np.mean([l2_dist(sports_ctx[i], sports_ctx[j])
                         for i in range(3) for j in range(3) if i != j])
between = np.mean([l2_dist(animal_ctx[i], sports_ctx[j])
                   for i in range(3) for j in range(3)])

print()
print("  KEY NUMBERS:")
print(f"    within-meaning (animal vs animal):   {within_animal:5.1f}")
print(f"    within-meaning (sports vs sports):   {within_sports:5.1f}")
print(f"    between-meaning (animal vs sports):  {between:5.1f}")
print(f"    ratio (between / within):            {between / ((within_animal + within_sports) / 2):.2f}x\n")

# ── Step 7: Heatmap ─────────────────────────────────────────────────
dist_matrix = np.array([[l2_dist(a, b) for b in all_ctx] for a in all_ctx])

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(dist_matrix, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Context of 'bat'", fontsize=12)
ax.set_ylabel("Context of 'bat'", fontsize=12)
ax.set_title("Distance between contextual embeddings of 'bat' (L2)", fontsize=13)
ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, edgecolor="white", lw=2))
ax.add_patch(plt.Rectangle((2.5, 2.5), 3, 3, fill=False, edgecolor="white", lw=2))
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{dist_matrix[i, j]:.0f}",
                ha="center", va="center", fontsize=8,
                color="white" if dist_matrix[i, j] > 20 else "black")
plt.colorbar(im, label="L2 distance")
plt.tight_layout()

output_path = "embeddings_distance_heatmap.png"
plt.savefig(output_path, dpi=150)
print(f"Heatmap saved to: {output_path}\n")

# ── Step 8: Interpretation ──────────────────────────────────────────
print("=" * 65)
print("  INTERPRETATION")
print("=" * 65)
print("  The dark-blue 3x3 blocks (top-left, bottom-right) are the")
print("  SAME-meaning groups: 'bat' sentences close together.")
print("  The lighter region between them shows DIFFERENT meanings.")
print()
print("  This proves the Class 4 point 'one embedding is not enough':")
print("  GPT-2 turns the static vector into a contextual one that")
print("  depends on surrounding words — same token, different meaning,")
print("  different coordinates in meaning-space.")
