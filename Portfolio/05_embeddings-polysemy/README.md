# Project 5: Embeddings & Polysemy — "One Embedding Is Not Enough"

**What it does:** Shows how GPT-2 represents the word "bat" differently depending on context, proving that a single static embedding can't capture all meanings of a word.

**Why it matters:** This is Stage 3 of how an LLM works — the bridge between tokens and meaning. Understanding contextual embeddings explains why the same word changes meaning in different sentences, and why modern LLMs can understand ambiguity.

## Concepts Demonstrated

- **Static embeddings:** a dictionary lookup — one fixed vector per token
- **Contextual embeddings:** the model modifies the vector based on surrounding words
- **Polysemy:** one word ("bat") with multiple meanings ("animal" vs "baseball tool")
- **Causal attention:** GPT-2 only "sees" context to the left, so the disambiguating clue must come before the target word
- **Similarity measurement:** L2 distance — smaller distance = closer in meaning-space

## How the Distance Is Measured

Every embedding is a vector of 768 numbers — a point in 768-dimensional space.
The L2 (Euclidean) distance between two embeddings is the straight-line distance
between those two points. If a vector had only 2 numbers it would be a dot on a
map, and the distance is √((x₂−x₁)² + (y₂−y₁)²). With 768 numbers it's the same
idea: take the difference per coordinate, square each, sum them all, square-root.

Think of a patient's vitals as a vector [HR, BP, SpO₂, temp]. Two patients with
identical vitals → distance 0. A patient with fever + low oxygen sits far from a
healthy one. "Distance" = how different the whole picture is.

So a small distance (≈10–18) means the two "bat" embeddings are nearly identical
in meaning; a large one (≈33) means the model placed them far apart in
meaning-space.

## How to Run

```bash
# From this directory
pip install -r requirements.txt
python embeddings.py
```

## Expected Output

```
Loading model: gpt2 ...
  Ready! Embedding dimension: 768

=================================================================
  Static embedding of 'bat' (dictionary lookup)
=================================================================
  A vector of 768 numbers, e.g. first 5: [-0.019, -0.139, 0.252, 0.16, 0.08]
  This vector is IDENTICAL in every sentence.

=================================================================
  Contextual embeddings — the proof
=================================================================
  L2 distance matrix of 'bat' embeddings (smaller = closer):

              A1      A2      A3      S1      S2      S3
        -------------------------------------------------
  A1         0.0    10.4     9.7    29.0    27.6    36.7
  A2        10.4     0.0    13.8    27.3    25.9    34.0
  A3         9.7    13.8     0.0    36.5    34.9    44.3
  S1        29.0    27.3    36.5     0.0    18.1    11.1
  S2        27.6    25.9    34.9    18.1     0.0    22.5
  S3        36.7    34.0    44.3    11.1    22.5     0.0

  KEY NUMBERS:
    within-meaning (animal vs animal):    11.3
    within-meaning (sports vs sports):    17.3
    between-meaning (animal vs sports):   32.9
    ratio (between / within):            2.30x

Heatmap saved to: embeddings_distance_heatmap.png
```

![Distance heatmap](embeddings_distance_heatmap.png)

## What I Learned

1. **Static ≠ contextual.** The dictionary embedding of "bat" is one fixed vector — it literally can't tell a flying animal from a baseball tool. The model must build meaning from context.

2. **Context matters, but only from the left.** GPT-2 uses causal attention: at the moment it processes "bat", it can only use tokens *before* it. I discovered this through a real debugging moment — putting "bat" at position 1 made two opposite-meaning sentences produce identical embeddings (distance 0.0).

3. **Distance is a meaning meter.** Same-meaning "bat" sentences cluster close together (L2 ≈ 10–18); different meanings are roughly 2.3× farther apart (L2 ≈ 33). The heatmap shows two clean blocks.

4. **This is why RAG and search work.** When models turn words into meaning-space coordinates, "doctor" and "physician" sit close together — that's how semantic search finds relevant text, even without matching keywords.

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Matplotlib + NumPy (heatmap)
- Model: GPT-2 (124M parameters)
