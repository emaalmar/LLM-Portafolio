# Project 1: LLM Next-Token Prediction

**What it does:** Shows how an LLM predicts the next token given a sequence of text.

**Why it matters:** This is the fundamental operation behind every LLM (GPT, Claude, Gemini). Understanding it is the first step to building AI applications.

## Concepts Demonstrated

- **Tokenization:** How text is split into pieces (tokens) the model can process
- **Probability distribution:** The model assigns a probability to every word in its vocabulary
- **Top-K candidates:** We show the top 5 most likely next tokens

## How to Run

```bash
# From this directory
pip install -r requirements.txt
python next_token.py
```

## Expected Output

```
Loading model: gpt2 ...
  Model loaded! (124M parameters)

Input:          'The cat sleeps on the'
Token IDs:      [464, 3797, 44263, 319, 262]
Tokens:         ['The', 'Ġcat', 'Ġsleeps', 'Ġon', 'Ġthe']

Top 5 predicted next tokens:
----------------------------------------
  1. ' floor' — 20.3%  ████████
  2. ' ground' — 7.9%  ███
  3. ' bed' — 5.4%  ██
  4. ' couch' — 5.2%  ██
  5. ' side' — 3.5%  █

Best prediction: 'The cat sleeps on the  floor'
```

## What I Learned

1. **Tokens are not words** — the tokenizer splits "sleeps" into a single token, but "cat" becomes "Ġcat" (with a space prefix). The model sees token IDs, not text.

2. **Predictions are probabilistic** — the model doesn't "know" the answer; it assigns probabilities to every possible next token. "Floor" wins at 20.3%, but "ground" and "bed" are also plausible.

3. **Self-attention powers this** — to predict the next token, the model uses self-attention (Q/K/V) to figure out which previous tokens matter most. "Cat" and "sleeps" are the key signals.

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Model: GPT-2 (124M parameters)


