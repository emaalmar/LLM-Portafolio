# Project 2: Autoregressive Text Generation

**What it does:** Shows how an LLM builds text word by word, feeding its own output back in.

**Why it matters:** This is the exact same loop that powers ChatGPT, Claude, and Gemini. Understanding it demystifies how LLMs "think."

## Concepts Demonstrated

- **Autoregressive loop:** predict 1 token → append → predict again → repeat
- **Temperature:** controls randomness (lower = more deterministic)
- **Sampling:** we sample from the probability distribution (not greedy — adds variety)

## How to Run

```bash
pip install -r requirements.txt
python autoregressive.py
```

## Expected Output

```
Prompt: 'The future of healthcare is'

============================================================
  AUTOREGRESSIVE LOOP — watch text build word by word
============================================================
  Step  1:     at  ->  The future of healthcare is at
  Step  2:  stake  ->  The future of healthcare is at stake
  Step  3:     in  ->  The future of healthcare is at stake in
  Step  4:  Europe  ->  The future of healthcare is at stake in Europe
  ...
  Step 30:   just  ->  The future of healthcare is at stake in Europe and the US...
```

## What I Learned

1. **The loop is simple:** predict one token, feed it back, repeat. The complexity is in the model, not the loop.

2. **Temperature matters:** at 0.8, the model is somewhat creative. At 0.1, it would always pick the most probable word (boring). At 1.5, it would be chaotic.

3. **Context grows with each step:** at step 1, the model only sees 7 tokens. By step 30, it sees 37 tokens. Each prediction is based on ALL of them.

4. **This is how ChatGPT works** — just with a billion-parameter model and better sampling strategies.

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Model: GPT-2 (124M parameters)

## Try It Yourself

Edit the `prompt` variable in `autoregressive.py` and try different inputs:

```python
prompt = "As a nurse, I believe AI will"
prompt = "In Deutschland fehlen"
prompt = "The best way to learn Python is"
```
