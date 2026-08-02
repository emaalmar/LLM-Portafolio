# Project 3: Self-Attention Visualization

**What it does:** Shows how a Transformer model decides which words matter using attention weights (Q, K, V).

**Why it matters:** Self-attention is the mechanism that makes Transformers smart — it's the "secret sauce" behind every LLM.

## Concepts Demonstrated

- **Q (Queries):** "What am I looking for?"
- **K (Keys):** "What do I contain?"
- **V (Values):** "What do I pass along?"
- **Attention weights:** how much each token should "attend to" every other token

## How to Run

```bash
pip install -r requirements.txt
python attention.py
```

## Expected Output

```
Input: 'The cat sleeps on the'
Tokens: ['The', 'cat', 'sleeps', 'on', 'the']

Attention weights (averaged across heads, last layer):
                     The        cat     sleeps         on        the
------------------------------------------------------------------------
         The       1.000       0.000       0.000       0.000       0.000
        cat        0.871       0.129       0.000       0.000       0.000
     sleeps        0.783       0.085       0.133       0.000       0.000
         on        0.719       0.082       0.116       0.083       0.000
        the        0.670       0.096       0.066       0.104       0.065

Heatmap saved to: attention_heatmap.png
```

## How to Read the Heatmap

- **Y-axis (rows):** "Who is looking" — each token asking a question
- **X-axis (columns):** "Who is being looked at" — each token being evaluated
- **Dark blue:** strong attention ("this token matters!")
- **Light:** weak attention ("ignore this")

## What I Learned

1. **Attention = understanding context:** the model figures out which words are important for each position, rather than treating all words equally.

2. **"The" dominates:** the first token gets the highest attention because it sets the structure of the sentence.

3. **"cat" and "sleeps" signal meaning:** the model learns that WHO (cat) and WHAT (sleeps) are the key signals to predict the next word.

4. **This is Q/K/V in action:** the model learns which tokens matter (Q/K attention scores) and uses that to combine information (V).

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Matplotlib (for heatmap)
- Model: GPT-2 (124M parameters)
