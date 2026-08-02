# Project 4: Base vs Instruct — Why Fine-Tuning Matters

**What it does:** Compares a base model (GPT-2) with an instruct model (Qwen2.5-0.5B-Instruct) on the same prompts, showing how fine-tuning transforms text prediction into conversational AI.

**Why it matters:** Fine-tuning (Stage 4) is what turns a language model into a chatbot. This demo shows the exact moment the model goes from "autocomplete" to "conversation partner."

## Concepts Demonstrated

- **Base models (Stage 3):** Trained on raw text — they continue patterns, don't answer questions
- **Instruct models (Stage 4):** Fine-tuned with SFT + RLHF/DPO — they follow instructions and respond to prompts
- **Tokenization (Stage 2):** BPE splits text into tokens; different models use different tokenizers
- **The 5 stages of an LLM:** The complete journey from raw data to a usable AI assistant

## How to Run

```bash
# From this directory
pip install -r requirements.txt
python base_vs_instruct.py
```

## Expected Output

```
Loading models...

  PROMPT 1: 'What is the capital of Japan?'

  GPT-2 (base):
  What is the capital of Japan?

It is based on the Tohoku-shi-ken area of Japan, where Kanno Shinkai...

  Qwen2.5 (instruct):
  The capital of Japan is Tokyo.


  PROMPT 2: 'Summarize the benefits of exercise in one sentence.'

  GPT-2 (base):
  Summarize the benefits of exercise in one sentence.

If you're going to be running, there are some things you should do before going...

  Qwen2.5 (instruct):
  Exercise has numerous beneficial effects on both physical and mental health,
  promoting overall well-being and improving quality of life.
```

## What I Learned

1. **Base models autocomplete, instruct models answer.** GPT-2 was trained to predict the next token in web text. It doesn't know it should answer your question — it just continues the pattern. Qwen was fine-tuned to follow instructions.

2. **Fine-tuning is the bridge.** Stage 3 (pre-training) gives you a language model. Stage 4 (fine-tuning with SFT + RLHF) gives you an assistant. The difference you see in the output IS what fine-tuning does.

3. **GPT-2 isn't "stupid"** — it's just not aligned. It learned language structure, grammar, and facts during pre-training. But without fine-tuning, it doesn't know HOW to use that knowledge in a conversation.

4. **The 5 stages of an LLM:**
   - Stage 1: Get data (Common Crawl, books, code)
   - Stage 2: Tokenize (BPE)
   - Stage 3: Pre-train (next-token prediction) → base model
   - Stage 4: Fine-tune (SFT + RLHF/DPO) → instruct model
   - Stage 5: Evaluate (benchmarks, human eval)

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- tiktoken 0.13 (OpenAI's tokenizer)
- Models: GPT-2 (124M) + Qwen2.5-0.5B-Instruct (500M)
