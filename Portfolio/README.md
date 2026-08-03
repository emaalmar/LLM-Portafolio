# Ema's AI Engineering Portfolio

**Nurse -> AI Engineer** | Healthcare + Machine Learning

## About Me

I'm a nurse transitioning into AI Engineering, combining clinical healthcare knowledge with modern machine learning skills. Currently based in Germany, completing an Ausbildung in nursing while building AI/LLM projects.

**Why healthcare + AI?** Because the best AI in healthcare is built by people who understand both the technology AND the clinical workflow. I've seen firsthand how data, automation, and intelligent systems can improve patient outcomes — and I want to build those systems.

## Skills

- **Programming:** Python, JavaScript, React
- **AI/ML:** Transformers, LLMs, Tokenization (BPE), Prompt Engineering, Fine-tuning (SFT, RLHF, DPO), RAG
- **Tools:** PyTorch, HuggingFace Transformers, LangChain, Promptfoo
- **Domain:** Healthcare data, clinical workflows, HealthTech

## Projects

| # | Project | Concepts | Status |
|---|---------|----------|--------|
| 1 | [LLM Next-Token Prediction](01_llm-next-token/) | Transformers, tokenization, probability | Done |
| 2 | [Autoregressive Text Generation](02_autoregressive-text-generation/) | Autoregressive loop, sampling, temperature | Done |
| 3 | [Self-Attention Visualization](03_self-attention-visualization/) | Q/K/V, attention weights, heatmap | Done |
| 4 | [Base vs Instruct: Fine-Tuning](04_base-vs-instruct-finetuning/) | Tokenization, BPE, base vs instruct, SFT/RLHF | Done |
| 5 | [Embeddings & Polysemy](05_embeddings-polysemy/) | Static vs contextual embeddings, meaning-space | Done |
| 6 | RAG for Medical Documents | Retrieval-Augmented Generation, embeddings | Planned |
| 7 | HealthTech Chatbot | LLM agents, LangGraph, tool use | Planned |

## How to Run

Each project has its own `README.md` with instructions. All projects run locally (no API keys needed for basic demos).

```bash
# Clone this repo
git clone https://github.com/emaalmar/LLM-Portafolio.git

# Go to any project
cd 01_llm-next-token

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python next_token.py
```

## What I Know

- **Transformers:** architecture that powers all modern LLMs
- **Self-attention (Q/K/V):** how models figure out which words matter
- **Autoregressive generation:** how text is built token by token
- **Tokenization (BPE):** how text becomes numbers the model can process
- **Embeddings (static vs contextual):** how meaning depends on context
- **Base vs Instruct models:** the difference between pre-training and fine-tuning
- **Fine-tuning (SFT, RLHF/DPO):** how we align models to follow instructions
- **Prompt engineering:** zero-shot, few-shot, chain-of-thought

## Contact

- GitHub: [https://github.com/emaalmar](https://github.com/emaalmar)
