# Projekt 4: Base vs Instruct — Warum Fine-Tuning entscheidend ist

**Was es macht:** Vergleicht ein Base-Modell (GPT-2) mit einem Instruct-Modell (Qwen2.5-0.5B-Instruct) anhand identischer Prompts und zeigt, wie Fine-Tuning aus Text-Vorhersage KI wird.

**Warum es wichtig ist:** Fine-Tuning (Stufe 4) ist der Moment, in dem ein Sprachmodell zum Chatbot wird. Dieses Demo zeigt genau diesen Unterschied.

## Konzepte

- **Base-Modelle (Stufe 3):** Auf rotem Text trainiert — sie setzen Muster fort, beantworten keine Fragen
- **Instruct-Modelle (Stufe 4):** Mit SFT + RLHF/DPO feinjustiert — sie befolgen Anweisungen
- **Tokenisierung (Stufe 2):** BPE zerlegt Text in Tokens; verschiedene Modelle nutzen verschiedene Tokenizer
- **Die 5 Stufen eines LLM:** Der vollständige Weg von Rohdaten zu einem funktionierenden KI-Assistenten

## Ausführung

```bash
pip install -r requirements.txt
python base_vs_instruct.py
```

## Erwartete Ausgabe

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

If you're going to be running, there are some things you should do...

  Qwen2.5 (instruct):
  Exercise has numerous beneficial effects on both physical and mental health,
  promoting overall well-being and improving quality of life.
```

## Was ich gelernt habe

1. **Base-Modelle ergänzen, Instruct-Modelle antworten.** GPT-2 wurde darauf trainiert, das nächste Token im Webtext vorherzusagen. Es weiß nicht, dass es deine Frage beantworten soll — es setzt nur das Muster fort. Qwen wurde feinjustiert, um Anweisungen zu befolgen.

2. **Fine-Tuning ist die Brücke.** Stufe 3 (Pre-Training) liefert ein Sprachmodell. Stufe 4 (Fine-Tuning mit SFT + RLHF) liefert einen Assistenten. Der Unterschied in der Ausgabe IST, was Fine-Tuning macht.

3. **GPT-2 ist nicht „dumm"** — es ist nur nicht ausgerichtet. Es hat während des Pre-Trainings Sprachstruktur, Grammatik und Fakten gelernt. Aber ohne Fine-Tuning weiß es nicht, wie es dieses Wissen in einem Gespräch nutzen soll.

4. **Die 5 Stufen eines LLM:**
   - Stufe 1: Daten beschaffen (Common Crawl, Bücher, Code)
   - Stufe 2: Tokenisieren (BPE)
   - Stufe 3: Pre-Training (Next-Token-Prediction) → Base-Modell
   - Stufe 4: Fine-Tuning (SFT + RLHF/DPO) → Instruct-Modell
   - Stufe 5: Evaluieren (Benchmarks, menschliche Bewertung)

## Tech-Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- tiktoken 0.13 (OpenAIs Tokenizer)
- Modelle: GPT-2 (124M) + Qwen2.5-0.5B-Instruct (500M)
