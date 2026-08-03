# Projekt 1: LLM Next-Token Prediction

**Was es macht:** Zeigt, wie ein LLM das nächste Token vorhersagt.

**Warum es wichtig ist:** Das ist die grundlegende Operation jedes LLM (GPT, Claude, Gemini). Wer sie versteht, versteht die Basis aller modernen KI-Anwendungen.

## Konzepte

- **Tokenisierung:** Wie Text in Stücke zerlegt wird, die das Modell verarbeiten kann
- **Wahrscheinlichkeitsverteilung:** Das Modell ordnet jedem Wort in seinem Vokabular eine Wahrscheinlichkeit zu
- **Top-K-Kandidaten:** Wir zeigen die 5 wahrscheinlichsten nächsten Tokens

## Ausführung

```bash
pip install -r requirements.txt
python next_token.py
```

## Erwartete Ausgabe

```
Input:          'The cat sleeps on the'
Token IDs:      [464, 3797, 44263, 319, 262]
Tokens:         ['The', 'Ġcat', 'Ġsleeps', 'Ġon', 'Ġthe']

Top 5 predicted next tokens:
  1. ' floor' — 20.3%
  2. ' ground' — 7.9%
  3. ' bed' — 5.4%
  4. ' couch' — 5.2%
  5. ' side' — 3.5%

Best prediction: 'The cat sleeps on the  floor'
```

## Was ich gelernt habe

1. **Tokens sind keine Wörter** — der Tokenizer zerlegt "sleeps" in ein Token, aber "cat" wird zu "Ġcat" (mit Leerzeichen-Präfix). Das Modell arbeitet mit Token-IDs, nicht mit Text.

2. **Vorhersagen sind probabilistisch** — das Modell "weiß" nicht die Antwort, sondern ordnet jeder Möglichkeit eine Wahrscheinlichkeit zu. "Floor" gewinnt mit 20,3%, aber "ground" und "bed" sind ebenfalls plausibel.

3. **Self-Attention steckt dahinter** — um das nächste Token vorherzusagen, nutzt das Modell Self-Attention (Q/K/V), um herauszufinden, welche vorherigen Tokens am wichtigsten sind. "Cat" und "sleeps" sind die Signale.

## Tech-Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Modell: GPT-2 (124 Mio. Parameter)
