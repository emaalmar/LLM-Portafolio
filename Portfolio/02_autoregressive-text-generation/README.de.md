# Projekt 2: Autoregressive Textgenerierung

**Was es macht:** Zeigt, wie ein LLM Text Wort für Wort aufbaut und die eigene Ausgabe als Eingabe zurückführt.

**Warum es wichtig ist:** Das ist exakt die Schleife, die ChatGPT, Claude und Gemini antreibt.

## Konzepte

- **Autoregressive Schleife:** 1 Token vorhersagen -> anhängen -> erneut vorhersagen -> wiederholen
- **Temperatur:** steuert die Zufälligkeit (niedriger = deterministischer)
- **Sampling:** wir ziehen aus der Wahrscheinlichkeitsverteilung (nicht "greedy" — das macht es vielfältiger)

## Ausführung

```bash
pip install -r requirements.txt
python autoregressive.py
```

## Erwartete Ausgabe

```
Prompt: 'The future of healthcare is'

  Step  1:     at  ->  The future of healthcare is at
  Step  2:  stake  ->  The future of healthcare is at stake
  Step  3:     in  ->  The future of healthcare is at stake in
  Step  4:  Europe  ->  The future of healthcare is at stake in Europe
  ...
  Step 30:   just  ->  The future of healthcare is at stake in Europe and the US...
```

## Was ich gelernt habe

1. **Die Schleife ist simpel:** ein Token vorhersagen, zurückführen, wiederholen. Die Komplexität liegt im Modell, nicht in der Schleife.

2. **Temperatur ist entscheidend:** bei 0.8 ist das Modell etwas kreativ. Bei 0.1 würde es immer das wahrscheinlichste Wort wählen. Bei 1.5 wäre es chaotisch.

3. **Kontext wächst mit jedem Schritt:** bei Schritt 1 sieht das Modell 7 Tokens. Bei Schritt 30 sind es 37. Jede Vorhersage basiert auf ALLEN bisherigen.

## Tech-Stack

- Python 3.14, PyTorch 2.13 (CPU), HuggingFace Transformers 5.14
- Modell: GPT-2 (124 Mio. Parameter)

## Selbst ausprobieren

Ändere die Variable `prompt` in `autoregressive.py`:
```python
prompt = "As a nurse, I believe AI will"
prompt = "In Deutschland fehlen"
prompt = "The best way to learn Python is"
```
