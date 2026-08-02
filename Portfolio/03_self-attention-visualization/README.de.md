# Projekt 3: Self-Attention-Visualisierung

**Was es macht:** Zeigt, wie ein Transformer-Modell mithilfe von Attention-Gewichten (Q, K, V) entscheidet, welche Wörter wichtig sind.

**Warum es wichtig ist:** Self-Attention ist der Mechanismus, der Transformer intelligent macht — das "Geheimrezept" jedes LLM.

## Konzepte

- **Q (Queries):** "Wonach suche ich?"
- **K (Keys):** "Was enthalte ich?"
- **V (Values):** "Was gebe ich weiter?"
- **Attention-Gewichte:** wie stark jedes Token jedes andere "wahrnimmt"

## Ausführung

```bash
pip install -r requirements.txt
python attention.py
```

## Die Heatmap lesen

- **Y-Achse (Zeilen):** "Wer schaut?" — jedes Token stellt eine Frage
- **X-Achse (Spalten):** "Wen schaut man an?" — jedes Token wird bewertet
- **Dunkelblau:** starke Attention ("dieses Token ist wichtig!")
- **Hell:** schwache Attention ("ignoriere das")

## Was ich gelernt habe

1. **Attention = Kontext verstehen:** Das Modell erkennt, welche Wörter an welcher Stelle wichtig sind, anstatt alle gleich zu behandeln.

2. **"The" dominiert:** Das erste Token bekommt die höchste Attention, weil es den Satz strukturiert.

3. **"cat" und "sleeps" transportieren Bedeutung:** Das Modell lernt, dass WER (cat) und WAS (sleeps) die wichtigsten Signale sind.

4. **Q/K/V in Aktion:** Das Modell lernt, welche Tokens wichtig sind (Q/K Attention-Scores) und nutzt das zur Informationskombination (V).

## Tech-Stack

- Python 3.14, PyTorch 2.13 (CPU), HuggingFace Transformers 5.14
- Matplotlib (Heatmap)
- Modell: GPT-2 (124 Mio. Parameter)
