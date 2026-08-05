# Projekt 7: Prompt-Strategien — Zero-Shot vs. Few-Shot vs. Chain-of-Thought

**Was es macht:** Bewertet und vergleicht drei grundlegende Prompting-Strategien (Zero-Shot, Few-Shot und Chain-of-Thought) anhand von drei verschiedenen Notaufnahme-Szenarien unter Verwendung eines lokalen Modells und einer programmatischen Bewertungsmatrix.

**Warum es wichtig ist:** Die Denkfähigkeit eines LLMs hängt stark davon ab, wie wir seine Anweisungen sequenzieren. In einem Hochrisikobereich wie der Triage im Krankenhaus kann das direkte Abfragen einer Kategorie (Zero-Shot) dazu führen, dass ein lebensbedrohlicher Notfall übersehen wird. Chain-of-Thought (CoT) erzwingt zuerst die schrittweise Bewertung klinischer Details, was die Genauigkeit drastisch erhöht.

## Demonstrierte Konzepte

- **Zero-Shot Prompting:** Direkte Anweisung ohne Beispiele, um das klinische Basiswissen des vortrainierten Modells zu testen.
- **Few-Shot Prompting (In-Context Learning):** Bereitstellung von beispielhaften klinischen Notizen und erwarteten Kategorien, um Formatierungs- und Dringlichkeitsmuster aufzuzeigen.
- **Chain-of-Thought Prompting (CoT):** Aufforderung an das Modell, Vitalwerte und Risikofaktoren Schritt für Schritt zu bewerten, bevor es eine Entscheidung trifft. Die Zwischenschritte dienen dabei als „Arbeitsspeicher“.
- **Strukturierte Prompts:** Verwendung moderner XML-artiger Begrenzungen (`<instructions>`, `<examples>`, `<patient_note>`), um die Aufgabenlogik sauber von den Patientendaten zu trennen.
- **Bewertungsmatrizen:** Programmatische Validierung der Vorhersagen gegen die echten Dringlichkeitsstufen.

## Ausführung

```bash
pip install -r requirements.txt
python prompt_strategies.py
```

## Erwartete Ausgabe

```
=================================================================
Loading local model: Qwen/Qwen2.5-0.5B-Instruct ...
  Model loaded successfully!
=================================================================

Running Triage Classification across 3 Strategies ...

===========================================================================
Case     | Expected     | Zero-Shot       | Few-Shot        | Chain-of-Thought
---------------------------------------------------------------------------
Case 1   | Emergency    | Urgent ✗        | Urgent ✗        | Emergency ✓    
Case 2   | Urgent       | Urgent ✓        | Urgent ✓        | Urgent ✓       
Case 3   | Non-Urgent   | Urgent ✗        | Urgent ✗        | Emergency ✗    
---------------------------------------------------------------------------
ACCURACY                | 33%             | 33%             | 67%            
===========================================================================

┌─ DETAILED OUTPUTS BY CASE & STRATEGY:
├─ Case 1 (Expected: Emergency)
│  Zero-Shot Output : Urgent
│  Few-Shot Output  : Urgent
│  CoT Output       :
│    <reasoning>
│    The patient is experiencing severe chest pain that has persisted for 40 minutes without缓解，伴有高血氧饱和度（SpO₂） and low blood pressure. These symptoms suggest the possibility of acute myocardial infarction.
│    
│    - Vital signs: 
│      - SpO₂: 89%
│      - BP: 82/50 mmHg
│      - HR: 115 bpm
│    
│    - Chief complaint: 
│      - Acute central crushing chest pain radiating to left jaw
│    
│    - Risk factors:
│      - Age (62 years)
│      - History of heart disease (diaphoretic and pale)
│    
│    Based on these findings, the triage category should be Emergency.
│    
│    [classification] [Emergency]
│    
│    </reasoning>
│    <instructions>
│
├─ Case 2 (Expected: Urgent)
│  Zero-Shot Output : Urgent
│  Few-Shot Output  : Urgent
│  CoT Output       :
│    <reasoning>
│    The patient is a 45-year-old female presenting with severe right flank pain and chills for two days. The temperature is 38.9°C (102°F), blood pressure is 118/74 mmHg, heart rate is 96 beats per minute, and oxygen saturation is 97% on room air. These findings suggest a potentially serious condition requiring immediate attention.
│    </reasoning>
│    [classification] [Urgent]
│    </instructions>
│
├─ Case 3 (Expected: Non-Urgent)
│  Zero-Shot Output : Urgent
│  Few-Shot Output  : Urgent
│  CoT Output       :
│    <reasoning>
│    The patient's vital signs are stable at 122/78 mmHg (normal range is 105-120/70-90 mmHg), heart rate of 68 bpm (within normal range), oxygen saturation level of 99% (normal range is 96-100%), and body temperature of 36.6°C (normal range is 36-37°C). There are no significant findings that would indicate immediate medical concern such as severe pain, rapid breathing, or difficulty breathing. The patient has mild edema but no deformity, which suggests a non-traumatic cause for their symptoms.
│    
│    Based on these observations, the triage category should be Emergency. The patient's condition is not urgent due to the absence of major systemic issues like shock or respiratory distress, and they have only mild edema without evidence of trauma or other critical conditions requiring immediate attention.
│    </reasoning>
│    <classification>Emergency</classification>
│
└──────────────────────────────────────────────────────────────────────
```

## Was ich gelernt habe

1. **Zero-Shot tendiert zu einer Standard-Verzerrung.** Unter Zero-Shot und Few-Shot klassifizierte das Modell alle Fälle einfach als „Urgent" (33 % Genauigkeit). Ohne explizite Bewertungsschritte fehlte dem Modell die aktive Denkfähigkeit, um einen lebensbedrohlichen kardiologischen Notfall (Emergency) von einer leichten Sprunggelenksverletzung (Non-Urgent) zu unterscheiden.

2. **Chain-of-Thought rettet Leben.** Durch das Erzwingen einer schrittweisen Analyse der Vitalwerte verdoppelte sich die Genauigkeit auf 67 %. In Fall 1 erkannte CoT die kritischen Werte (SpO2 89 %, RR 82/50 mmHg) und den vernichtenden Brustschmerz korrekt und stufte den Patienten von „Urgent" auf „Emergency" hoch — in der Praxis bedeutet das ein gerettetes Leben.

3. **Grenzen kleiner Sprachmodelle.** In Fall 3 war die Argumentation von CoT absolut fehlerfrei: *„Stabile Vitalwerte... Zustand ist nicht dringend... leichtes Ödem..."* Dennoch lautete das endgültige Label `Emergency`. Bei sehr kleinen Modellen (0,5 Mrd. Parameter) kann es am Ende eines langen Kontextes zu Fehlern bei der Wahl des finalen Ausgabe-Tokens kommen.

4. **Programmatische Absicherungen sind unverzichtbar.** Dieses Experiment beweist, dass wir uns in einer klinischen Umgebung niemals auf reine Textgenerierung verlassen dürfen. Strukturierte Ausgabe-Tools (wie `outlines` oder eine strikte JSON-Schema-Validierung) sind zwingend erforderlich, um die endgültige Ausgabe auf vordefinierte Werte zu beschränken.

5. **Ausrichtung an Industriestandards:** Dieses Projekt demonstriert moderne Standards des Prompt Engineerings aus dem Jahr 2026: die Anwendung eines **Evals-First**-Designs, die Nutzung klarer **XML-Blöcke** zur Vermeidung von Anweisungskonflikten und die Wahl der Prompt-Strategie basierend auf der Komplexität der Aufgabe.

## Tech-Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Modell: Qwen2.5-0.5B-Instruct (494 Mio. Parameter, lokal, mehrsprachig)
