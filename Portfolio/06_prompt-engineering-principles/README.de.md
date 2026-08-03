# Projekt 6: Prinzipien des Prompt Engineerings — Von vagen Texten zu klinischer Genauigkeit

**Was es macht:** Verwandelt eine unübersichtliche Pflegeübergabe in einen strukturierten SBAR-Bericht und bewertet drei zunehmend durchdachte Prompts anhand einer objektiven klinischen Bewertungsmatrix — mit einer Verbesserung von 50 % auf 92 %.

**Warum es wichtig ist:** Ein LLM weiß nicht, was du *meinst* — es folgt nur dem, was du *schreibst*. Im Gesundheitswesen kann ein vager Prompt ein kritisches Detail verlieren, etwa „Dr. Schmidt informiert" oder „ABG-Verlaufskontrolle". Prompt Engineering macht aus „viel Glück" ein reproduzierbares, überprüfbares System.

## Demonstrierte Konzepte

- **Prinzip 1 — Gute Anweisungen geben:** Rolle, Aufgabe und Einschränkungen sagen dem Modell, wie es denken soll
- **Prinzip 2 — Das Ausgabeformat festlegen:** SBAR erzwingen (Situation, Background, Assessment, Recommendation) formt die Antwort
- **Prinzip 3 — Beispiele geben (Few-Shot):** ein durchgearbeitetes Beispiel zeigt erwarteten Stil und Vollständigkeit
- **Prinzip 4 — Eine Bewertungsmethode schaffen:** eine programmatische Matrix bewertet die Ausgabe wie eine klinische Checkliste
- **Prinzip 5 — Die Aufgabe aufteilen:** nummerierte Schritte führen das Modell durch jeden Teil des Berichts
- **Deterministische Generierung:** `do_sample=False` macht den Prompt-Vergleich fair und reproduzierbar

## Ausführung

```bash
pip install -r requirements.txt
python prompt_principles.py
```

## Erwartete Ausgabe

```
=================================================================
Loading model: Qwen/Qwen2.5-0.5B-Instruct ...
  Model loaded successfully!
=================================================================

┌─ RAW NURSE SHIFT NOTE:
│  pt Miller, Room 304, 68yo M. Admitted yesterday with acute dyspnea and COPD exacerbation.
SpO2 was 88% on room air at 06:00, started on 2L O2 via nasal cannula, now 94%.
BP 138/82, HR 88, Temp 37.1C. Patient reports feeling less short of breath but still fatigued.
Nebulizer treatment (albuterol) given at 08:00 with good response.
Needs follow-up ABG at 12:00 and chest physio scheduled for 14:00. Dr. Schmidt notified of status.
└────────────────────────────────────────────────────────────

┌─ DEMO 1: Naive Prompt (No structure, unguided)
│  The patient is admitted to the hospital with acute dyspnea and COPD exacerbation. The oxygen saturation (SpO2) is currently at 88%, with an initial oxygen flow rate of 2 liters per minute administered via nasal cannula. The patient's blood pressure is 138/82 mmHg, heart rate is 88 beats per minute, and body temperature is 37.1°C. The patient reports feeling less short of breath but still feels fatigued.
│  
│  The patient has been prescribed Nebulizer therapy with albuterol at 08:00, which has provided some improvement in symptoms. A follow-up ABG (arterial blood gas) will be conducted at 12:00, and a chest physiotherapy appointment will be scheduled for 14:00. The doctor has informed the patient about their current condition.
└────────────────────────────────────────────────────────────

┌─ DEMO 2: Principles 1 & 2 (Instructions + Output Format)
│  Situation: A patient with acute dyspnea and COPD exacerbation is admitted to the hospital.
│  Background: The patient has been experiencing severe respiratory distress due to COPD exacerbation.
│  Assessment: The patient's oxygen saturation (SpO2) is currently 88%, indicating hypoxemia. They have been prescribed supplemental oxygen via nasal cannula and are receiving albuterol inhalation therapy. Their blood pressure is stable at 138/82 mmHg, heart rate is 88 beats per minute, and temperature is 37.1°C.
│  Recommendation: Follow up an arterial blood gas (ABG) test at 12:00 to assess the current state of their respiratory acid-base balance. Schedule a chest physiotherapy session at 14:00 to address any potential issues related to their breathing difficulties.
└────────────────────────────────────────────────────────────

┌─ DEMO 3: Principles 3 & 5 (Few-Shot Example + Task Division)
│  Situation: Patient Miller, 68yo male, Room 304.
│  Background: Admitted yesterday with acute dyspnea and COPD exacerbation.
│  Assessment: Current oxygen saturation (SpO2) is 88%, BP 138/82 mmHg, HR 88 bpm, Temp 37.1°C. The patient has been receiving 2L O2 via nasal cannula since the morning. No other interventions have been made yet.
│  Immediate Interventions: Nebulizer treatment (albuterol) initiated at 08:00 with good response.
│  Scheduled Tasks: Follow-up ABG at 12:00 and chest physio scheduled for 14:00. Dr. Schmidt notified of the status.
│  Recommendation: Immediate follow-up ABG at 12:00 and chest physio scheduled for 14:00. Dr. Schmidt notified of the status.
└────────────────────────────────────────────────────────────

=================================================================
  PRINCIPLE 4: Programmatic Evaluation Rubric
=================================================================

Metric                       | Demo 1 (Naive)     | Demo 2 (P1+2)      | Demo 3 (P1+2+3+5) 
----------------------------------------------------------------------------------------
SBAR Headers (max 4)         | 0/4                | 4/4                | 4/4               
Vitals Capture (max 5)       | 4/5                | 4/5                | 4/5               
Pending Tasks (max 2)        | 2/2                | 2/2                | 2/2               
Doctor Notified              | No                 | No                 | Yes               
Total Clinical Score         | 6/12 (50%)         | 10/12 (83%)        | 11/12 (92%)        

=================================================================
  TAKEAWAY: Applying the 5 principles transformed a vague text
  generator into a reliable, clinical-grade handover tool!
=================================================================
```

## Was ich gelernt habe

1. **Der naive Prompt versagt lautlos.** Demo 1 liest sich flüssig und erwähnt sogar die ABG — und erreicht doch nur 50 %: keine SBAR-Struktur, kein „Dr. Schmidt informiert". Eine Übergabe ohne die Arzt-Benachrichtigung ist ein klinisches Risiko, selbst wenn sie sich *gut anhört*.

2. **Anweisungen + Format (Prinzipien 1 & 2) verdoppelten den Score auf 83 %.** Allein die Rolle und die SBAR-Überschriften brachten das Modell dazu, jede Information zu strukturieren — aber die Arzt-Benachrichtigung fehlte weiterhin, weil nichts im Prompt danach suchte.

3. **Few-Shot + Aufgabenaufteilung (Prinzipien 3 & 5) erreichten 92 %.** Das Beispiel und die nummerierten Schritte wirkten wie eine gedankliche Checkliste: *Schritt 4: extrahiere ... Arzt-Benachrichtigungen* — genau deshalb tauchte „Dr. Schmidt notified" endlich auf.

4. **Die Bewertung ist das unterschätzte Prinzip.** Im Gesundheitswesen kann man eine KI-Ausgabe nicht nach Gefühl beurteilen. Prinzip 4 verwandelt eine Übergabe in objektive Werte — dieselbe Denkweise wie eine Medikamenten-Doppelkontrolle oder eine WHO-Operations-Checkliste.

5. **Schreiben vs. Engineering.** Einen Prompt zu schreiben heißt, Text ohne Plan zu tippen; einen Prompt zu *enginieeren* heißt, ein Ziel zu definieren, einen Rahmen (SBAR) zu wählen, das Denken zu strukturieren und das Ergebnis zu messen. Das eine ist Zufall, das andere ein System.

## Tech-Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Modell: Qwen2.5-0.5B-Instruct (494 Mio. Parameter, lokal)
