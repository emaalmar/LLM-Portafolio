# Project 6: Prompt Engineering Principles — From Vague Text to Clinical Accuracy

**What it does:** Turns a messy nurse shift note into a structured SBAR handover report, and scores three increasingly engineered prompts against an objective clinical rubric — showing a jump from 50% to 92% accuracy.

**Why it matters:** An LLM doesn't know what you *mean* — it only follows what you *write*. In healthcare, a vague prompt can drop a critical detail like "Dr. Schmidt notified" or "follow-up ABG". Prompt engineering turns "good luck" into a reproducible, verifiable system.

## Concepts Demonstrated

- **Principle 1 — Give good instructions:** role, task, and constraints tell the model how to think
- **Principle 2 — Specify the output format:** forcing SBAR (Situation, Background, Assessment, Recommendation) shapes the answer
- **Principle 3 — Give examples (few-shot):** a worked example shows the expected style and completeness
- **Principle 4 — Create an evaluation method:** a programmatic rubric scores the output like a clinical checklist
- **Principle 5 — Divide the task:** numbered steps guide the model through every part of the report
- **Deterministic generation:** `do_sample=False` makes the prompt comparison fair and reproducible

## How to Run

```bash
pip install -r requirements.txt
python prompt_principles.py
```

## Expected Output

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

## What I Learned

1. **The naive prompt fails silently.** Demo 1 reads fluently and even mentions the ABG — yet it scored 50%: no SBAR structure, no "Dr. Schmidt notified". A handover that drops the physician notification is a clinical liability, even if it *sounds* good.

2. **Instructions + format (Principles 1 & 2) doubled the score to 83%.** Just assigning a role and forcing the SBAR headers made the model organize every fact — but it still missed the doctor notification, because nothing in the prompt told it to look for it.

3. **Few-shot + task division (Principles 3 & 5) reached 92%.** The worked example and the numbered steps acted like a mental checklist: *Step 4: extract ... doctor notifications* is exactly why "Dr. Schmidt notified" finally appeared.

4. **Evaluation is the most underrated principle.** In healthcare you can't judge an AI output by vibes. Principle 4 converts a handover into objective scores — the same mindset as a medication double-check or a WHO surgical checklist.

5. **Writing vs engineering.** Writing a prompt is typing text without a plan; engineering a prompt means defining a goal, choosing a framework (SBAR), structuring the reasoning, and measuring the result. One is luck; the other is a system.

## Tech Stack

- Python 3.14
- PyTorch 2.13 (CPU)
- HuggingFace Transformers 5.14
- Model: Qwen2.5-0.5B-Instruct (494M parameters, local)
