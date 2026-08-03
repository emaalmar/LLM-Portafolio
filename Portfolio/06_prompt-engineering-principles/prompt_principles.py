"""
Project 6: The 5 Principles of Prompt Engineering
==================================================
Task: Convert a messy nurse shift note into a structured SBAR handover report,
and measure how each prompt-engineering principle improves the result.

The Five Principles Demonstrated:
  1. Give good instructions (role, task, constraints)
  2. Specify the output format (SBAR: Situation, Background, Assessment, Recommendation)
  3. Give examples (few-shot template)
  4. Create an evaluation method (programmatic clinical rubric)
  5. Divide the task (step-by-step processing)

Model: Qwen2.5-0.5B-Instruct (local, CPU-friendly instruction model).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load the local instruction model ─────────────────────────
# WHY: An *instruct* model is fine-tuned to follow instructions, which is
# exactly what prompt engineering relies on.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 65)
print(f"Loading model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("  Model loaded successfully!")
print("=" * 65)
print()

# ── Step 2: Raw input data (a messy real-world nurse shift note) ─────
# WHY: Real clinical notes are unstructured. The patient data below contains
# everything a handover needs — but buried in free text.
NURSE_NOTE = """
pt Miller, Room 304, 68yo M. Admitted yesterday with acute dyspnea and COPD exacerbation.
SpO2 was 88% on room air at 06:00, started on 2L O2 via nasal cannula, now 94%.
BP 138/82, HR 88, Temp 37.1C. Patient reports feeling less short of breath but still fatigued.
Nebulizer treatment (albuterol) given at 08:00 with good response.
Needs follow-up ABG at 12:00 and chest physio scheduled for 14:00. Dr. Schmidt notified of status.
"""


def generate_response(prompt_text, max_new_tokens=300):
    """Run one prompt through the model and return only the generated text.
    WHY: do_sample=False keeps the output deterministic, so the comparison
    between prompts is fair and reproducible."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("┌─ RAW NURSE SHIFT NOTE:")
print(f"│  {NURSE_NOTE.strip()}")
print("└" + "─" * 60)
print()

# ── DEMO 1: The naive prompt (no engineering at all) ─────────────────
# WHY: A baseline. Without structure the model has to guess what "summarize"
# means, so it may drop critical clinical details.
naive_prompt = f"Summarize this nurse note:\n{NURSE_NOTE}"
print("┌─ DEMO 1: Naive Prompt (No structure, unguided)")
naive_output = generate_response(naive_prompt)
for line in naive_output.splitlines():
    print(f"│  {line}")
print("└" + "─" * 60)
print()

# ── DEMO 2: Principles 1 & 2 (clear instructions + output format) ────
# WHY: A role ("clinical nurse leader") tells the model HOW to think, and
# forcing the SBAR format tells it WHAT shape the answer must take.
structured_prompt = f"""You are a clinical nurse leader. Summarize the following nurse note using standard SBAR format.

Format:
Situation: ...
Background: ...
Assessment: ...
Recommendation: ...

Nurse Note:
{NURSE_NOTE}
"""
print("┌─ DEMO 2: Principles 1 & 2 (Instructions + Output Format)")
structured_output = generate_response(structured_prompt)
for line in structured_output.splitlines():
    print(f"│  {line}")
print("└" + "─" * 60)
print()

# ── DEMO 3: Principles 3 & 5 (example + task division) ───────────────
# WHY: A worked example (few-shot) shows the expected style, and splitting the
# job into numbered steps guides the model token-by-token through every
# part of the SBAR report — nothing gets skipped.
few_shot_prompt = f"""You are an experienced triage nurse leader. Your task is to process nurse shift notes into a clinical SBAR report.

Follow these steps carefully:
Step 1: Extract patient identifiers and admission reason for Situation.
Step 2: Extract medical history and admission timing for Background.
Step 3: Extract current vitals (SpO2, BP, HR, Temp) and immediate interventions for Assessment.
Step 4: Extract scheduled tasks, labs, and doctor notifications for Recommendation.

Example Input:
pt Jones, Room 102, 54yo F. Admitted 2 days ago with asthma. Vitals: SpO2 96% on 1L O2, BP 120/75, HR 72. Given Budesonide at 07:00. Follow-up X-ray at 15:00. Dr. Lee informed.

Example Output:
Situation: Patient Jones, 54yo female, Room 102.
Background: Admitted 2 days ago with acute asthma exacerbation.
Assessment: Currently stable on 1L O2 (SpO2 96%), BP 120/75, HR 72. Budesonide administered at 07:00.
Recommendation: Follow-up chest X-ray at 15:00. Dr. Lee notified.

Now process this Nurse Note:
{NURSE_NOTE}
"""
print("┌─ DEMO 3: Principles 3 & 5 (Few-Shot Example + Task Division)")
few_shot_output = generate_response(few_shot_prompt)
for line in few_shot_output.splitlines():
    print(f"│  {line}")
print("└" + "─" * 60)
print()

# ── DEMO 4: Principle 4 (a programmatic evaluation method) ───────────
# WHY: In healthcare you can't judge a summary by feel — a rubric scores it
# objectively, the same way a checklist audits a real handover.
print("=" * 65)
print("  PRINCIPLE 4: Programmatic Evaluation Rubric")
print("=" * 65)
print()


def evaluate_clinical_summary(text):
    """Score an SBAR handover against 4 objective clinical criteria."""
    scores = {}

    # Check 1: Does the output use all 4 SBAR headers?
    headers = ["situation", "background", "assessment", "recommendation"]
    found_headers = sum(1 for h in headers if h in text.lower())
    scores["SBAR Headers (max 4)"] = f"{found_headers}/4"

    # Check 2: Are the critical vitals captured?
    vitals_keywords = ["spo2", "88%", "94%", "138/82", "37.1"]
    found_vitals = sum(1 for v in vitals_keywords if v in text.lower())
    scores["Vitals Capture (max 5)"] = f"{found_vitals}/5"

    # Check 3: Are the pending tasks mentioned?
    tasks = ["abg", "physio"]
    found_tasks = sum(1 for t in tasks if t in text.lower())
    scores["Pending Tasks (max 2)"] = f"{found_tasks}/2"

    # Check 4: Is the physician notification recorded?
    scores["Doctor Notified"] = "Yes" if "schmidt" in text.lower() else "No"

    # Total clinical score = sum of all checks
    total_points = found_headers + found_vitals + found_tasks + (1 if "schmidt" in text.lower() else 0)
    max_points = 4 + 5 + 2 + 1
    scores["Total Clinical Score"] = f"{total_points}/{max_points} ({total_points/max_points:.0%})"

    return scores


outputs = {
    "Demo 1 (Naive Prompt)": naive_output,
    "Demo 2 (Principles 1 & 2)": structured_output,
    "Demo 3 (Principles 1, 2, 3, 5)": few_shot_output,
}

print(f"{'Metric':<28} | {'Demo 1 (Naive)':<18} | {'Demo 2 (P1+2)':<18} | {'Demo 3 (P1+2+3+5)':<18}")
print("-" * 88)

metrics = ["SBAR Headers (max 4)", "Vitals Capture (max 5)", "Pending Tasks (max 2)", "Doctor Notified", "Total Clinical Score"]
results = {name: evaluate_clinical_summary(text) for name, text in outputs.items()}

for m in metrics:
    v1 = results["Demo 1 (Naive Prompt)"][m]
    v2 = results["Demo 2 (Principles 1 & 2)"][m]
    v3 = results["Demo 3 (Principles 1, 2, 3, 5)"][m]
    print(f"{m:<28} | {v1:<18} | {v2:<18} | {v3:<18}")

print()
print("=" * 65)
print("  TAKEAWAY: Applying the 5 principles transformed a vague text")
print("  generator into a reliable, clinical-grade handover tool!")
print("=" * 65)
