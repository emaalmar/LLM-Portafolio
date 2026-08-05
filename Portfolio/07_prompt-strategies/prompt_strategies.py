"""
Project 7: Prompt Strategies — Zero-Shot vs. Few-Shot vs. Chain-of-Thought
========================================================================
Task: Evaluate and compare how three standard prompting strategies perform
at classifying Emergency Department patient triage severity levels.

This standalone, CPU-friendly demo uses Qwen2.5-0.5B-Instruct to show:
  1. Zero-Shot (no examples)
  2. Few-Shot (in-context exemplars)
  3. Chain-of-Thought (intermediate step-by-step reasoning)

Evaluation is done programmatically against ground-truth labels.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Step 1: Load Local Instruction Model ──────────────────────────────
# WHY: Instruction-tuned models are optimized to recognize formatting cues,
# roles, and reasoning instructions in conversational prompts.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 65)
print(f"Loading local model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
print("  Model loaded successfully!")
print("=" * 65)
print()

# ── Step 2: Define Clinical Patient Cases (Ground Truth Data) ─────────
# WHY: We run our prompting strategies against three realistic triage notes:
# - Case 1: An emergency cardiac scenario (critical vitals, crushing pain).
# - Case 2: An urgent acute pain/fever scenario (suspected kidney infection).
# - Case 3: A non-urgent musculoskeletal trauma (stable vitals, simple sprain).
PATIENT_CASES = [
    {
        "id": "Case 1",
        "expected": "Emergency",
        "note": "62yo M with acute central crushing chest pain radiating to left jaw, onset 40 mins ago. SpO2 89% on room air, BP 82/50, HR 115, Temp 36.8C. Diaphoretic and pale."
    },
    {
        "id": "Case 2",
        "expected": "Urgent",
        "note": "45yo F presenting with 2-day history of right flank pain and chills. Temp 38.9C, BP 118/74, HR 96, SpO2 97%. Severe costovertebral angle tenderness."
    },
    {
        "id": "Case 3",
        "expected": "Non-Urgent",
        "note": "24yo M twisted right ankle while playing basketball 2 days ago. Mild edema, no deformity, weight-bearing with slight limp. Vitals: BP 122/78, HR 68, SpO2 99%, Temp 36.6C."
    }
]


def generate_response(prompt_text, max_new_tokens=250):
    """Generate a text response deterministically from the model.
    WHY: do_sample=False keeps results reproducible across prompt strategy comparisons."""
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


# ── Step 3: Define Prompt Builders (Zero-Shot, Few-Shot, CoT) ─────────

def build_zero_shot_prompt(note_text):
    """
    Strategy 1: Zero-Shot Prompting
    WHY: Direct task description without any examples. Evaluates baseline model behavior.
    Uses clean XML block delimiters to separate rules from clinical input.
    """
    return f"""<instructions>
You are an Emergency Department Triage Nurse. Classify the patient note into exactly one category:
- Emergency
- Urgent
- Non-Urgent

Respond with only the category name.
</instructions>

<patient_note>
{note_text}
</patient_note>"""


def build_few_shot_prompt(note_text):
    """
    Strategy 2: Few-Shot Prompting (In-Context Learning)
    WHY: Demonstrates expected output schema and classification patterns via
    exemplars. Allows the model to adapt "on the fly" without changing weights.
    """
    return f"""<instructions>
You are an Emergency Department Triage Nurse. Classify the patient note into exactly one category:
- Emergency
- Urgent
- Non-Urgent

Follow the pattern shown in the examples.
</instructions>

<examples>
<example>
<patient_note>
70yo F found unresponsive by family. BP 70/40, HR 130, SpO2 82%. Severe respiratory distress.
</patient_note>
<classification>Emergency</classification>
</example>

<example>
<patient_note>
19yo M requesting refill for routine allergy medication. Vitals normal. No acute distress.
</patient_note>
<classification>Non-Urgent</classification>
</example>
</examples>

<patient_note>
{note_text}
</patient_note>"""


def build_cot_prompt(note_text):
    """
    Strategy 3: Chain-of-Thought (CoT) Prompting (Step-by-Step Reasoning)
    WHY: Guides the model through sequential reasoning steps (vitals → symptoms → risk factors)
    before printing the classification. Generating reasoning tokens first acts as a "working memory".
    """
    return f"""<instructions>
You are an Emergency Department Triage Nurse.
Analyze the patient note step-by-step before determining the triage category.

Follow these steps in your response:
1. Examine vital signs and flag any life-threatening values.
2. Evaluate reported symptoms and chief complaint risk.
3. Determine final category: Emergency, Urgent, or Non-Urgent.

Format your output using these tags:
<reasoning>
[Your step-by-step clinical evaluation]
</reasoning>
<classification>[Emergency | Urgent | Non-Urgent]</classification>
</instructions>

<patient_note>
{note_text}
</patient_note>"""


# ── Step 4: Run Comparison Experiment ─────────────────────────────────
print("Running Triage Classification across 3 Strategies ...\n")

results = []

for case in PATIENT_CASES:
    case_id = case["id"]
    expected = case["expected"]
    note = case["note"]

    # Generate model responses
    out_zero = generate_response(build_zero_shot_prompt(note), max_new_tokens=50)
    out_few = generate_response(build_few_shot_prompt(note), max_new_tokens=50)
    out_cot = generate_response(build_cot_prompt(note), max_new_tokens=250)

    # Programmatic helper to extract classification tags
    def extract_label(text):
        for category in ["Emergency", "Urgent", "Non-Urgent"]:
            if category.lower() in text.lower():
                return category
        return "Unknown"

    pred_zero = extract_label(out_zero)
    pred_few = extract_label(out_few)
    pred_cot = extract_label(out_cot)

    results.append({
        "case": case_id,
        "expected": expected,
        "zero": (pred_zero, pred_zero == expected, out_zero),
        "few": (pred_few, pred_few == expected, out_few),
        "cot": (pred_cot, pred_cot == expected, out_cot)
    })

# ── Step 5: Output Results Matrix ─────────────────────────────────────
print("=" * 75)
print(f"{'Case':<8} | {'Expected':<12} | {'Zero-Shot':<15} | {'Few-Shot':<15} | {'Chain-of-Thought':<15}")
print("-" * 75)

zero_correct = 0
few_correct = 0
cot_correct = 0

for r in results:
    z_str = f"{r['zero'][0]} {'✓' if r['zero'][1] else '✗'}"
    f_str = f"{r['few'][0]} {'✓' if r['few'][1] else '✗'}"
    c_str = f"{r['cot'][0]} {'✓' if r['cot'][1] else '✗'}"

    if r['zero'][1]: zero_correct += 1
    if r['few'][1]: few_correct += 1
    if r['cot'][1]: cot_correct += 1

    print(f"{r['case']:<8} | {r['expected']:<12} | {z_str:<15} | {f_str:<15} | {c_str:<15}")

print("-" * 75)
num_cases = len(PATIENT_CASES)
print(f"{'ACCURACY':<23} | {zero_correct/num_cases:<15.0%} | {few_correct/num_cases:<15.0%} | {cot_correct/num_cases:<15.0%}")
print("=" * 75)
print()

# Detailed raw outputs for clinical inspection
print("┌─ DETAILED OUTPUTS BY CASE & STRATEGY:")
for r in results:
    print(f"├─ {r['case']} (Expected: {r['expected']})")
    print(f"│  Zero-Shot Output : {r['zero'][2]}")
    print(f"│  Few-Shot Output  : {r['few'][2]}")
    print(f"│  CoT Output       :\n│    " + r['cot'][2].replace('\n', '\n│    '))
    print("│")
print("└" + "─" * 70)
