"""
Generate reference answers for each rubric question using a lightweight LLM call.
One call per scenario (not per question) — sends GT + all questions, gets back
tailored answers. No PDF needed.

Usage:
    python generate_answers.py --input ../data/scenarios_v5.json --output ../data/scenarios_v5_with_ans.json
    python generate_answers.py --input ../data/scenarios_v5.json --output ../data/scenarios_v5_with_ans.json --model gemini-3.8-flash
"""

import json
import re
import os
import argparse
import subprocess
import time

GEMINI_PYTHON = (
    "/home/idies/workspace/Storage/xyu1/persistent"
    "/pytorch_env/sam3_gcloud/bin/python"
)
GEMINI_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "Nanospectroscopy", "python_version", "benchmark",
    "auto_literature_download", "gemini_call_v2.py",
)

GEMINI_MODEL = None

ANSWER_PROMPT = """You are an expert in XANES spectroscopy. Generate a reference answer for each question below.

RULES:
- Use ONLY the ground truth (GT) and sample conditions provided — do not add any external knowledge
- Each answer should directly address what the specific question asks
- Keep answers concise but complete (3-6 sentences each)
- EVERY answer — regardless of question type — must include a reasoning part
  that explains WHY these phases/fractions arise from the sample conditions:
  * Identification: list the specific phases/references, THEN explain why these
    phases are expected given the sample conditions (composition, temperature, etc.)
  * Quantification: give the numerical fractions, THEN explain why these specific
    values result from the conditions
  * Reasoning: show the full logical chain from conditions → mechanism → outcome
  * Spectral: describe the features, THEN explain what structural/electronic
    properties of this sample produce those features
- Use the key_reasoning field AND the sample conditions to build the explanation

SAMPLE CONDITIONS (what the LLM being tested will see):
{sample_prompt}

GROUND TRUTH:
{gt_json}

QUESTIONS:
{questions_text}

Return ONLY a JSON object mapping question IDs to answer strings:
{{
  "q1": "answer text for q1...",
  "q2": "answer text for q2...",
  ...
}}
"""


def call_gemini(prompt, timeout=120):
    cmd = [GEMINI_PYTHON, GEMINI_SCRIPT]
    if GEMINI_MODEL:
        cmd += ["--model", GEMINI_MODEL]
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            return None
        response = json.loads(result.stdout.strip())
        if not response.get("ok"):
            return None
        return response["text"]
    except Exception:
        return None


def parse_json(raw_text):
    if not raw_text:
        return None
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\n", "", text)
    text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None


def generate_answers_for_scenario(gt, rubric, sample_prompt):
    """One LLM call: sample conditions + GT + all questions → all answers."""
    questions_text = ""
    for qid, q_info in rubric.items():
        qtype = q_info.get("type", "")
        question = q_info.get("question", "")
        questions_text += f"{qid} [{qtype}]: {question}\n"

    # Extract just the conditions part (before ## Questions)
    conditions_part = sample_prompt.split("## Questions")[0].strip()

    prompt = ANSWER_PROMPT.format(
        sample_prompt=conditions_part,
        gt_json=json.dumps(gt, indent=2),
        questions_text=questions_text,
    )

    raw = call_gemini(prompt)
    return parse_json(raw)


def main():
    parser = argparse.ArgumentParser(description="Generate reference answers via LLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", type=str, default="gemini-3.8-flash")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    global GEMINI_MODEL
    GEMINI_MODEL = args.model
    print(f"Using model: {GEMINI_MODEL}")

    with open(args.input) as f:
        scenarios = json.load(f)

    total_answers = 0
    failed = 0

    for i, s in enumerate(scenarios):
        sid = s.get("id", "?")[:50]
        gt = s.get("ground_truth", {})
        rubric = s.get("rubric", {})

        if not rubric:
            continue

        sample_prompt = s.get("prompt", "")
        answers = generate_answers_for_scenario(gt, rubric, sample_prompt)

        if answers and isinstance(answers, dict):
            matched = 0
            for qid, q_info in rubric.items():
                if qid in answers:
                    q_info["answer"] = answers[qid]
                    matched += 1
                    total_answers += 1
            print(f"  [{i+1}/{len(scenarios)}] {sid}: {matched}/{len(rubric)} answers")
        else:
            print(f"  [{i+1}/{len(scenarios)}] {sid}: FAILED")
            failed += 1

        time.sleep(args.sleep)

    with open(args.output, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nGenerated {total_answers} answers across {len(scenarios)} scenarios")
    print(f"Failed: {failed}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
