import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "finetune/out/lora_adapter"


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            return obj
        except Exception:
            continue
    return None


def extract_score_json(text: str) -> Optional[Dict[str, Any]]:
    obj = extract_last_json(text)
    if not obj:
        return None

    required = {
        "mitre_accuracy",
        "ioc_reasoning",
        "contextual_analysis",
        "reasoning_quality",
        "overall_score",
    }
    if not required.issubset(obj.keys()):
        return None

    try:
        parsed = {
            "mitre_accuracy": float(obj["mitre_accuracy"]),
            "ioc_reasoning": float(obj["ioc_reasoning"]),
            "contextual_analysis": float(obj["contextual_analysis"]),
            "reasoning_quality": float(obj["reasoning_quality"]),
            "overall_score": float(obj["overall_score"]),
        }
    except Exception:
        return None

    return parsed


def extract_feedback_json(text: str) -> Optional[Dict[str, Any]]:
    # 1) Normal parse dene
    obj = extract_last_json(text)
    if obj:
        required = {"missing_points", "what_to_add_next_time", "example_improvement"}
        if required.issubset(obj.keys()):
            if not isinstance(obj["missing_points"], list):
                obj["missing_points"] = []
            if not isinstance(obj["what_to_add_next_time"], list):
                obj["what_to_add_next_time"] = []
            if not isinstance(obj["example_improvement"], str):
                obj["example_improvement"] = str(obj["example_improvement"])
            return obj

    # 2) JSON yarım kesildiyse fallback parse
    missing = re.findall(r'"missing_points"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    add_next = re.findall(r'"what_to_add_next_time"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    improve = re.findall(r'"example_improvement"\s*:\s*"(.*)', text, re.DOTALL)

    def parse_list_block(block: str):
        items = re.findall(r'"(.*?)"', block, re.DOTALL)
        return [x.strip() for x in items if x.strip()]

    result = {
        "missing_points": parse_list_block(missing[0]) if missing else [],
        "what_to_add_next_time": parse_list_block(add_next[0]) if add_next else [],
        "example_improvement": improve[0].strip().rstrip('}').strip() if improve else "",
    }

    if result["missing_points"] or result["what_to_add_next_time"] or result["example_improvement"]:
        return result

    return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, round(float(x), 2)))


def recalc_overall(scores: Dict[str, float]) -> float:
    return round(
        (
            scores["mitre_accuracy"]
            + scores["ioc_reasoning"]
            + scores["contextual_analysis"]
            + scores["reasoning_quality"]
        )
        / 4.0,
        2,
    )


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, messages, max_new_tokens=220) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def score_analysis(tokenizer, model, raw_log: Dict[str, Any], analysis: str) -> Dict[str, float]:
    system_prompt = """
You are a SOC Investigation Quality Scorer.

Return ONLY one valid JSON object and nothing else.

Schema:
{
  "mitre_accuracy": float,
  "ioc_reasoning": float,
  "contextual_analysis": float,
  "reasoning_quality": float,
  "overall_score": float
}

Rules:
- Scores must be between 0.0 and 1.0
- overall_score must equal the mean of the 4 metrics rounded to 2 decimals
- No explanation
- No markdown
- No extra text before or after JSON
""".strip()

    user_prompt = f"""RAW LOG:
{json.dumps(raw_log, ensure_ascii=False, indent=2)}

ANALYST INVESTIGATION:
{analysis}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = generate_text(tokenizer, model, messages, max_new_tokens=120)
    parsed = extract_score_json(text)

    if not parsed:
        raise ValueError(f"Could not parse score JSON from model output:\n{text}")

    parsed = {k: clamp01(v) for k, v in parsed.items()}
    parsed["overall_score"] = recalc_overall(parsed)
    return parsed


def generate_feedback(
    tokenizer,
    model,
    raw_log: Dict[str, Any],
    analysis: str,
    scores: Dict[str, float],
) -> Dict[str, Any]:
    system_prompt = """
You are a SOC investigation QA reviewer.

Given a RAW LOG, an ANALYST INVESTIGATION, and numeric SCORES, produce concise improvement feedback.

Return ONLY one valid JSON object and nothing else.

Schema:
{
  "missing_points": ["...", "..."],
  "what_to_add_next_time": ["...", "..."],
  "example_improvement": "..."
}

Rules:
- Be specific to the log and analyst text
- Do not repeat the raw log
- missing_points: 2 to 4 short bullets
- what_to_add_next_time: 2 to 4 short bullets
- example_improvement: exactly 2 sentences, realistic SOC analyst style
- Focus on what is missing, weak, or could be improved
- No markdown
- No extra text
""".strip()

    user_prompt = f"""RAW LOG:
{json.dumps(raw_log, ensure_ascii=False, indent=2)}

ANALYST INVESTIGATION:
{analysis}

SCORES:
{json.dumps(scores, ensure_ascii=False, indent=2)}

Write feedback that explains what is missing and how the analyst could improve the investigation.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = generate_text(tokenizer, model, messages, max_new_tokens=420)
    parsed = extract_feedback_json(text)

    if not parsed:
        raise ValueError(f"Could not parse feedback JSON from model output:\n{text}")

    parsed["missing_points"] = [str(x).strip() for x in parsed["missing_points"] if str(x).strip()]
    parsed["what_to_add_next_time"] = [
        str(x).strip() for x in parsed["what_to_add_next_time"] if str(x).strip()
    ]
    parsed["example_improvement"] = parsed["example_improvement"].strip()

    return parsed


def main():
    parser = argparse.ArgumentParser(description="SOC score + feedback generator")
    parser.add_argument("--raw-log", required=True, help="Path to raw_log.json")
    parser.add_argument("--analysis", required=True, help="Path to analysis.txt")
    parser.add_argument("--output", required=False, help="Optional output JSON path")
    args = parser.parse_args()

    raw_log = load_json(args.raw_log)
    analysis = load_text(args.analysis)

    tokenizer, model = build_model_and_tokenizer()

    scores = score_analysis(tokenizer, model, raw_log, analysis)
    feedback = generate_feedback(tokenizer, model, raw_log, analysis, scores)

    result = {
        "scores": scores,
        "feedback": feedback,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
