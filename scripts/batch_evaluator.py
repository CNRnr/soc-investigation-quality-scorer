import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "finetune/out/lora_adapter"
DATASET_PATH = "data/dataset_300.json"
OUTPUT_PATH = "output/batch_results.jsonl"


def extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    candidates = re.findall(r"\{[\s\S]*?\}", text)
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def extract_score_json(text: str) -> Optional[Dict[str, float]]:
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


def generate_text(tokenizer, model, messages, max_new_tokens=120) -> str:
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


def get_gold_scores(sample: Dict[str, Any]) -> Dict[str, float]:
    scores = sample.get("scores", {}) or {}
    gold = {
        "mitre_accuracy": clamp01(scores.get("mitre_accuracy", 0.0)),
        "ioc_reasoning": clamp01(scores.get("ioc_depth", 0.0)),
        "contextual_analysis": clamp01(scores.get("completeness", 0.0)),
        "reasoning_quality": clamp01(scores.get("reasoning_quality", 0.0)),
    }
    gold["overall_score"] = recalc_overall(gold)
    return gold


def mae(a: float, b: float) -> float:
    return abs(a - b)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def main():
    Path("output").mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenizer, model = build_model_and_tokenizer()

    results = []
    maes = {
        "mitre_accuracy": [],
        "ioc_reasoning": [],
        "contextual_analysis": [],
        "reasoning_quality": [],
        "overall_score": [],
    }

    pred_avgs = {k: [] for k in maes.keys()}
    gold_avgs = {k: [] for k in maes.keys()}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for idx, sample in enumerate(dataset, start=1):
            raw_log = sample["raw_log"]
            analysis = sample["analysis_text"]

            gold = get_gold_scores(sample)
            pred = score_analysis(tokenizer, model, raw_log, analysis)

            row = {
                "index": idx,
                "attack_id": sample.get("attack_id"),
                "attack_name": sample.get("attack_name"),
                "analysis_quality": sample.get("analysis_quality"),
                "gold_scores": gold,
                "pred_scores": pred,
                "delta": {
                    k: round(pred[k] - gold[k], 2)
                    for k in gold.keys()
                }
            }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.append(row)

            for k in maes.keys():
                maes[k].append(mae(pred[k], gold[k]))
                pred_avgs[k].append(pred[k])
                gold_avgs[k].append(gold[k])

            if idx % 25 == 0:
                print(f"Processed {idx}/{len(dataset)}")

    summary = {
        "samples": len(results),
        "average_pred_scores": {k: mean(v) for k, v in pred_avgs.items()},
        "average_gold_scores": {k: mean(v) for k, v in gold_avgs.items()},
        "mae": {k: mean(v) for k, v in maes.items()},
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open("output/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
