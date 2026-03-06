import json, random
from pathlib import Path

SYSTEM_PROMPT = """You are a SOC Investigation Quality Scorer.
Given a RAW LOG and an ANALYST INVESTIGATION, output ONLY valid JSON with continuous scores in [0,1].

Use this anchor scale:
0.0 missing/incorrect
0.3 weak
0.5 acceptable
0.7 good
0.9 very good
1.0 excellent

Return JSON only with this schema:
{
  "mitre_accuracy": number,
  "ioc_reasoning": number,
  "contextual_analysis": number,
  "reasoning_quality": number,
  "overall_score": number
}

overall_score MUST equal the arithmetic mean of the 4 metrics rounded to 2 decimals.
Do not add extra keys. Do not add explanations.
"""

def round2(x: float) -> float:
    return float(f"{x:.2f}")

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def main():
    in_path = Path("data/dataset_300.json")
    if not in_path.exists():
        raise FileNotFoundError("data/dataset_300.json not found. Put your dataset there.")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    assert isinstance(data, list), "Expected dataset to be a list of samples"

    rng = random.Random(42)
    rng.shuffle(data)

    n = len(data)
    n_val = max(30, int(0.2 * n))
    val = data[:n_val]
    train = data[n_val:]

    out_train = Path("data/train.jsonl")
    out_val = Path("data/val.jsonl")
    out_train.write_text("", encoding="utf-8")
    out_val.write_text("", encoding="utf-8")

    def to_record(sample):
        raw_log = sample.get("raw_log", {})
        analyst = sample.get("analysis_text", "")
        scores = sample.get("scores", {}) or {}

        mitre = clamp01(scores.get("mitre_accuracy", 0.0))
        ioc = clamp01(scores.get("ioc_depth", 0.0))
        context = clamp01(scores.get("completeness", 0.0))   # proxy for contextual analysis
        reasoning = clamp01(scores.get("reasoning_quality", 0.0))

        overall = round2((mitre + ioc + context + reasoning) / 4.0)

        target = {
            "mitre_accuracy": round2(mitre),
            "ioc_reasoning": round2(ioc),
            "contextual_analysis": round2(context),
            "reasoning_quality": round2(reasoning),
            "overall_score": overall
        }

        user = (
            "RAW LOG:\n" + json.dumps(raw_log, ensure_ascii=False)
            + "\n\nANALYST INVESTIGATION:\n" + analyst
        )

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}
            ]
        }

    with out_train.open("a", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(to_record(s), ensure_ascii=False) + "\n")

    with out_val.open("a", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(to_record(s), ensure_ascii=False) + "\n")

    print(f"train: {len(train)} -> {out_train}")
    print(f"val:   {len(val)} -> {out_val}")

if __name__ == "__main__":
    main()
