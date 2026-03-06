import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "finetune/out/lora_adapter"

def extract_last_json(text: str):
    # text içindeki tüm JSON benzeri objeleri bul, sondakini parse et
    candidates = re.findall(r'\{[\s\S]*?\}', text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            required = {
                "mitre_accuracy",
                "ioc_reasoning",
                "contextual_analysis",
                "reasoning_quality",
                "overall_score",
            }
            if required.issubset(obj.keys()):
                return obj
        except Exception:
            continue
    return None

def recalc_overall(obj):
    return round(
        (
            float(obj["mitre_accuracy"]) +
            float(obj["ioc_reasoning"]) +
            float(obj["contextual_analysis"]) +
            float(obj["reasoning_quality"])
        ) / 4,
        2,
    )

def main():
    with open("data/dataset_300.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    sample = dataset[0]
    raw_log = sample["raw_log"]
    analysis = sample["analysis_text"]

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

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Sadece modelin yeni ürettiği kısmı al
    generated_ids = outputs[0][input_len:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print("\nMODEL OUTPUT\n")
    print(text)

    parsed = extract_last_json(text)

    print("\nPARSED JSON\n")
    print(parsed)

    if parsed:
        checked = recalc_overall(parsed)
        print("\nOVERALL CHECK\n")
        print("model:", parsed["overall_score"])
        print("calc :", checked)

if __name__ == "__main__":
    main()
