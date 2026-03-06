import json
import re
from typing import Any, Dict, Optional

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_PATH = "finetune/out/lora_adapter"


@st.cache_resource
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


def extract_feedback_json(text: str) -> Optional[Dict[str, Any]]:
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


def quality_label(overall: float) -> str:
    if overall < 0.50:
        return "Poor"
    if overall < 0.75:
        return "Medium"
    return "Good"


def quality_color(overall: float) -> str:
    if overall < 0.50:
        return "#dc2626"
    if overall < 0.75:
        return "#d97706"
    return "#16a34a"


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


def generate_feedback(tokenizer, model, raw_log: Dict[str, Any], analysis: str, scores: Dict[str, float]) -> Dict[str, Any]:
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
    parsed["what_to_add_next_time"] = [str(x).strip() for x in parsed["what_to_add_next_time"] if str(x).strip()]

    cleaned = parsed["example_improvement"].strip().rstrip('"').rstrip("\\").strip()
    parts = re.split(r'(?<=[.!?])\s+', cleaned)
    parsed["example_improvement"] = " ".join(parts[:2]).strip()

    return parsed


def score_card(title: str, value: float):
    pct = int(value * 100)
    bar_color = quality_color(value)
    st.markdown(
        f"""
        <div style="
            border:1px solid #2a2a2a;
            border-radius:14px;
            padding:14px;
            background:#111827;
            min-height:120px;
        ">
            <div style="font-size:14px;color:#cbd5e1;margin-bottom:8px;">{title}</div>
            <div style="font-size:32px;font-weight:700;color:white;">{value:.2f}</div>
            <div style="margin-top:10px;background:#1f2937;border-radius:999px;height:10px;overflow:hidden;">
                <div style="width:{pct}%;background:{bar_color};height:10px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_badge(label: str, overall: float):
    color = quality_color(overall)
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:8px 14px;
            border-radius:999px;
            background:{color}22;
            color:{color};
            border:1px solid {color};
            font-weight:700;
            font-size:14px;
            margin-bottom:12px;
        ">
            {label} · Overall {overall:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="SOC Investigation Scorer", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    textarea, .stTextArea textarea {
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SOC Investigation Quality Scorer")
st.caption("Fine-tuned Mistral + LoRA · Continuous scoring + feedback")

default_log = {
    "EventID": 4688,
    "NewProcessName": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    "ParentProcessName": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
    "CommandLine": "powershell.exe -NoP -W Hidden -EncodedCommand SQBFAFgA",
    "User": "CONTOSO\\unknown_user25",
    "Host": "WIN10-CLIENT01",
    "IntegrityLevel": "Medium",
}

default_analysis = """Process creation telemetry shows powershell encodedcommand execution (T1059.001). The command powershell.exe -NoP -W Hidden -EncodedCommand SQBFAFgA indicates PowerShell execution with encoded input. Parent-child relationship shows powershell spawning powershell, which may indicate script execution. User context CONTOSO\\unknown_user25 should be validated. Overall this activity is suspicious and may indicate post-compromise activity."""

with st.sidebar:
    st.header("Controls")
    if st.button("Load Example Input"):
        st.session_state["raw_log_text"] = json.dumps(default_log, ensure_ascii=False, indent=2)
        st.session_state["analysis_text"] = default_analysis

    st.markdown("---")
    st.subheader("Quality Bands")
    st.write("**Poor**: 0.00 – 0.49")
    st.write("**Medium**: 0.50 – 0.74")
    st.write("**Good**: 0.75 – 1.00")

    st.markdown("---")
    st.subheader("About")
    st.write("This demo scores SOC analyst investigation quality using a fine-tuned local LLM.")
    st.write("Inputs: raw process log + analyst investigation text.")
    st.write("Outputs: continuous metric scores and targeted feedback.")

raw_log_text = st.text_area(
    "RAW LOG (JSON)",
    value=st.session_state.get("raw_log_text", json.dumps(default_log, ensure_ascii=False, indent=2)),
    height=320,
    key="raw_log_text",
)

analysis_text = st.text_area(
    "ANALYST INVESTIGATION",
    value=st.session_state.get("analysis_text", default_analysis),
    height=320,
    key="analysis_text",
)

if st.button("Evaluate", type="primary", use_container_width=True):
    try:
        raw_log = json.loads(raw_log_text)
    except Exception as e:
        st.error(f"RAW LOG JSON parse error: {e}")
        st.stop()

    if not analysis_text.strip():
        st.error("ANALYST INVESTIGATION cannot be empty.")
        st.stop()

    progress = st.progress(0, text="Loading model...")
    try:
        tokenizer, model = build_model_and_tokenizer()
        progress.progress(30, text="Scoring analyst investigation...")
        scores = score_analysis(tokenizer, model, raw_log, analysis_text)
        progress.progress(70, text="Generating feedback...")
        feedback = generate_feedback(tokenizer, model, raw_log, analysis_text, scores)
        progress.progress(100, text="Evaluation complete.")
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.markdown("---")
    st.subheader("Evaluation Result")

    label = quality_label(scores["overall_score"])
    result_badge(label, scores["overall_score"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        score_card("MITRE Accuracy", scores["mitre_accuracy"])
    with c2:
        score_card("IOC Reasoning", scores["ioc_reasoning"])
    with c3:
        score_card("Contextual Analysis", scores["contextual_analysis"])
    with c4:
        score_card("Reasoning Quality", scores["reasoning_quality"])
    with c5:
        score_card("Overall Score", scores["overall_score"])

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("### Missing Points")
        if feedback["missing_points"]:
            for item in feedback["missing_points"]:
                st.write(f"- {item}")
        else:
            st.write("No major missing points detected.")

        st.markdown("### Example Improvement")
        st.write(feedback["example_improvement"])

    with right:
        st.markdown("### What to Add Next Time")
        if feedback["what_to_add_next_time"]:
            for item in feedback["what_to_add_next_time"]:
                st.write(f"- {item}")
        else:
            st.write("No additional guidance available.")

    st.markdown("---")
    with st.expander("Show raw JSON output"):
        st.json({"scores": scores, "feedback": feedback})
