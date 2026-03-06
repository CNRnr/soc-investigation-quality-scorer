import ollama
import json
import re

system_prompt = """
You are a SOC Investigation Quality Evaluator.

IMPORTANT:
- You DO NOT determine if activity is malicious.
- You ONLY evaluate the QUALITY of the analyst's investigation.
- You MUST evaluate ONLY what is explicitly written in the analyst analysis.
- Do NOT infer or assume missing reasoning.

Scoring Criteria:

1. MITRE Accuracy (0-1)
2. IOC Reasoning (0-1)
3. Contextual Analysis (0-1)
4. Reasoning Quality (0-1)

Be strict. Do not be generous.

Return STRICTLY in JSON format:

{
  "mitre_accuracy": {"score": float, "justification": ""},
  "ioc_reasoning": {"score": float, "justification": ""},
  "contextual_analysis": {"score": float, "justification": ""},
  "reasoning_quality": {"score": float, "justification": ""},
  "confidence_penalty_applied": true/false,
  "missing_points": [],
  "final_comment": ""
}
"""

# --- SAMPLE INPUT ---
raw_log = """
EventID: 4688
NewProcessName: powershell.exe
CommandLine: powershell.exe -EncodedCommand SQBFAFgA
User: CORP\\unknown_user
"""

analyst_analysis = """
PowerShell executed with encoded command. This is suspicious activity.
"""

# --- PROMPT ---
user_prompt = f"""
RAW LOG:
{raw_log}

ANALYST ANALYSIS:
{analyst_analysis}
"""

response = ollama.chat(
    model='mistral',
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

content = response['message']['content']

# --- PARSE JSON SAFELY ---
try:
    result = json.loads(content)
except:
    print("Model did not return valid JSON.")
    print(content)
    exit()

# =========================
# 🔥 RULE ENFORCEMENT LAYER
# =========================

analysis_lower = analyst_analysis.lower()

# --- MITRE Enforcement ---
if not re.search(r"t\d{4}", analysis_lower) and "mitre" not in analysis_lower:
    result["mitre_accuracy"]["score"] = min(result["mitre_accuracy"]["score"], 0.3)
    result["missing_points"].append("Explicit MITRE technique mapping missing")

# --- IOC Enforcement ---
ioc_keywords = ["decode", "base64", "parameter", "command breakdown"]
if not any(keyword in analysis_lower for keyword in ioc_keywords):
    result["ioc_reasoning"]["score"] = min(result["ioc_reasoning"]["score"], 0.4)

# --- Context Enforcement ---
context_keywords = ["parent", "process chain", "user context", "host", "privilege"]
if not any(keyword in analysis_lower for keyword in context_keywords):
    result["contextual_analysis"]["score"] = min(result["contextual_analysis"]["score"], 0.3)

# --- Reasoning Quality Enforcement ---
if len(analyst_analysis.split()) < 25:
    result["reasoning_quality"]["score"] = min(result["reasoning_quality"]["score"], 0.4)

# --- FINAL SCORE CALCULATION ---
final_score = (
    result["mitre_accuracy"]["score"] +
    result["ioc_reasoning"]["score"] +
    result["contextual_analysis"]["score"] +
    result["reasoning_quality"]["score"]
) / 4

result["overall_score"] = round(final_score, 2)

# --- OUTPUT ---
print(json.dumps(result, indent=2))