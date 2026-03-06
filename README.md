# SOC Investigation Quality Scoring AI

# SOC Investigation Quality Scoring AI

AI-assisted SOC investigation quality assurance, analyst coaching, and performance benchmarking system.

This project presents a local LLM-based evaluator designed to score the quality of SOC analyst investigations from raw security logs and analyst-written investigation reports. The system produces continuous quality scores across multiple reasoning dimensions and generates structured feedback to help improve investigation depth, consistency, and analytical quality.

---

## Why This Project Matters

In most SOC environments, investigation quality is difficult to measure consistently. Analysts may review the same alert differently, and investigation quality often depends on personal experience, team culture, and undocumented reasoning habits.

This project explores a different approach:

**Can a fine-tuned local LLM evaluate SOC analyst investigation quality in a structured and repeatable way?**

The answer, in this prototype, is yes.

---

## Core Idea

Instead of treating an investigation as simply correct or incorrect, this system evaluates the **quality of the reasoning** inside the investigation.

The model scores analyst investigations across four dimensions:

- **MITRE Accuracy**  
  How accurately the investigation maps observed behavior to ATT&CK techniques.

- **IOC Reasoning**  
  How well the analyst explains why the observed indicators matter.

- **Contextual Analysis**  
  How well the analyst uses user, host, process ancestry, integrity level, and execution context.

- **Reasoning Quality**  
  How coherent, actionable, and analytically useful the investigation is.

Each metric is scored on a **continuous scale between 0 and 1**, and the system also calculates an **overall investigation quality score**.

---

## SOC Team Use Case

This concept is especially valuable for SOC teams that want to formalize and scale their own internal investigation standards.

A SOC can adapt this system to its own:

- detection philosophy
- analyst workflow
- escalation standards
- customer reporting expectations
- internal investigation culture
- operational know-how

By training the evaluator on investigations that reflect a SOC team's own quality standards, the system can become a **custom investigation quality model** aligned with the team's real operational practice.

This makes it possible to:

- standardize investigation quality
- benchmark analyst performance
- scale internal SOC methodology
- measure reasoning quality over time
- show customers the analytical quality of delivered investigation work

In other words, this can evolve into a **SOC quality measurement and service benchmarking product**.

---

## SOC Analyst Training Use Case

This system can also be used as a **SOC analyst training and coaching tool**.

An analyst can submit:

- a raw log
- a written investigation

and the system will return:

- structured quality scores
- missing reasoning points
- suggestions for improvement
- an example of a stronger investigation

That makes the system useful for:

- analyst onboarding
- junior analyst development
- investigation writing practice
- quality review exercises
- internal SOC training programs

---

## Architecture

Raw Security Log
+
Analyst Investigation
        ↓
Fine-Tuned Local LLM Evaluator
        ↓
Continuous Quality Scores
(MITRE / IOC / Context / Reasoning)
        ↓
Feedback Generator
        ↓
Investigation Improvement Suggestions

Dataset

The dataset used in this project was built from a controlled lab workflow:

Attack simulation with Atomic Red Team

Execution in a VMware cyber range

Log collection through Splunk SIEM

Raw log extraction

Analyst investigation writing

Quality scoring with continuous labels

Dataset characteristics

10 attack techniques

10 scenarios per technique

Good / Medium / Poor investigation examples

Continuous scoring model

Total samples: 300

Model and Training

This project uses:

Mistral 7B Instruct

QLoRA fine-tuning

local GPU training

a custom SOC investigation quality dataset

The model was trained to learn investigation reasoning quality patterns, not just static label mapping.

Evaluation Results

The fine-tuned evaluator was tested against the gold continuous scores in the dataset.

| Metric              | MAE        |
| ------------------- | ---------- |
| MITRE Accuracy      | 0.0265     |
| IOC Reasoning       | 0.0285     |
| Contextual Analysis | 0.0268     |
| Reasoning Quality   | 0.0292     |
| Overall Score       | **0.0173** |

These results indicate that the evaluator closely matches the reference scoring logic used during dataset construction.

Demo

This repository includes a Streamlit demo that allows users to:

paste raw security logs

write analyst investigations

generate continuous quality scores

receive structured improvement feedback

Run locally:

pip install -r requirements.txt
streamlit run streamlit_app.py

Example Output

{
  "scores": {
    "mitre_accuracy": 0.98,
    "ioc_reasoning": 0.88,
    "contextual_analysis": 0.86,
    "reasoning_quality": 0.88,
    "overall_score": 0.90
  },
  "feedback": {
    "missing_points": [
      "Validate user legitimacy and logon source",
      "Check for nearby suspicious events"
    ],
    "what_to_add_next_time": [
      "Include process ancestry and nearby network connections",
      "Use contextual data to determine intent and risk"
    ],
    "example_improvement": "..."
  }
}

Repository Structure

soc-investigation-quality-scorer/
├── data/
├── scripts/
├── input/
├── output/
├── finetune/
├── streamlit_app.py
├── README.md
└── .gitignore

Research Goal

This project investigates whether a fine-tuned local LLM can reliably evaluate SOC analyst reasoning quality and provide meaningful feedback for investigation improvement.

The broader goal is to move toward:

AI-assisted SOC investigation quality assurance

analyst coaching systems

performance benchmarking tools

customer-facing SOC quality measurement

Future Work

Potential extensions include:

larger and more diverse datasets

more ATT&CK techniques

better benign/admin scenario coverage

stronger feedback generation

analyst coaching mode

cross-team and cross-SOC benchmarking

model deployment without shipping full training artifacts in the repo
