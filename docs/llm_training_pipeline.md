
# LLM Training Pipeline

This document describes the end-to-end training and evaluation pipeline used in the project.

## Pipeline Overview

```text
VMware Cyber Range
        ↓
Atomic Red Team Attack Simulation
        ↓
Windows Telemetry Collection
(4688 / 4104 / other relevant logs)
        ↓
Splunk SIEM Ingestion
        ↓
SPL-based Log Extraction
        ↓
Raw Log Dataset Construction
        ↓
Analyst Investigation Writing
(Good / Medium / Poor)
        ↓
Continuous Quality Scoring
(MITRE / IOC / Context / Reasoning)
        ↓
JSONL Training Set Preparation
        ↓
Mistral 7B QLoRA Fine-Tuning
        ↓
Fine-Tuned SOC Investigation Scorer
        ↓
Batch Evaluation
        ↓
Streamlit Demo
```

## Step-by-Step Description

### 1. Lab Execution
Attack simulations were executed in a VMware-based cyber range using Atomic Red Team.

### 2. Telemetry Collection
Windows process creation and PowerShell logging were enabled. Relevant telemetry included:

- Event ID 4688 (Process Creation)
- Event ID 4104 (PowerShell ScriptBlock Logging)

### 3. Splunk Ingestion
Logs were forwarded to Splunk SIEM and indexed for analysis and extraction.

### 4. SPL Query Filtering
Custom SPL queries were used to isolate technique-specific activity and export the relevant raw telemetry.

### 5. Dataset Construction
Each extracted attack scenario was converted into a dataset entry containing:

- raw log
- analyst investigation text
- quality label / score data

### 6. Investigation Quality Variants
For each scenario, multiple investigation quality levels were written to simulate different analyst skill levels:

- Good
- Medium
- Poor

### 7. Continuous Quality Scoring
Each investigation was scored across four continuous dimensions:

- MITRE Accuracy
- IOC Reasoning
- Contextual Analysis
- Reasoning Quality

The final overall score was calculated as the arithmetic mean of the four dimensions.

### 8. Training Preparation
The dataset was converted into JSONL chat-format examples for supervised fine-tuning.

### 9. Fine-Tuning
Mistral 7B Instruct was fine-tuned locally using QLoRA.

### 10. Evaluation
The resulting model was evaluated on the dataset using batch scoring and compared to the gold continuous scores.

### 11. Demo Deployment
A Streamlit interface was built to demonstrate real-time scoring and feedback generation.

## Purpose of the Pipeline

This pipeline was designed not only to train a scorer, but to show how raw security telemetry can be transformed into a structured SOC reasoning quality dataset and then into a practical analyst evaluation system.
