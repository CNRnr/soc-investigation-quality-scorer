
# Dataset Schema

This document describes the structure of the SOC investigation quality dataset used in the project.

## Dataset Format

The dataset is stored as JSON and contains one entry per investigation sample.

Each sample represents:

- one attack scenario
- one analyst-written investigation
- one set of quality scores

## Example High-Level Structure

```json
{
  "attack_id": "T1057",
  "attack_name": "Process Discovery",
  "scenario_id": "T1057_SCN_01",
  "scenario_type": "malicious",
  "raw_log": {
    "EventID": 4688,
    "NewProcessName": "C:\\Windows\\System32\\tasklist.exe",
    "ParentProcessName": "C:\\Windows\\System32\\cmd.exe",
    "CommandLine": "tasklist.exe /svc",
    "User": "CONTOSO\\user1",
    "Host": "WIN10-CLIENT01",
    "IntegrityLevel": "Medium"
  },
  "analysis_text": "The command tasklist.exe /svc indicates process discovery activity...",
  "analysis_quality": "good",
  "scores": {
    "mitre_accuracy": 0.90,
    "ioc_depth": 0.85,
    "completeness": 0.80,
    "reasoning_quality": 0.88
  }
}
```

## Field Definitions

| Field | Type | Description |
|---|---|---|
| `attack_id` | string | MITRE ATT&CK technique ID |
| `attack_name` | string | Technique name |
| `scenario_id` | string | Unique scenario identifier |
| `scenario_type` | string | Scenario type such as malicious, suspicious, or benign_admin |
| `raw_log` | object | Raw Windows/Splunk-derived security event |
| `analysis_text` | string | Analyst-written investigation text |
| `analysis_quality` | string | Investigation quality label such as good / medium / poor |
| `scores` | object | Continuous quality scores |

## `raw_log` Object

The `raw_log` field typically contains telemetry such as:

| Field | Description |
|---|---|
| `EventID` | Windows event identifier |
| `NewProcessName` | Process created |
| `ParentProcessName` | Parent process |
| `CommandLine` | Executed command |
| `User` | User context |
| `Host` | Hostname |
| `IntegrityLevel` | Process integrity level |

Depending on the source event, additional fields may also appear.

## `scores` Object

The `scores` field contains the gold continuous labels used for training and evaluation.

| Field | Description |
|---|---|
| `mitre_accuracy` | Quality of ATT&CK mapping |
| `ioc_depth` | Depth of IOC interpretation |
| `completeness` | Coverage of contextual elements |
| `reasoning_quality` | Overall reasoning clarity and quality |

## Score Range

All score values are continuous and normalized between `0.0` and `1.0`.

Interpretation:

| Score | Meaning |
|---|---|
| 0.0 | Missing / incorrect |
| 0.3 | Weak |
| 0.5 | Acceptable |
| 0.7 | Good |
| 0.9 | Very good |
| 1.0 | Excellent |

## Overall Score

During inference, the evaluator computes:

```text
overall_score = (mitre_accuracy + ioc_reasoning + contextual_analysis + reasoning_quality) / 4
```

## Schema Notes

- `ioc_depth` in the original dataset is used as the training proxy for `ioc_reasoning`
- `completeness` is used as the training proxy for `contextual_analysis`
- the dataset is designed for continuous quality scoring, not binary classification

## Intended Use

This schema supports:

- supervised fine-tuning
- batch evaluator benchmarking
- Streamlit scoring demo
- analyst training and feedback workflows
