
# Dataset Generation Methodology

This dataset was created to train and evaluate a SOC investigation quality scoring model.

## Lab Environment

The dataset was generated inside a controlled cyber range.

Infrastructure:
- VMware virtual lab
- Windows endpoints
- Splunk SIEM
- Atomic Red Team attack simulations

## Attack Simulation

Attacks were executed using **Atomic Red Team** techniques mapped to MITRE ATT&CK.

Focus areas:
- Discovery
- Defense Evasion
- Enumeration activities

Each technique was executed multiple times to produce different telemetry scenarios.

## Telemetry Collection

Telemetry sources:

| Event ID | Description |
|---|---|
| 4688 | Process Creation |
| 4104 | PowerShell ScriptBlock Logging |

Logs were forwarded into Splunk for indexing.

## Log Extraction

Relevant logs were extracted using Splunk SPL queries designed to isolate attack behaviors.

These queries filtered for:
- enumeration commands
- registry access
- network discovery
- PowerShell persistence behaviors

## Investigation Writing

For each attack scenario:

Three analyst investigations were created:

| Type | Description |
|---|---|
| Good | Detailed reasoning and context |
| Medium | Partial reasoning |
| Poor | Minimal investigation |

## Scoring Model

Each investigation was scored across four dimensions:

- MITRE Accuracy
- IOC Reasoning
- Contextual Analysis
- Reasoning Quality

Score range:

0.0 → missing  
0.5 → acceptable  
1.0 → excellent  

Overall score formula:

overall_score = (mitre + ioc + context + reasoning) / 4

## Final Dataset

Dataset size:

- Techniques: **10**
- Scenarios per technique: **10**
- Investigation variants per scenario: **3**

Total samples:

**300**
