# SOC Investigation Quality Scoring AI

AI-powered system for evaluating SOC analyst investigations using continuous security reasoning metrics.

This project presents a local LLM-based evaluator that scores SOC analyst investigations based on raw security logs and analyst-written investigation reports.

The system evaluates investigation quality across four dimensions:

- MITRE Accuracy
- IOC Reasoning
- Contextual Analysis
- Reasoning Quality

Each investigation receives a continuous score between **0 and 1**, along with feedback describing missing reasoning elements and suggestions for improvement.

## Use Cases

### SOC Performance Measurement
SOC teams can adapt this system using their own investigation standards and operational knowledge. The model can be trained to reflect the SOC's internal investigation methodology and used to benchmark analyst performance and investigation quality.

### SOC Service Quality Benchmarking
The system allows SOC teams to demonstrate investigation quality to customers by measuring the analytical depth and reasoning quality of investigations.

### SOC Analyst Training
The system can also be used as an analyst training assistant. Analysts can submit their investigation reports and receive feedback on missing reasoning steps and how to improve their analysis.

## Architecture

Raw Log + Analyst Investigation  
↓  
Fine-tuned LLM Evaluator  
↓  
Quality Scores (MITRE / IOC / Context / Reasoning)  
↓  
Feedback Generator  
↓  
Investigation Improvement Suggestions

## Technologies

- Python
- Mistral 7B
- QLoRA fine-tuning
- HuggingFace Transformers
- PEFT / TRL
- Streamlit
- Splunk
- Atomic Red Team
- VMware Lab Environment

## Demo

The repository includes a Streamlit interface that allows users to:

- input raw security logs
- write analyst investigations
- generate quality scores
- receive improvement feedback

Run the demo:

streamlit run streamlit_app.py


## Research Goal

This project explores whether fine-tuned local LLMs can reliably evaluate SOC analyst reasoning quality and provide meaningful feedback for investigation improvement.
