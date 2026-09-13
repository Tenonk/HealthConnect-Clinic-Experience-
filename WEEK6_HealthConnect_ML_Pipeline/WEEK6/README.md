# HealthConnect Clinic – Integrated ML Pipeline

**AnalystLab Africa Experience Lab | Week 6**  
**Track:** Machine Learning Engineering  
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data and AI

---

## 📌 Project Overview

HealthConnect Clinic faces a high rate of **patient no-shows**. This repository contains the **integrated and validated ML pipeline** that predicts the likelihood of a patient missing a scheduled appointment.

**Central question:**  
> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

**Week 6 focus:** Integration → Advanced Development → Validation  
(Building on the Week 5 baseline models, preprocessing and notebooks.)

---

## 🔄 Week 5 → Week 6 Transition

| Item | Description |
|------|-------------|
| **Main Week 5 output** | Notebooks for EDA, temporal preprocessing, baseline models (Decision Tree, Random Forest, Gradient Boosting) + saved artefacts (`column_transformer.pkl`, `decision_tree.pkl`, `random_forest.pkl`) |
| **Most important component** | Decision Tree baseline (Accuracy ≈ 61.1 %, F1 ≈ 0.555) + reusable ColumnTransformer |
| **Main limitation** | Pipeline lived only inside notebooks; no reproducible CLI, no input/output validation, no logging, no automated integration checks |
| **Relevant tracks** | Data Science (models & features), Data Analytics (insights that informed feature engineering) |
| **Week 6 intent** | Turn notebook artefacts into a **validated, integrated, reproducible pipeline** ready for Week 7 testing |

---

## 🎯 Week 6 Objectives Achieved

- [x] Review Week 5 pipeline structure and identify integration gaps
- [x] Integrate Data Science model artefacts into a production-oriented pipeline
- [x] Ensure data processing ↔ model input/output compatibility
- [x] Implement validation checks for pipeline inputs and outputs
- [x] Test interaction between preprocessing and model components
- [x] Add logging and configuration management
- [x] Document technical dependencies and prepare for Week 7 testing

---

## 📁 Repository Structure (Week 6)

```text
WEEK6/
├── configs/
│   └── config.yaml                 # Central configuration
├── data/
│   ├── raw/                        # Original immutable dataset
│   └── processed/                  # Transformed features + predictions
├── docs/                           # Design docs, Week 5 report, integration notes
├── image/                          # EDA & evaluation plots from Week 5
├── models/
│   ├── column_transformer.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
├── notebooks/                      # Original Week 5 exploratory notebooks (kept for reference)
├── src/
│   ├── __init__.py
│   ├── data_validation.py          # Input / output validation
│   ├── feature_engineering.py      # Reproducible feature engineering
│   ├── preprocessing.py            # ColumnTransformer builder
│   ├── train.py                    # Model training & evaluation
│   ├── predict.py                  # Batch inference + risk labels
│   ├── pipeline.py                 # End-to-end orchestration
│   ├── run_pipeline.py             # CLI entry point
│   └── utils.py                    # Config, logging, helpers
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train / rebuild pipeline artefacts
python -m src.run_pipeline train

# Run integration validation checks
python -m src.run_pipeline validate

# Run inference on a CSV
python -m src.run_pipeline infer --input data/raw/HealthConnect_Appointment_Data.csv --output data/processed/predictions.csv
```

---

## 🔗 Cross-Track Integration (Mandatory Week 6)

| Field | Content |
|-------|---------|
| **Track collaborated with** | Data Science (model artefacts & evaluation) + Data Analytics (feature insights) |
| **Project dependency** | Candidate models + ColumnTransformer produced in Week 5 notebooks |
| **Information / output received** | `decision_tree.pkl`, `random_forest.pkl`, `column_transformer.pkl`, feature list, performance metrics, key predictors (`booking_lead_days`, distance, history…) |
| **Information / output provided** | Integrated pipeline interface, validation contract, reproducible training & inference entry points, risk-label output schema |
| **Integration activity completed** | Encapsulated notebook logic into modular `src/` package; added validation, logging, config-driven execution; verified end-to-end data flow |
| **What changed as a result** | Pipeline is now callable, testable and reproducible outside notebooks; clear input/output contract for downstream consumers (dashboards, GenAI, operations) |
| **Evidence** | `src/pipeline.py`, `configs/config.yaml`, `tests/test_pipeline.py`, this README, logs produced by `run_pipeline validate` |

---

## 📤 Input → Output Contract (Inference)

| **Inputs** | **Outputs** |
|------------|-------------|
| patient demographics & history | `appointment_id` |
| appointment details & timing | `no_show_probability` (0–1) |
| reminder information | `no_show_risk_label` (Low / Medium / High) |
| distance & waiting time | `predicted_class`, `prediction_timestamp`, `model_version` |

---

## 🧪 Integration Validation Checks

The command `python -m src.run_pipeline validate` verifies:

1. All required model artefacts exist  
2. Preprocessor and model can be loaded  
3. End-to-end inference on a sample succeeds  
4. Output schema matches the contract  

---

## ⚠️ Limitations & Risks (Updated Week 6)

| Area | Status |
|------|--------|
| Performance still modest (~61 % accuracy) | Unchanged – model improvement is Week 7 / DS responsibility |
| Class imbalance & threshold sensitivity | Documented; risk labels use configurable thresholds |
| Temporal leakage | Mitigated by strict chronological split |
| Concept drift | Not yet monitored – planned for Week 7/8 |
| sklearn version compatibility | Pipeline rebuilds transformer to avoid pickle version issues |
| External data (weather, holidays) | Not available – noted as future enhancement |

---

## 📋 Week 7 Testing Requirements (Prepared)

1. Unit tests for each `src/` module with edge cases  
2. Integration tests on full temporal test set  
3. Threshold / calibration analysis  
4. Latency & memory profiling of the pipeline  
5. Compatibility checks with Data Analytics dashboards and GenAI assistant  
6. Documentation of failure modes and escalation paths  

---

## 👤 Author

**Machine Learning Engineering**  
Tenon KONE  
AnalystLab Africa Experience Lab – Week 6

---

## 📝 Notes

- All data is **fictional and anonymised**.  
- Week 6 does **not** claim production deployment readiness.  
- Progression: Week 5 notebooks → Week 6 integrated pipeline → Week 7 testing & refinement.

*Last updated: Week 6 – Integration, Advanced Development & Validation*
