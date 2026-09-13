# HealthConnect Clinic – Machine Learning Engineering

**AnalystLab Africa Experience Lab**  
**Track:** Machine Learning Engineering  
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data and AI  
**Current phase:** Week 6 — Integration, Advanced Development & Validation

---

## 📌 Project Overview

HealthConnect Clinic is a fictional outpatient healthcare provider facing a high rate of **patient no-shows**. Missed appointments lead to wasted clinical capacity, longer waiting times, and increased administrative overhead.

**Central question:**
> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

As a **Machine Learning Engineer**, my role is to design a **reliable, reproducible, and production-oriented ML system** that predicts the likelihood of a patient missing a scheduled appointment (no-show prediction).

---

## 🔄 Project progression

| Week | Focus | Location |
|------|--------|----------|
| **Week 4** | Problem understanding, system design, architecture | Design docs in `WEEK5/docs/` |
| **Week 5** | EDA, preprocessing, baseline models (Decision Tree, RF, GB) | [`WEEK5/`](./WEEK5/) |
| **Week 6** | **Integrated pipeline, validation, cross-track integration** | [`WEEK6/`](./WEEK6/) ← **current deliverable** |
| Week 7 | Testing, refinement, end-to-end validation | Planned |

> **Week 6 is the main deliverable.**  
> Week 5 is kept for traceability (do not restart the project from scratch).

---

## 📁 Repository structure

```text
.
├── WEEK5/                          # Week 5 – Analysis & baseline models
│   ├── data/
│   ├── notebooks/                  # EDA, preprocessing, modelling
│   ├── models/                     # Baseline artefacts
│   ├── image/                      # EDA & evaluation plots
│   └── docs/                       # Reports & design docs
│
├── WEEK6/                          # Week 6 – Integrated ML pipeline (MAIN)
│   ├── src/                        # Production code
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── preprocessing.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── pipeline.py
│   │   ├── run_pipeline.py         # CLI entry point
│   │   └── utils.py
│   ├── configs/config.yaml
│   ├── tests/
│   ├── data/
│   ├── models/
│   ├── notebooks/                  # Kept for reference
│   ├── docs/
│   │   ├── Week6_Project_Summary.md
│   │   └── Week6_Integration_Evidence.md
│   ├── requirements.txt
│   └── README.md                   # Detailed Week 6 documentation
│
├── README.md                       # This file
└── .gitignore
```

---

## 🚀 Quick start (Week 6 pipeline)

```bash
cd WEEK6
pip install -r requirements.txt

# Train / rebuild artefacts
python -m src.run_pipeline train

# Run integration validation checks
python -m src.run_pipeline validate

# Inference on a CSV
python -m src.run_pipeline infer --input data/raw/HealthConnect_Appointment_Data.csv --output data/processed/predictions.csv
```

Full details: see [`WEEK6/README.md`](./WEEK6/README.md).

---

## 📤 Input → Output contract (inference)

| **Inputs** | **Outputs** |
|------------|-------------|
| Patient demographics & history | `appointment_id` |
| Appointment details & timing | `no_show_probability` (0–1) |
| Reminder information | `no_show_risk_label` (Low / Medium / High) |
| Distance & waiting time | `predicted_class`, `prediction_timestamp`, `model_version` |

---

## 🔗 Week 6 highlights

- Notebook logic extracted into a **modular `src/` package**
- Config-driven pipeline (`configs/config.yaml`)
- Input/output **validation** + logging
- CLI entry point for train / validate / infer
- **Cross-track integration** documented (Data Science models + Analytics insights)
- Prepared for **Week 7 testing**

---

## 📊 Data resources

| Resource | Description |
|----------|-------------|
| `HealthConnect_Appointment_Data.csv` | Fictional & anonymised appointment records |
| `HealthConnect_Data_Dictionary` | Variable definitions and notes |

**Target:** `appointment_outcome` → binary (No-Show incl. Cancelled = 0, Attended = 1)  
**Key predictors:** `booking_lead_days`, distance, patient history, reminders, temporal features

---

## ⚠️ Assumptions, limitations & risks

**Assumptions**
- Dataset is sufficiently representative for a prototype
- Historical features are available at prediction time
- Cancelled appointments merged into No-Show (Week 5 business decision)

**Limitations (current)**
- Model performance still modest (~61% accuracy) — improvement planned with DS in Week 7
- No production deployment yet (Week 6 = integration & validation only)
- Concept-drift monitoring not implemented

**Main risks**

| Risk | Mitigation |
|------|------------|
| Temporal data leakage | Strict chronological train/test split |
| Class imbalance | Documented; threshold & risk labels configurable |
| Over-reliance on predictions | Clear limitations + human-in-the-loop |

---

## 🛠️ Tech stack

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **ML:** scikit-learn (Decision Tree baseline, Random Forest)
- **Config:** YAML
- **Tracking:** Git + GitHub, model artefacts
- **Environment:** `requirements.txt`

---

## 👤 Author

**Machine Learning Engineering**  
Tenon KONE  
AnalystLab Africa Experience Lab

---

## 📝 License & notes

This project is part of the **AnalystLab Africa Experience Lab** internship programme.  
All data is **fictional and anonymised**. Do not treat results as real-world clinical recommendations.

---

*Last updated: Week 6 – Integration, Advanced Development & Validation*
