# HealthConnect Clinic – Machine Learning Engineering

**AnalystLab Africa Experience Lab**  
**Track:** Machine Learning Engineering  
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data and AI  
**Current phase:** Week 7 — Testing, Refinement & End-to-End Validation

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
| **Week 6** | Integrated pipeline, validation, cross-track integration | [`WEEK6/`](./WEEK6/) |
| **Week 7** | **Testing, refinement, end-to-end validation** | [`WEEK7/`](./WEEK7/) ← **current deliverable** |
| Week 8 | Final integration → Presentation | Planned |

> **Week 7 is the main deliverable.**  
> Previous weeks are kept for traceability (do not restart the project from scratch).

---

## 📁 Repository structure

```text
.
├── WEEK5/                          # Week 5 – Analysis & baseline models
│   ├── data/
│   ├── notebooks/
│   ├── models/
│   ├── image/
│   └── docs/
│
├── WEEK6/                          # Week 6 – Integrated ML pipeline
│   ├── src/
│   ├── configs/
│   ├── tests/
│   ├── docs/Week6_*.md
│   └── README.md
│
├── WEEK7/                          # Week 7 – Testing & reliability (MAIN)
│   ├── src/                        # Production pipeline
│   ├── configs/config.yaml         # v7.0.0
│   ├── tests/test_pipeline.py      # 22 automated tests
│   ├── docs/
│   │   ├── Week7_Test_Plan.md
│   │   ├── Week7_Testing_Record.md
│   │   ├── Week7_CrossTrack_Evidence.md
│   │   └── Week7_Project_Summary.md
│   ├── models/
│   ├── data/
│   ├── requirements.txt
│   └── README.md
│
├── README.md                       # This file
└── .gitignore
```

---

## 🚀 Quick start (Week 7 – current)

```bash
cd WEEK7
pip install -r requirements.txt
pip install pytest

# Run automated test suite (22 tests)
pytest tests/ -v

# Pipeline CLI
python -m src.run_pipeline validate
python -m src.run_pipeline train
python -m src.run_pipeline infer --input data/raw/HealthConnect_Appointment_Data.csv --output data/processed/predictions.csv
```

Details: see [`WEEK7/README.md`](./WEEK7/README.md).

---

## 📤 Input → Output contract (validated in Week 7)

| **Inputs** | **Outputs** |
|------------|-------------|
| Patient demographics & history | `appointment_id` |
| Appointment details & timing | `no_show_probability` (0–1) |
| Reminder information | `no_show_risk_label` (Low / Medium / High) |
| Distance & waiting time | `predicted_class`, `prediction_timestamp`, `model_version` |

---

## 🔗 Week 7 highlights

- **22 automated tests** – all passed (config, validation, features, prediction schema, invalid inputs, integration)
- Formal **test plan** + **testing record** with pass/fail and retests
- **Cross-track testing** with Data Science (model ↔ pipeline interface)
- Invalid inputs rejected with clear errors (no silent failure)
- Pipeline version **7.0.0** – ready for Week 8 final integration
- Residual model accuracy (~61%) documented as Data Science ownership

---

## 📊 Data resources

| Resource | Description |
|----------|-------------|
| `HealthConnect_Appointment_Data.csv` | Fictional & anonymised appointment records |
| `HealthConnect_Data_Dictionary` | Variable definitions and notes |

**Target:** binary (No-Show incl. Cancelled = 0, Attended = 1)  
**Key predictors:** `booking_lead_days`, distance, patient history, reminders, temporal features

---

## ⚠️ Assumptions, limitations & risks

**Assumptions**
- Dataset is sufficiently representative for a prototype
- Historical features are available at prediction time
- Cancelled appointments merged into No-Show

**Limitations (current)**
- Model predictive performance still modest (~61% accuracy) — owned by Data Science
- Not a production deployment (Week 7 = testing & reliability)
- Latency / load testing optional for Week 8

**Main risks**

| Risk | Mitigation |
|------|------------|
| Temporal data leakage | Strict chronological train/test split |
| Invalid inputs | Explicit validation + automated negative tests |
| Over-reliance on predictions | Documented limitations + human-in-the-loop |

---

## 🛠️ Tech stack

- **Language:** Python 3.10+
- **Data / ML:** pandas, numpy, scikit-learn
- **Config:** YAML
- **Tests:** pytest
- **Tracking:** Git + GitHub

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

*Last updated: Week 7 – Testing, Refinement & End-to-End Validation*
