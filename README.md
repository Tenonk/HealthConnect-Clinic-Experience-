# HealthConnect Clinic – Machine Learning Engineering

**AnalystLab Africa Experience Lab | Week 4**  
**Track:** Machine Learning Engineering  
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data and AI

---

## 📌 Project Overview

HealthConnect Clinic is a fictional outpatient healthcare provider facing a high rate of **patient no-shows**. Missed appointments lead to wasted clinical capacity, longer waiting times, and increased administrative overhead.

**Central question:**  
> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

As a **Machine Learning Engineer**, my role is to design a **reliable, reproducible, and production-oriented ML system** that predicts the likelihood of a patient missing a scheduled appointment (no-show prediction).

> ⚠️ **Week 4 focus:** Problem understanding, system design, architecture, and planning only.  
> No final model training or deployment is expected at this stage.

---

## 🎯 Week 4 Objectives

- [x] Understand the HealthConnect business problem
- [x] Review the appointment dataset and data dictionary
- [x] Define the ML problem and target variable
- [x] Specify system inputs and outputs
- [x] Design high-level system architecture
- [x] Define the end-to-end ML workflow
- [x] Identify key dependencies, assumptions, limitations, and risks
- [x] Produce a professional ML System Design Document + diagrams
- [x] Prepare a clear foundation for Week 5 development

---

## 📁 Repository Structure

```text
healthconnect-ml-engineering/
├── data/
│   ├── raw/                          # Original immutable dataset
│   └── processed/                    # Cleaned / feature-engineered data (Week 5+)
├── notebooks/
│   ├── 01_data_exploration.ipynb     # (Week 5)
│   ├── 02_feature_engineering.ipynb  # (Week 5)
│   └── 03_model_prototyping.ipynb    # (Week 5)
├── src/
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── models/                           # Versioned model artefacts (Week 5+)
├── configs/                          # YAML / JSON configs
├── docs/
│   ├── HealthConnect_ML_System_Design_Document_Week4.pdf
│   ├── Fiche_Pedagogique_HealthConnect_ML_Engineering_Week4.pdf
│   ├── ML_System_Architecture_Diagram.png
│   ├── ML_Workflow_Design_Diagram.png
│   └── ML_Input_Output_Contract.png
├── requirements.txt
└── README.md
```

---

## 📊 Data Resources

| Resource | Description |
|----------|-------------|
| `HealthConnect_Appointment_Data.csv` | Fictional & anonymised appointment records |
| `HealthConnect_Data_Dictionary` | Variable definitions and notes |

**Key variables for no-show prediction:**
- `previous_no_shows`, `previous_appointments`
- `booking_lead_days`
- `reminder_sent` / `reminder_channel`
- `distance_to_clinic_km`, `waiting_time_minutes`
- `appointment_type`, `appointment_day`, `appointment_time`
- `age` / `age_group`, `gender`
- **Target:** `appointment_outcome` → `Attended` | `No-Show` | `Cancelled`

---

## 🏗️ System Design (Week 4 Deliverable)

### High-Level Architecture

```text
Data Sources
    ↓
Data Ingestion & Validation
    ↓
Feature Engineering & Preprocessing
    ↓
Model Training Pipeline
    ↓
Model Registry & Versioning
    ↓
Inference / Scoring Service (batch)
    ↓
Downstream Consumers (dashboards, alerts, GenAI)
```

### Input → Output Contract

| **Inputs** | **Outputs** |
|------------|-------------|
| patient demographics & history | `appointment_id` |
| appointment details & timing | `no_show_probability` (0–1) |
| reminder information | `no_show_risk_label` (Low / Medium / High) |
| distance & waiting time | `prediction_timestamp` |
| | `model_version` |

### Proposed ML Workflow (10 stages)

1. Data Acquisition & Understanding  
2. Data Quality Assessment  
3. Target Definition & Handling of Cancellations  
4. Feature Engineering  
5. Temporal Train / Validation / Test Split  
6. Model Development (baseline + stronger candidates)  
7. Evaluation & Selection  
8. Packaging & Versioning  
9. Inference Pipeline  
10. Monitoring & Retraining Strategy  

---

## 📄 Week 4 Deliverables

| File | Description |
|------|-------------|
| [ML System Design Document](docs/HealthConnect_ML_System_Design_Document_Week4.pdf) | Full design document (architecture, workflow, risks, summary) |
| [Architecture Diagram](docs/ML_System_Architecture_Diagram.png) | High-level system architecture |
| [Workflow Diagram](docs/ML_Workflow_Design_Diagram.png) | 10-stage ML pipeline |
| [Input/Output Contract](docs/ML_Input_Output_Contract.png) | Prediction system interface |
| [Fiche Pédagogique (FR)](docs/Fiche_Pedagogique_HealthConnect_ML_Engineering_Week4.pdf) | Pedagogical summary in French |

---

## ⚙️ Tech Stack (planned)

- **Language:** Python 3.10+
- **Data:** pandas, numpy
- **ML:** scikit-learn (baseline), optionally XGBoost / LightGBM
- **Tracking / Versioning:** Git + GitHub, model artefacts + metadata
- **Environment:** `requirements.txt` / conda environment
- **Future:** MLflow, FastAPI (serving)

---

## 🚀 Week 5 Focus (Next Steps)

- Thorough data quality assessment and cleaning  
- Feature engineering pipeline  
- Train and evaluate a strong baseline model  
- Document initial performance metrics and feature importance  
- Refine the system design based on empirical findings  

---

## ⚠️ Key Assumptions & Risks

**Assumptions**
- Dataset is sufficiently representative for a prototype  
- Historical features (`previous_no_shows`, etc.) are available at prediction time  
- Cancelled appointments can be excluded (or treated separately) without major bias  

**Main Risks**
| Risk | Mitigation |
|------|------------|
| Temporal data leakage | Time-based train/test split |
| Class imbalance | Stratified sampling, class weights, PR-AUC |
| Concept drift | Monitoring + periodic retraining |
| Over-reliance on predictions | Clear documentation of limitations + human-in-the-loop |

---

## 📚 References & Resources

- AnalystLab Africa – Experience Lab Week 4 Assignment  
- HealthConnect Appointment Dataset & Data Dictionary  
- HealthConnect Clinic Knowledge Base (primarily for GenAI track)

---

## 👤 Author

**Machine Learning Engineering Junior  **  
Tenon KONE  

---

## 📝 License & Notes

This project is part of the **AnalystLab Africa Experience Lab** internship programme.  
All data is **fictional and anonymised**. Do not treat results as real-world clinical recommendations.

---

*Last updated: Week 4 – Problem Understanding & System Design*
