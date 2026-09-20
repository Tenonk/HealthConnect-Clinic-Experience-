# HealthConnect Clinic – ML Pipeline Testing & Reliability

**AnalystLab Africa Experience Lab | Week 7**  
**Track:** Machine Learning Engineering  
**Phase:** Testing → Refinement → End-to-End Validation

---

## Project overview

HealthConnect Clinic aims to reduce patient no-shows using data and AI.  
This repository contains the **tested and refined ML pipeline** built from the Week 6 integrated system.

**Central question:**  
> How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

---

## Week 6 → Week 7 transition

| Item | Content |
|------|---------|
| Main Week 6 output | Integrated pipeline (`src/`, config, CLI, basic validation) |
| Key component | Raw → features → model → risk labels |
| Main limitation | Limited automated tests; weak invalid-input coverage |
| What we test | Data workflow, preprocessing, model I/O, invalid inputs, reproducibility |
| Tracks involved | Data Science (model contract) |
| Goal | Reliability evidence + readiness for Week 8 |

---

## Repository structure

```text
WEEK7/
├── configs/config.yaml          # v7.0.0
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── Week7_Test_Plan.md
│   ├── Week7_Testing_Record.md
│   ├── Week7_CrossTrack_Evidence.md
│   └── Week7_Project_Summary.md
├── models/                      # Transformer + candidate models
├── src/                         # Production pipeline modules
├── tests/test_pipeline.py       # Expanded Week 7 suite
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
cd WEEK7
pip install -r requirements.txt
pip install pytest

# Run automated tests
pytest tests/ -v

# Pipeline CLI (from Week 6, still valid)
python -m src.run_pipeline validate
python -m src.run_pipeline train
python -m src.run_pipeline infer --input data/raw/HealthConnect_Appointment_Data.csv --output data/processed/predictions.csv
```

---

## Week 7 deliverable package

| Artefact | Description |
|----------|-------------|
| Integrated pipeline | `src/` (validation, features, preprocessing, train, predict, pipeline) |
| Test plan | `docs/Week7_Test_Plan.md` |
| Test results / record | `docs/Week7_Testing_Record.md` |
| Cross-track evidence | `docs/Week7_CrossTrack_Evidence.md` |
| Project summary | `docs/Week7_Project_Summary.md` |
| Automated tests | `tests/test_pipeline.py` |
| Config | `configs/config.yaml` (v7.0.0) |

---

## Input → Output contract (unchanged, validated)

| Inputs | Outputs |
|--------|---------|
| Demographics, history, appointment & reminder fields | `appointment_id` |
| | `no_show_probability` (0–1) |
| | `no_show_risk_label` (Low / Medium / High) |
| | `prediction_timestamp`, `model_version` |

---

## Test summary

- **18 test cases** covering config, raw validation, feature engineering, processed features, prediction schema, integration, invalid inputs, reproducibility  
- Invalid inputs (empty data, missing columns, negative lead days, out-of-range probabilities) **must raise errors** — no silent failure  
- Cross-track testing with **Data Science** completed and documented  

---

## Remaining issues (for Week 8)

| Issue | Owner | Notes |
|-------|--------|------|
| Model accuracy ~61% | Data Science | Pipeline is not the bottleneck |
| Latency / load tests | MLE (optional) | Not blocking |
| Final multi-track wiring | PM + all tracks | Week 8 |

---

## Author

**Machine Learning Engineering**  
Tenon KONE  
AnalystLab Africa Experience Lab – Week 7

---

*Last updated: Week 7 – Testing, Refinement & End-to-End Validation*
