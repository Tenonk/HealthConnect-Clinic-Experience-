# HealthConnect ML Pipeline – Week 7 Test Plan

**Track:** Machine Learning Engineering  
**Phase:** Testing → Refinement → End-to-End Validation  
**Pipeline version:** 7.0.0

---

## 1. Week 6 → Week 7 Transition

| # | Item | Content |
|---|------|---------|
| 1 | Main Week 6 output | Integrated ML pipeline package (`src/`, config, CLI, basic validation) |
| 2 | Most important component | End-to-end pipeline: raw data → features → model → risk labels |
| 3 | Main limitation | Limited automated tests; error handling incomplete for invalid inputs; modest model performance (~61% acc) |
| 4 | Component requiring testing | Full pipeline (data validation, preprocessing, model integration, inference outputs) |
| 5 | Relevant tracks | Data Science (model artefacts), Data Analytics (feature insights) |
| 6 | What we test | Data workflow, preprocessing, model I/O compatibility, invalid inputs, reproducibility |
| 7 | What we improve / validate | Reliability, error handling, test coverage, readiness for Week 8 |

---

## 2. Testing objectives

1. Confirm the pipeline correctly processes valid inputs and produces the expected schema.
2. Confirm invalid / unexpected inputs are rejected with clear errors (no silent failure).
3. Confirm preprocessing and model components remain compatible.
4. Confirm reproducibility from documented setup (config + artefacts).
5. Identify remaining technical issues before final integration (Week 8).

---

## 3. Test cases

| ID | Category | Scenario | Expected result |
|----|----------|----------|-----------------|
| TC-01 | Config | Load `configs/config.yaml` | Config dict with project, data, model, validation keys |
| TC-02 | Raw validation | Valid sample (≥10 rows, required columns) | Pass |
| TC-03 | Raw validation | Empty DataFrame | Fail (`DataValidationError`) |
| TC-04 | Raw validation | Missing required column (`gender`) | Fail |
| TC-05 | Raw validation | Negative `booking_lead_days` | Fail |
| TC-06 | Raw validation | Too few rows (< min_rows) | Fail |
| TC-07 | Feature eng. | Engineer features on sample | `lead_cat_temp`, month/day columns present |
| TC-08 | Feature eng. | Target mapping | Binary {0, 1} only |
| TC-09 | Feature eng. | Select features | IDs dropped from X |
| TC-10 | Processed | Empty feature matrix | Fail |
| TC-11 | Processed | NaN in features | Fail |
| TC-12 | Prediction | Valid prediction DataFrame | Pass |
| TC-13 | Prediction | Probability outside [0, 1] | Fail |
| TC-14 | Prediction | Missing required output column | Fail |
| TC-15 | Integration | `validate_integration()` | Returns dict of checks |
| TC-16 | Invalid input | Empty DF to inference | Exception raised |
| TC-17 | Invalid input | Missing date columns | Exception / validation failure |
| TC-18 | Reproducibility | Raw data + config files exist | Pass |

---

## 4. How to run tests

```bash
cd WEEK7
pip install -r requirements.txt
pip install pytest
pytest tests/ -v
```

---

## 5. Scope boundaries

- **In scope:** Pipeline reliability, I/O validation, preprocessing ↔ model interaction, error handling, reproducibility.
- **Out of scope (Week 7):** Hyperparameter tuning / new model training (Data Science), production deployment, dashboard UI testing.

---

*HealthConnect ML Engineering – Week 7*
