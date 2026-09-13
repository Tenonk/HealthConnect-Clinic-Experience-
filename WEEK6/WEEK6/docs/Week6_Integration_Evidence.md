# Week 6 – Cross-Track Integration Evidence

## Mandatory Integration Record

1. **Track collaborated with:**  
   Data Science (model artefacts) + Data Analytics (feature insights)

2. **Project dependency:**  
   Week 5 baseline classification models and ColumnTransformer; feature engineering decisions driven by EDA findings (especially `booking_lead_days`).

3. **Information / output received:**  
   - `models/decision_tree.pkl`, `models/random_forest.pkl`, `models/column_transformer.pkl`  
   - Performance metrics (Accuracy, F1, ROC-AUC)  
   - Feature importance rankings  
   - Temporal split logic and target definition (Cancelled → No-Show)

4. **Information / output provided:**  
   - Integrated pipeline (`src/pipeline.py`)  
   - Stable inference contract (probability + risk label)  
   - Validation checks and logging  
   - Configuration file and CLI for reproducible runs  
   - Prepared testing plan for Week 7

5. **Integration activity completed:**  
   - Extracted notebook logic into modular, testable Python package  
   - Ensured preprocessor output shape and feature order are compatible with the loaded models  
   - Added end-to-end validation command that exercises the full data → features → model → prediction flow  
   - Documented the hand-off points for other tracks

6. **What changed as a result:**  
   - Pipeline is no longer notebook-only  
   - Other tracks can now consume predictions via a defined schema  
   - Integration gaps identified in Week 5 have been closed  
   - Project is ready for coordinated testing in Week 7

7. **Evidence:**  
   - Source code under `src/`  
   - `configs/config.yaml`  
   - `tests/test_pipeline.py`  
   - Updated `README.md`  
   - This document  
   - Week 6 Project Summary  

---

## Integration Issue Log (Week 6)

| ID | Issue | Severity | Status | Resolution / Mitigation |
|----|-------|----------|--------|-------------------------|
| INT-01 | Week 5 artefacts pickled with different sklearn version | Medium | Mitigated | Pipeline can rebuild transformer from raw data |
| INT-02 | No automated input validation | High | Resolved | `data_validation.py` + checks in pipeline |
| INT-03 | Hard-coded paths and magic numbers | Medium | Resolved | Central `config.yaml` |
| INT-04 | Missing logging | Low | Resolved | Structured logging via `utils.setup_logging` |
| INT-05 | Empty `src/` and `tests/` directories | High | Resolved | Full package + initial tests |
| INT-06 | Model performance still modest | Medium | Open | Deferred to Week 7 / Data Science collaboration |

---

## Decision Log (Week 6)

| Decision | Rationale |
|----------|-----------|
| Keep Decision Tree as default model | Best Accuracy / F1 balance from Week 5; simplest for interpretation |
| Merge Cancelled into No-Show | Consistent with Week 5 business logic; low volume of Cancelled |
| Provide both probability and risk label | Better for operational and GenAI consumers |
| Rebuild preprocessor instead of forcing old pickle | Avoid version lock-in; improve reproducibility |
