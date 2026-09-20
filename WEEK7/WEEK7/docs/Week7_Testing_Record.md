# HealthConnect ML Pipeline – Week 7 Testing Record

**Track:** Machine Learning Engineering  
**Date:** September 2026

---

## Testing record (summary table)

| Component Tested | Testing Objective | Test / Scenario | Expected Result | Actual Result | Pass/Fail | Issue Identified | Action Taken | Retest Result | Collaborating Track | Evidence |
|------------------|-------------------|-----------------|-----------------|---------------|-----------|------------------|--------------|---------------|---------------------|----------|
| Config loader | Config loads correctly | TC-01 Load YAML | Keys present | Keys present | **Pass** | — | — | — | — | `configs/config.yaml`, pytest |
| Raw data validation | Reject invalid datasets | TC-03 Empty DF | DataValidationError | Raised | **Pass** | — | — | — | — | `tests/test_pipeline.py` |
| Raw data validation | Reject missing columns | TC-04 Drop gender | DataValidationError | Raised | **Pass** | — | — | — | — | pytest |
| Raw data validation | Reject negative lead days | TC-05 | DataValidationError | Raised | **Pass** | — | — | — | — | pytest |
| Feature engineering | Binary target mapping | TC-08 | Values in {0,1} | Mapped correctly | **Pass** | Week 6 StringDtype edge case | Fixed mapping logic (pandas string dtype) | Pass | Data Science (target definition) | `feature_engineering.py` |
| Prediction schema | Probabilities in [0,1] | TC-13 | Fail on 1.5 | Raised | **Pass** | — | — | — | — | pytest |
| Pipeline integration | Artefacts + smoke checks | TC-15 `validate_integration` | Dict of checks | Dict returned | **Pass** | Partial if pickle version mismatch | Pipeline can rebuild transformer | Pass | Data Science | `pipeline.py` |
| Inference – empty input | No silent failure | TC-16 | Exception | Exception raised | **Pass** | Week 6: weak coverage of invalid paths | Expanded negative tests | Pass | — | pytest |
| Inference – missing dates | Schema integrity | TC-17 | Failure | Failure observed | **Pass** | — | Documented | — | — | pytest |
| Reproducibility | Documented setup | TC-18 | Files exist | Exist | **Pass** | — | — | — | — | repo structure |

---

## Issues identified & refinements

| ID | Issue | Severity | Status | Resolution |
|----|-------|----------|--------|------------|
| W7-01 | Limited automated test coverage in Week 6 | High | **Resolved** | Expanded `tests/test_pipeline.py` (18 scenarios: happy path + invalid inputs) |
| W7-02 | Target mapping fragile with pandas StringDtype | Medium | **Resolved** | Robust map + `pd.to_numeric` in `feature_engineering.py` (from Week 6 fix, re-validated) |
| W7-03 | Invalid inputs could fail unclearly | Medium | **Improved** | Explicit validation tests for empty DF, missing columns, negative lead days, bad probabilities |
| W7-04 | sklearn pickle version sensitivity | Medium | **Mitigated** | Pipeline can rebuild preprocessor from raw data + config |
| W7-05 | Model accuracy still ~61% | Medium | **Open** | Not an MLE defect; deferred to Data Science / Week 8 collaboration |
| W7-06 | No latency / load tests | Low | **Open** | Documented for Week 8 if needed |

---

## Cross-track testing activity

| Field | Content |
|-------|---------|
| Track collaborated with | **Data Science** |
| Project dependency | Candidate model + ColumnTransformer from DS/Week 5–6 |
| Component tested | Preprocessing ↔ model interface inside integrated pipeline |
| Information received | Model artefacts, feature list, target definition (Cancelled→No-Show) |
| Information provided | Pipeline test results, I/O contract confirmation, remaining technical issues |
| Testing activity | Ran integration validation + unit tests on model load and prediction schema |
| Finding | Artefacts load; output schema stable; performance limit is model-side not pipeline-side |
| Action | Documented that accuracy improvement is DS ownership; pipeline remains compatible |
| Retest | Integration checks pass after documentation/config version bump |
| What changed | Clear separation of pipeline reliability (MLE) vs model performance (DS) |
| Evidence | This record + `tests/test_pipeline.py` + `Week7_CrossTrack_Evidence.md` |
| Impact on HealthConnect | Reduces risk of silent integration failures before Week 8 final assembly |

---

## End-to-end validation answers

1. **Component tested:** Integrated no-show prediction pipeline (validation → features → model → risk labels).  
2. **Expected:** Valid inputs produce probability + risk label; invalid inputs raise clear errors.  
3. **Actual:** Matches expectations for defined test cases.  
4. **Weakness:** Model predictive power still modest; not a pipeline defect.  
5. **Changed:** Expanded tests, hardened invalid-input paths, version bump to 7.0.0.  
6. **Retest:** Automated suite covers critical paths.  
7. **Other track:** Data Science (model contract).  
8. **Contribution:** Reliable scoring service for dashboards / GenAI / operations.  
9. **Still needed for Week 8:** Optional latency profile; coordinate final model choice with DS; wire to PM integration checklist.

---

*HealthConnect ML Engineering – Week 7 Testing Record*
