# Week 7 – Cross-Track Testing & Refinement Evidence

**Track:** Machine Learning Engineering  
**Mandatory HC-POD activity**

---

## 1. Track collaborated with
**Data Science** (primary) — model artefacts and prediction contract.

## 2. Project dependency
Pipeline consumes the candidate classification model and preprocessor defined with Data Science (Week 5 baseline, Week 6 integration).

## 3. Component / output being tested
- Model load path inside `HealthConnectPipeline`
- Prediction output schema (`no_show_probability`, `no_show_risk_label`, `model_version`)
- Compatibility of engineered features with model input

## 4. Information / output received
- `decision_tree.pkl` / related artefacts  
- Target definition (Cancelled merged into No-Show)  
- Feature importance signals used in engineering (`booking_lead_days`, etc.)

## 5. Information / output provided
- Automated test results for pipeline ↔ model interface  
- Confirmation that invalid inputs are rejected before scoring  
- List of remaining open issues (model accuracy owned by DS)

## 6. Testing activity completed
1. Defined test cases TC-12 to TC-17 covering prediction schema and invalid inputs.  
2. Executed integration validation (`validate_integration`).  
3. Documented pass/fail in Week 7 Testing Record.

## 7. Issue or finding identified
- Pipeline integration is stable.  
- Residual performance (~61% accuracy) is a **modelling** limitation, not a pipeline failure.  
- Clear ownership: MLE = reliability & I/O; DS = predictive performance.

## 8. Refinement / action taken
- Expanded negative-path tests so failures surface early.  
- Bumped pipeline version to 7.0.0 and documented remaining issues for Week 8.  
- No forced retrain of the model in MLE track (respects track boundaries).

## 9. Retest result
Integration and schema tests **pass**. Open model-performance item remains tracked for DS/Week 8.

## 10. What changed as a result
- Stronger evidence that the integrated component is ready for multi-track assembly.  
- Explicit hand-off note for Data Science on accuracy.  
- Reduced risk of “it works in my notebook” failures at final integration.

## 11. Evidence
- `docs/Week7_Test_Plan.md`  
- `docs/Week7_Testing_Record.md`  
- `tests/test_pipeline.py`  
- `configs/config.yaml` (v7.0.0)  
- This document

## 12. How this improved the overall HealthConnect solution
A tested, reliable scoring pipeline can be safely consumed by Analytics dashboards, GenAI assistant context, and operations workflows without silent schema or validation failures.

---

**Pattern followed:** Test → Finding → Action → Retest → Validated improvement (pipeline reliability validated; model accuracy deferred with ownership clarity).
