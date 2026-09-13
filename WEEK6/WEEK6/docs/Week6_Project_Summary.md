# Week 6 Project Summary – HealthConnect ML Engineering

**Intern:** Tenon KONE  
**Track:** Machine Learning Engineering  
**Date:** September 2026

---

## 1. What was planned
Transform the Week 5 notebook-based baseline (EDA + preprocessing + Decision Tree / Random Forest) into an integrated, validated, reproducible ML pipeline ready for broader testing in Week 7.

## 2. What was completed
- Full modular `src/` package (validation, feature engineering, preprocessing, train, predict, orchestration)
- Central YAML configuration
- CLI entry point (`python -m src.run_pipeline`)
- Input/output validation and logging
- Integration validation suite
- Updated repository structure and professional README
- Cross-track integration documentation

## 3. What was improved from Week 5
| Week 5 | Week 6 |
|--------|--------|
| Logic only in notebooks | Reusable Python package |
| No automated validation | Explicit input/output checks |
| Manual artefact loading | Config-driven + CLI |
| No logging | Structured logging |
| Hard-coded paths | Path helpers + config |
| No end-to-end test | `validate` command + pytest skeleton |

## 4. What was integrated
- Data Science model artefacts (`decision_tree.pkl`, `column_transformer.pkl`)
- Feature engineering logic derived from Analytics / EDA insights (`booking_lead_days` categories, temporal features)

## 5. Tracks collaborated with
- **Data Science** (primary) – models and evaluation metrics
- **Data Analytics** (secondary) – feature importance signals used to design engineering steps

## 6. What was exchanged
**Received:** trained models, preprocessor, performance tables, key predictive variables.  
**Provided:** pipeline interface, prediction schema (probability + risk label), validation contract, reproducible training/inference entry points.

## 7. What changed as a result
The no-show prediction capability is no longer trapped inside notebooks. It can be invoked, tested and handed to other tracks (dashboards, GenAI assistant, operations) with a clear contract.

## 8. Key findings / development outcomes
- Pipeline successfully rebuilds 57-feature matrix matching Week 5.
- Decision Tree remains the recommended candidate for operational binary classification.
- Integration gaps (missing `src/`, empty tests, no config) have been closed.

## 9. Major challenges
- sklearn version mismatch when unpickling Week 5 artefacts → solved by making the pipeline able to rebuild the transformer.
- Target mapping edge cases with pandas StringDtype.
- Network constraints during dependency installation in the sandbox.

## 10. Important decisions
- Keep Decision Tree as default model (best Accuracy/F1 balance from Week 5).
- Merge Cancelled into No-Show (consistent with Week 5 business decision).
- Use risk labels (Low/Medium/High) in addition to raw probability for downstream consumers.
- Strict temporal split retained.

## 11. Remaining issues
- Model performance still modest (~61 % accuracy) – requires hyper-parameter tuning / better features (Week 7 / DS).
- Full pytest coverage still light.
- Monitoring / concept-drift detection not yet implemented.

## 12. Contribution to overall HealthConnect project
Provides a reliable, validated prediction service that other tracks can consume. Reduces the risk that model work remains isolated and unusable for operational or GenAI use-cases.

## 13. Proposed focus for Week 7
1. Expand automated tests (unit + integration).  
2. Threshold / calibration analysis and possible model re-training collaboration with DS.  
3. Latency & resource profiling.  
4. Interface hand-off to Analytics dashboard and GenAI assistant.  
5. Document failure modes and escalation paths.
