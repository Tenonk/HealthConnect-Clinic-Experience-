"""
Inference / prediction module for HealthConnect.
Produces probability + risk label outputs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger("healthconnect.predict")


def risk_label_from_proba(proba: float, thresholds: tuple = (0.35, 0.55)) -> str:
    """
    Map probability of No-Show (class 0) or of Attended?
    Convention: model predicts class 1 = Attended.
    We want no_show_probability = P(No-Show) = 1 - P(Attended).
    """
    low, high = thresholds
    if proba >= high:
        return "High"
    if proba >= low:
        return "Medium"
    return "Low"


def predict_batch(
    model: Any,
    X: pd.DataFrame,
    appointment_ids: Optional[List[str] | pd.Series] = None,
    model_version: str = "6.0.0",
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Run batch inference.
    Returns DataFrame with:
      - appointment_id (if provided)
      - no_show_probability
      - predicted_class (0/1)
      - no_show_risk_label
      - prediction_timestamp
      - model_version
    """
    logger.info("Running batch prediction on %d samples", len(X))

    if hasattr(model, "predict_proba"):
        # Assume positive class = Attended (1)
        proba_attended = model.predict_proba(X)[:, 1]
        no_show_proba = 1.0 - proba_attended
    else:
        preds = model.predict(X)
        no_show_proba = 1.0 - preds.astype(float)
        proba_attended = preds.astype(float)

    predicted_class = (proba_attended >= decision_threshold).astype(int)

    risk_labels = [risk_label_from_proba(p) for p in no_show_proba]

    result = pd.DataFrame(
        {
            "no_show_probability": no_show_proba,
            "predicted_class": predicted_class,
            "no_show_risk_label": risk_labels,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
        }
    )

    if appointment_ids is not None:
        result.insert(0, "appointment_id", list(appointment_ids))

    logger.info(
        "Predictions complete. Risk distribution: %s",
        result["no_show_risk_label"].value_counts().to_dict(),
    )
    return result
