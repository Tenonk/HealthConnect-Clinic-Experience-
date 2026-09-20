"""
Model training utilities for HealthConnect.
Supports training baseline tree models and saving artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger("healthconnect.train")


MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> Any:
    """Instantiate and fit a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY)}")

    cls = MODEL_REGISTRY[model_name]
    model = cls(**kwargs)
    logger.info("Training %s on %d samples, %d features", model_name, len(X_train), X_train.shape[1])
    model.fit(X_train, y_train)
    logger.info("Training complete for %s", model_name)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception as e:
            logger.warning("Could not compute ROC-AUC: %s", e)
            metrics["roc_auc"] = float("nan")
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def save_model(model: Any, path: str | Path) -> None:
    """Persist trained model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str | Path) -> Any:
    """Load trained model."""
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
