"""
Data validation module for HealthConnect pipeline.
Validates raw and intermediate inputs/outputs.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("healthconnect.validation")


class DataValidationError(Exception):
    """Raised when validation fails."""


def validate_raw_data(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 10,
    max_missing_ratio: float = 0.3,
) -> None:
    """
    Validate raw appointment dataset.
    Raises DataValidationError on critical issues.
    """
    logger.info("Validating raw data shape=%s", df.shape)

    if df.empty or len(df) < min_rows:
        raise DataValidationError(
            f"Dataset has too few rows: {len(df)} (min required: {min_rows})"
        )

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # Check missing ratio on key columns
    for col in required_columns:
        if col in df.columns:
            ratio = df[col].isna().mean()
            if ratio > max_missing_ratio:
                raise DataValidationError(
                    f"Column '{col}' has {ratio:.1%} missing values "
                    f"(max allowed: {max_missing_ratio:.1%})"
                )

    # Basic type / value checks
    if "appointment_outcome" in df.columns:
        valid_outcomes = {"Attended", "No-Show", "Cancelled"}
        unique = set(df["appointment_outcome"].dropna().unique())
        invalid = unique - valid_outcomes
        if invalid:
            logger.warning("Unexpected outcome values: %s", invalid)

    if "booking_lead_days" in df.columns:
        if (df["booking_lead_days"] < 0).any():
            raise DataValidationError("Negative booking_lead_days detected")

    logger.info("Raw data validation passed")


def validate_processed_features(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    expected_n_features: Optional[int] = None,
) -> None:
    """Validate processed feature matrix and optional target."""
    logger.info("Validating processed features shape=%s", X.shape)

    if X.empty:
        raise DataValidationError("Processed feature matrix is empty")

    if X.isna().any().any():
        n_na = X.isna().sum().sum()
        raise DataValidationError(f"Processed features contain {n_na} NaN values")

    if expected_n_features is not None and X.shape[1] != expected_n_features:
        raise DataValidationError(
            f"Expected {expected_n_features} features, got {X.shape[1]}"
        )

    if y is not None:
        if len(y) != len(X):
            raise DataValidationError(
                f"X and y length mismatch: {len(X)} vs {len(y)}"
            )
        if y.isna().any():
            raise DataValidationError("Target contains NaN values")
        unique = set(y.unique())
        if not unique.issubset({0, 1}):
            logger.warning("Target contains unexpected values: %s", unique)

    logger.info("Processed features validation passed")


def validate_prediction_output(
    predictions: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
) -> None:
    """Validate prediction DataFrame structure and value ranges."""
    if required_cols is None:
        required_cols = [
            "appointment_id",
            "no_show_probability",
            "no_show_risk_label",
            "model_version",
        ]

    missing = [c for c in required_cols if c not in predictions.columns]
    if missing:
        raise DataValidationError(f"Prediction output missing columns: {missing}")

    if "no_show_probability" in predictions.columns:
        probs = predictions["no_show_probability"]
        if not ((probs >= 0) & (probs <= 1)).all():
            raise DataValidationError("Probabilities outside [0, 1] range")

    logger.info("Prediction output validation passed (n=%d)", len(predictions))
