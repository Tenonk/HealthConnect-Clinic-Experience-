"""
Feature engineering for HealthConnect no-show prediction.
Reproducible transformations matching Week 5 notebook logic.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("healthconnect.features")


def engineer_features(
    df: pd.DataFrame,
    lead_bins: List[float] | None = None,
    lead_labels: List[str] | None = None,
) -> pd.DataFrame:
    """
    Apply feature engineering steps:
    - Parse dates
    - Create lead time category
    - Extract month / day names
    - Map target (if present)
    """
    data = df.copy()
    logger.info("Starting feature engineering on %d rows", len(data))

    # Parse dates
    for col in ["booking_date", "appointment_date"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")

    # Target mapping (binary: 0 = No-Show incl. Cancelled, 1 = Attended)
    if "appointment_outcome" in data.columns:
        mapping = {"No-Show": 0, "Attended": 1, "Cancelled": 0}
        # Always try map first (works for string / object)
        mapped = data["appointment_outcome"].map(mapping)
        if mapped.isna().all():
            # Already numeric – merge Cancelled (2) into No-Show (0)
            data["appointment_outcome"] = data["appointment_outcome"].replace(2, 0)
        else:
            data["appointment_outcome"] = mapped
        data["appointment_outcome"] = pd.to_numeric(data["appointment_outcome"], errors="coerce").astype("Int64")

    # Lead time category
    if "booking_lead_days" in data.columns:
        bins = lead_bins or [0, 7, 30, 90, np.inf]
        labels = lead_labels or [
            "Last minute (0-7d)",
            "Short term (8-30d)",
            "Medium term (31-90d)",
            "Long term (90d+)",
        ]
        data["lead_cat_temp"] = pd.cut(
            data["booking_lead_days"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

    # Temporal features from dates
    if "booking_date" in data.columns:
        data["booking_month"] = data["booking_date"].dt.month_name()
        data["booking_day"] = data["booking_date"].dt.day_name()
    if "appointment_date" in data.columns:
        data["appointment_month"] = data["appointment_date"].dt.month_name()
        # Overwrite appointment_day with consistent day_name if needed
        data["appointment_day"] = data["appointment_date"].dt.day_name()

    logger.info(
        "Feature engineering complete. New columns: lead_cat_temp, booking_month, "
        "appointment_month, booking_day"
    )
    return data


def select_features_and_target(
    data: pd.DataFrame,
    id_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series | None]:
    """
    Drop IDs and separate X / y.
    Returns (X, y) where y may be None if target not present.
    """
    id_columns = id_columns or ["appointment_id", "patient_id"]
    drop_cols = [c for c in id_columns if c in data.columns]

    y = None
    if "appointment_outcome" in data.columns:
        y = data["appointment_outcome"].copy()
        drop_cols.append("appointment_outcome")

    X = data.drop(columns=drop_cols, errors="ignore")
    logger.info("Selected features: %d columns (dropped %s)", X.shape[1], drop_cols)
    return X, y


def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    date_col: str = "appointment_date",
    ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Strict chronological split on appointment_date.
    First `ratio` portion = train, remainder = test.
    """
    if date_col not in X.columns:
        raise ValueError(f"Date column '{date_col}' not found for temporal split")

    # Ensure sorted
    order = X[date_col].argsort()
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)

    split_idx = int(len(X_sorted) * ratio)
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]

    logger.info(
        "Temporal split: train=%d (%s → %s), test=%d (%s → %s)",
        len(X_train),
        X_train[date_col].min(),
        X_train[date_col].max(),
        len(X_test),
        X_test[date_col].min(),
        X_test[date_col].max(),
    )
    return X_train, X_test, y_train, y_test
