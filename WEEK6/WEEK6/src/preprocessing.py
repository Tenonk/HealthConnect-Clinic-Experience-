"""
Preprocessing pipeline builder for HealthConnect.
Builds a reproducible ColumnTransformer matching Week 5 logic.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("healthconnect.preprocessing")


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str],
) -> ColumnTransformer:
    """
    Create ColumnTransformer:
    - Categorical: most_frequent imputer + OneHotEncoder (drop first, handle unknown)
    - Numeric: median imputer + StandardScaler
    - Remainder (e.g. previous_no_shows): passthrough
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_features),
            ("num", num_pipeline, numeric_features),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    logger.info(
        "Built preprocessor: %d cat, %d num features",
        len(categorical_features),
        len(numeric_features),
    )
    return transformer


def drop_date_columns(
    X: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove raw date columns before transformation."""
    date_cols = date_cols or ["booking_date", "appointment_date"]
    cols_to_drop = [c for c in date_cols if c in X.columns]
    if cols_to_drop:
        logger.info("Dropping date columns: %s", cols_to_drop)
        return X.drop(columns=cols_to_drop)
    return X


def fit_transform_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer, List[str]]:
    """
    Fit preprocessor on train, transform train & test.
    Returns transformed DataFrames, fitted transformer, and feature names.
    """
    X_train = drop_date_columns(X_train)
    X_test = drop_date_columns(X_test)

    # Ensure feature lists only contain existing columns
    cat_feats = [c for c in categorical_features if c in X_train.columns]
    num_feats = [c for c in numeric_features if c in X_train.columns]

    transformer = build_preprocessor(cat_feats, num_feats)
    X_train_t = transformer.fit_transform(X_train)
    X_test_t = transformer.transform(X_test)

    # Reconstruct column names
    try:
        feature_names = list(transformer.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn
        encoded = (
            transformer.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(cat_feats)
        )
        passthrough = [
            c
            for c in X_train.columns
            if c not in cat_feats and c not in num_feats
        ]
        feature_names = list(encoded) + list(num_feats) + passthrough

    X_train_df = pd.DataFrame(X_train_t, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_t, columns=feature_names)

    logger.info(
        "Transformed features: train=%s, test=%s, n_features=%d",
        X_train_df.shape,
        X_test_df.shape,
        len(feature_names),
    )
    return X_train_df, X_test_df, transformer, feature_names


def save_preprocessor(transformer: ColumnTransformer, path: str) -> None:
    """Persist fitted ColumnTransformer."""
    joblib.dump(transformer, path)
    logger.info("Saved preprocessor to %s", path)


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load fitted ColumnTransformer."""
    transformer = joblib.load(path)
    logger.info("Loaded preprocessor from %s", path)
    return transformer
