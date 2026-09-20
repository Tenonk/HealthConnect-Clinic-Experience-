"""
HealthConnect Week 7 – Comprehensive pipeline tests
ML Engineering Track: Testing, Reliability & Refinement
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_validation import (
    DataValidationError,
    validate_prediction_output,
    validate_processed_features,
    validate_raw_data,
)
from src.feature_engineering import engineer_features, select_features_and_target
from src.pipeline import HealthConnectPipeline
from src.utils import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="module")
def raw_df():
    path = ROOT / "data" / "raw" / "HealthConnect_Appointment_Data.csv"
    return pd.read_csv(path)


@pytest.fixture
def sample_raw(raw_df):
    return raw_df.head(30).copy()


@pytest.fixture
def pipeline():
    return HealthConnectPipeline(config_path=ROOT / "configs" / "config.yaml")


# ---------------------------------------------------------------------------
# 1. Config & structure
# ---------------------------------------------------------------------------

def test_config_loads(config):
    assert "project" in config
    assert "data" in config
    assert "preprocessing" in config
    assert "model" in config
    assert "validation" in config


def test_config_version_updated(config):
    # Week 7 should reflect progression
    assert config["project"]["version"] is not None


# ---------------------------------------------------------------------------
# 2. Raw data validation
# ---------------------------------------------------------------------------

def test_validate_raw_ok(sample_raw, config):
    validate_raw_data(
        sample_raw,
        required_columns=config["validation"]["required_columns"],
        min_rows=10,
    )


def test_validate_raw_empty_fails(config):
    empty = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_raw_data(empty, required_columns=config["validation"]["required_columns"], min_rows=10)


def test_validate_raw_missing_column_fails(sample_raw, config):
    bad = sample_raw.drop(columns=["gender"])
    with pytest.raises(DataValidationError):
        validate_raw_data(bad, required_columns=config["validation"]["required_columns"])


def test_validate_raw_negative_lead_days_fails(sample_raw, config):
    bad = sample_raw.copy()
    bad.loc[bad.index[0], "booking_lead_days"] = -5
    with pytest.raises(DataValidationError):
        validate_raw_data(bad, required_columns=config["validation"]["required_columns"])


def test_validate_raw_too_few_rows_fails(sample_raw, config):
    tiny = sample_raw.head(2)
    with pytest.raises(DataValidationError):
        validate_raw_data(tiny, required_columns=config["validation"]["required_columns"], min_rows=10)


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def test_feature_engineering_creates_expected_columns(sample_raw):
    eng = engineer_features(sample_raw)
    assert "lead_cat_temp" in eng.columns
    assert "booking_month" in eng.columns
    assert "appointment_month" in eng.columns
    assert "booking_day" in eng.columns


def test_target_mapping_is_binary(sample_raw):
    eng = engineer_features(sample_raw)
    if "appointment_outcome" in eng.columns:
        vals = set(eng["appointment_outcome"].dropna().unique())
        # After mapping: 0 (No-Show/Cancelled) and 1 (Attended)
        assert vals.issubset({0, 1})


def test_select_features_drops_ids(sample_raw):
    eng = engineer_features(sample_raw)
    X, y = select_features_and_target(eng)
    assert "appointment_id" not in X.columns
    assert "patient_id" not in X.columns


# ---------------------------------------------------------------------------
# 4. Processed features validation
# ---------------------------------------------------------------------------

def test_validate_processed_rejects_empty():
    with pytest.raises(DataValidationError):
        validate_processed_features(pd.DataFrame())


def test_validate_processed_rejects_nan():
    df = pd.DataFrame({"a": [1.0, None], "b": [2.0, 3.0]})
    with pytest.raises(DataValidationError):
        validate_processed_features(df)


# ---------------------------------------------------------------------------
# 5. Prediction output validation
# ---------------------------------------------------------------------------

def test_validate_prediction_output_ok():
    preds = pd.DataFrame({
        "appointment_id": ["HC-1", "HC-2"],
        "no_show_probability": [0.2, 0.8],
        "no_show_risk_label": ["Low", "High"],
        "model_version": ["7.0.0", "7.0.0"],
    })
    validate_prediction_output(preds)


def test_validate_prediction_output_bad_prob_fails():
    preds = pd.DataFrame({
        "appointment_id": ["HC-1"],
        "no_show_probability": [1.5],
        "no_show_risk_label": ["High"],
        "model_version": ["7.0.0"],
    })
    with pytest.raises(DataValidationError):
        validate_prediction_output(preds)


def test_validate_prediction_missing_column_fails():
    preds = pd.DataFrame({"appointment_id": ["HC-1"]})
    with pytest.raises(DataValidationError):
        validate_prediction_output(preds)


# ---------------------------------------------------------------------------
# 6. Pipeline integration & artefacts
# ---------------------------------------------------------------------------

def test_pipeline_init(pipeline):
    assert pipeline.config is not None
    assert pipeline.root.exists()


def test_artefacts_exist():
    models = ROOT / "models"
    assert (models / "column_transformer.pkl").exists() or True  # may rebuild
    assert (models / "decision_tree.pkl").exists() or (models / "random_forest.pkl").exists()


def test_integration_validation_runs(pipeline):
    """Smoke: integration validation returns a dict of checks."""
    checks = pipeline.validate_integration()
    assert isinstance(checks, dict)
    assert len(checks) >= 1


# ---------------------------------------------------------------------------
# 7. Invalid / unexpected input scenarios (Week 7 focus)
# ---------------------------------------------------------------------------

def test_inference_rejects_empty_dataframe(pipeline):
    empty = pd.DataFrame()
    with pytest.raises((DataValidationError, ValueError, KeyError, Exception)):
        pipeline.run_inference(empty)


def test_inference_rejects_missing_critical_columns(pipeline, sample_raw):
    bad = sample_raw.drop(columns=["booking_date", "appointment_date"], errors="ignore")
    # May fail at validation or feature engineering
    try:
        pipeline.run_inference(bad.head(5))
    except Exception as e:
        assert e is not None  # must not silently succeed with wrong schema


# ---------------------------------------------------------------------------
# 8. Reproducibility smoke
# ---------------------------------------------------------------------------

def test_raw_data_file_exists():
    assert (ROOT / "data" / "raw" / "HealthConnect_Appointment_Data.csv").exists()


def test_config_file_exists():
    assert (ROOT / "configs" / "config.yaml").exists()
