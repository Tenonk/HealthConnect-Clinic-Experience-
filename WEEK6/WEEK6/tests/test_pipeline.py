"""
Integration and unit tests for HealthConnect Week 6 pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_validation import DataValidationError, validate_raw_data
from src.feature_engineering import engineer_features
from src.pipeline import HealthConnectPipeline
from src.utils import load_config


@pytest.fixture
def sample_raw():
    path = ROOT / "data" / "raw" / "HealthConnect_Appointment_Data.csv"
    return pd.read_csv(path).head(50)


def test_config_loads():
    cfg = load_config()
    assert "project" in cfg
    assert cfg["project"]["version"] == "6.0.0"


def test_validate_raw_ok(sample_raw):
    cfg = load_config()
    validate_raw_data(
        sample_raw,
        required_columns=cfg["validation"]["required_columns"],
        min_rows=10,
    )


def test_validate_raw_missing_column(sample_raw):
    cfg = load_config()
    bad = sample_raw.drop(columns=["gender"])
    with pytest.raises(DataValidationError):
        validate_raw_data(bad, required_columns=cfg["validation"]["required_columns"])


def test_feature_engineering(sample_raw):
    engineered = engineer_features(sample_raw)
    assert "lead_cat_temp" in engineered.columns
    assert "booking_month" in engineered.columns
    assert "appointment_month" in engineered.columns


def test_pipeline_init():
    pipe = HealthConnectPipeline()
    assert pipe.config is not None
    assert pipe.root.exists()


def test_integration_validation():
    """Requires that artefacts exist (run train first if needed)."""
    pipe = HealthConnectPipeline()
    # At minimum artefacts from Week 5 should allow partial checks
    checks = pipe.validate_integration()
    assert isinstance(checks, dict)
    assert "artefact_exists_transformer" in checks
