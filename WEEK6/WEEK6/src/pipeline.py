"""
HealthConnect Integrated ML Pipeline - Week 6
End-to-end orchestration: validate → engineer → preprocess → train/predict → validate outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .data_validation import (
    DataValidationError,
    validate_prediction_output,
    validate_processed_features,
    validate_raw_data,
)
from .feature_engineering import (
    engineer_features,
    select_features_and_target,
    temporal_split,
)
from .preprocessing import (
    drop_date_columns,
    fit_transform_features,
    load_preprocessor,
    save_preprocessor,
)
from .train import evaluate_model, load_model, save_model, train_model
from .predict import predict_batch
from .utils import ensure_dir, get_project_root, load_config, setup_logging

logger = logging.getLogger("healthconnect.pipeline")


class HealthConnectPipeline:
    """
    Integrated pipeline for no-show prediction.
    Can run full training flow or inference-only flow using saved artefacts.
    """

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config = load_config(config_path)
        self.root = get_project_root()
        self.transformer = None
        self.model = None
        self.feature_names: list = []
        self.metrics: Dict[str, float] = {}

        # Setup logging from config
        log_cfg = self.config.get("logging", {})
        setup_logging(
            level=log_cfg.get("level", "INFO"),
            log_file=log_cfg.get("file"),
            log_format=log_cfg.get("format"),
        )
        logger.info("HealthConnectPipeline initialised (v%s)", self.config["project"]["version"])

    # ------------------------------------------------------------------
    # Paths helpers
    # ------------------------------------------------------------------
    def _path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    # ------------------------------------------------------------------
    # Training / integration flow
    # ------------------------------------------------------------------
    def run_training_pipeline(self, retrain: bool = True) -> Dict[str, Any]:
        """
        Full training pipeline:
        1. Load & validate raw data
        2. Feature engineering
        3. Temporal split
        4. Preprocessing (fit + transform)
        5. Train model(s)
        6. Evaluate
        7. Save artefacts
        8. Validate outputs
        """
        logger.info("=" * 60)
        logger.info("Starting TRAINING pipeline")
        logger.info("=" * 60)

        cfg = self.config
        data_cfg = cfg["data"]
        prep_cfg = cfg["preprocessing"]
        model_cfg = cfg["model"]
        val_cfg = cfg["validation"]

        # 1. Load raw
        raw_path = self._path(data_cfg["raw_path"])
        df = pd.read_csv(raw_path)
        logger.info("Loaded raw data: %s", df.shape)

        validate_raw_data(
            df,
            required_columns=val_cfg["required_columns"],
            min_rows=val_cfg["min_rows"],
            max_missing_ratio=val_cfg["max_missing_ratio"],
        )

        # 2. Feature engineering
        df = engineer_features(
            df,
            lead_bins=prep_cfg.get("lead_bins"),
            lead_labels=prep_cfg.get("lead_labels"),
        )

        # 3. Select X/y and temporal split
        X, y = select_features_and_target(df, id_columns=data_cfg["id_columns"])
        if y is None:
            raise DataValidationError("Target column missing after engineering")

        X_train, X_test, y_train, y_test = temporal_split(
            X, y, date_col="appointment_date", ratio=prep_cfg["temporal_split_ratio"]
        )

        # 4. Preprocessing
        cat_feats = prep_cfg["categorical_features"]
        num_feats = [f for f in prep_cfg["numeric_features"] if f != "previous_no_shows"]

        X_train_t, X_test_t, transformer, feature_names = fit_transform_features(
            X_train, X_test, cat_feats, num_feats
        )
        self.transformer = transformer
        self.feature_names = feature_names

        validate_processed_features(X_train_t, y_train)
        validate_processed_features(X_test_t, y_test)

        # Persist processed data
        processed_dir = ensure_dir(self._path(data_cfg["processed_dir"]))
        X_train_t.to_csv(processed_dir / "X_train_processed.csv", index=False)
        X_test_t.to_csv(processed_dir / "X_test_processed.csv", index=False)
        y_train.to_frame().to_csv(processed_dir / "y_train_processed.csv", index=False)
        y_test.to_frame().to_csv(processed_dir / "y_test_processed.csv", index=False)

        # Save transformer
        models_dir = ensure_dir(self._path(model_cfg["models_dir"]))
        transformer_path = models_dir / "column_transformer.pkl"
        save_preprocessor(transformer, str(transformer_path))

        # 5. Train models
        results = {}
        for name in model_cfg.get("candidate_models", ["decision_tree"]):
            if not retrain and (models_dir / f"{name}.pkl").exists():
                model = load_model(models_dir / f"{name}.pkl")
                logger.info("Loaded existing model: %s", name)
            else:
                model = train_model(name, X_train_t, y_train)
                save_model(model, models_dir / f"{name}.pkl")

            metrics = evaluate_model(model, X_test_t, y_test)
            results[name] = {"model": model, "metrics": metrics}
            logger.info("Model %s metrics: %s", name, metrics)

        # Select default
        default_name = model_cfg.get("default_model", "decision_tree")
        self.model = results[default_name]["model"]
        self.metrics = results[default_name]["metrics"]

        logger.info("TRAINING pipeline completed successfully")
        return {
            "models": {k: v["metrics"] for k, v in results.items()},
            "feature_names": feature_names,
            "n_train": len(X_train_t),
            "n_test": len(X_test_t),
        }

    # ------------------------------------------------------------------
    # Inference flow (uses saved artefacts)
    # ------------------------------------------------------------------
    def run_inference(
        self,
        input_df: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Inference on new data:
        1. Validate input
        2. Engineer features
        3. Load preprocessor + model
        4. Transform
        5. Predict
        6. Validate output
        """
        logger.info("=" * 60)
        logger.info("Starting INFERENCE pipeline")
        logger.info("=" * 60)

        cfg = self.config
        model_cfg = cfg["model"]
        prep_cfg = cfg["preprocessing"]
        val_cfg = cfg["validation"]
        paths = cfg["paths"]["models"]

        # Keep IDs for output
        appointment_ids = None
        if "appointment_id" in input_df.columns:
            appointment_ids = input_df["appointment_id"].tolist()

        validate_raw_data(
            input_df,
            required_columns=[c for c in val_cfg["required_columns"] if c != "appointment_outcome"],
            min_rows=1,
            max_missing_ratio=val_cfg["max_missing_ratio"],
        )

        df = engineer_features(
            input_df,
            lead_bins=prep_cfg.get("lead_bins"),
            lead_labels=prep_cfg.get("lead_labels"),
        )
        X, _ = select_features_and_target(df, id_columns=cfg["data"]["id_columns"])
        X = drop_date_columns(X)

        # Load artefacts if not already in memory
        if self.transformer is None:
            self.transformer = load_preprocessor(str(self._path(paths["transformer"])))
        if self.model is None:
            name = model_name or model_cfg["default_model"]
            model_path = self._path(paths.get(name, f"models/{name}.pkl"))
            self.model = load_model(model_path)

        X_t = self.transformer.transform(X)
        # Convert to DataFrame with expected columns if possible
        try:
            feat_names = list(self.transformer.get_feature_names_out())
            X_t = pd.DataFrame(X_t, columns=feat_names)
        except Exception:
            X_t = pd.DataFrame(X_t)

        validate_processed_features(X_t)

        preds = predict_batch(
            self.model,
            X_t,
            appointment_ids=appointment_ids,
            model_version=cfg["project"]["version"],
            decision_threshold=model_cfg.get("decision_threshold", 0.5),
        )

        validate_prediction_output(preds)
        logger.info("INFERENCE pipeline completed successfully")
        return preds

    # ------------------------------------------------------------------
    # Integration validation helpers
    # ------------------------------------------------------------------
    def validate_integration(self) -> Dict[str, bool]:
        """
        Run a set of checks confirming that preprocessing and model are compatible.
        Returns a dict of check_name → passed.
        """
        logger.info("Running integration validation checks")
        checks = {}

        models_dir = self._path(self.config["model"]["models_dir"])
        paths = self.config["paths"]["models"]

        # 1. Artefacts exist
        for key, rel in paths.items():
            p = self._path(rel)
            checks[f"artefact_exists_{key}"] = p.exists()
            if not p.exists():
                logger.warning("Missing artefact: %s", p)

        # 2. Can load transformer and model
        try:
            tr = load_preprocessor(str(self._path(paths["transformer"])))
            checks["transformer_loadable"] = True
            self.transformer = tr
        except Exception as e:
            logger.error("Transformer load failed: %s", e)
            checks["transformer_loadable"] = False

        try:
            default = self.config["model"]["default_model"]
            m = load_model(self._path(paths.get(default, f"models/{default}.pkl")))
            checks["model_loadable"] = True
            self.model = m
        except Exception as e:
            logger.error("Model load failed: %s", e)
            checks["model_loadable"] = False

        # 3. End-to-end dry-run on a small sample of raw data
        try:
            raw = pd.read_csv(self._path(self.config["data"]["raw_path"]))
            sample = raw.head(20)
            preds = self.run_inference(sample)
            checks["end_to_end_inference"] = len(preds) == 20
            checks["output_schema_valid"] = "no_show_probability" in preds.columns
        except Exception as e:
            logger.error("End-to-end inference failed: %s", e)
            checks["end_to_end_inference"] = False
            checks["output_schema_valid"] = False

        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        logger.info("Integration validation: %d/%d checks passed", passed, total)
        for k, v in checks.items():
            logger.info("  [%s] %s", "PASS" if v else "FAIL", k)
        return checks
