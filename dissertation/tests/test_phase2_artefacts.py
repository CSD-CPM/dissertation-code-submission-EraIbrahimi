"""Phase 2 (predictive pillar) artefact integrity."""
from __future__ import annotations

import json

import pandas as pd

from src import config as cfg


def test_phase2_optuna_complete():
    payload = json.loads((cfg.METRICS / "optuna_best_params.json").read_text())
    assert payload["n_trials_completed"] == 100, (
        f"Optuna: expected 100 completed trials, got {payload['n_trials_completed']}."
    )
    assert payload["n_trials_failed"] == 0, (
        f"Optuna: expected 0 failed trials, got {payload['n_trials_failed']}."
    )


def test_phase2_cv_5_folds():
    cv = pd.read_csv(cfg.METRICS / "cv_xgb_vs_ppml.csv")
    assert len(cv) == 5, (
        f"Phase 2 CV table: expected exactly 5 fold rows (5-fold "
        f"non-overlapping CV), got {len(cv)}."
    )
    # Sanity: required columns
    required = {
        "last_train_year", "test_year", "n_test",
        "ppml_rmse_eur_thousands", "ppml_r2",
        "xgb_rmse_eur_thousands",  "xgb_r2",
        "winner",
    }
    missing = required - set(cv.columns)
    assert not missing, f"cv_xgb_vs_ppml.csv missing columns: {missing}"


def test_phase2_holdout_present():
    payload = json.loads((cfg.METRICS / "holdout_2023_2024.json").read_text())
    assert "xgboost" in payload
    per_year = payload["xgboost"]["per_year"]
    assert "2023" in per_year and "2024" in per_year, (
        f"Holdout JSON: missing per-year entry for 2023 or 2024: "
        f"{sorted(per_year.keys())}"
    )
    assert "combined" in payload["xgboost"]
    # PPML pooled-combined block should also be there
    assert "ppml_predictive_combined" in payload, (
        "Holdout JSON missing pooled-prediction PPML combined block."
    )


def test_phase2_scenario_verdict():
    payload = json.loads((cfg.METRICS / "scenario_verdict.json").read_text())
    required = {
        "improvement_pct", "scenario", "scenario_description",
        "overfit_flag_C", "scenarios_applied",
        "in_sample_r2", "oos_r2", "r2_gap",
        "xgb_holdout_rmse", "xgb_holdout_r2",
        "ppml_holdout_rmse", "ppml_holdout_r2",
    }
    missing = required - set(payload.keys())
    assert not missing, (
        f"scenario_verdict.json missing keys: {missing}"
    )
    valid_labels = {"A", "intermediate_xgb_gain", "B",
                    "intermediate_ppml_gain", "ppml_wins_big"}
    assert payload["scenario"] in valid_labels, (
        f"scenario_verdict.scenario must be one of {valid_labels}; "
        f"got {payload['scenario']!r}."
    )
