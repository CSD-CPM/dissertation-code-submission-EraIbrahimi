"""Contract test: every artefact a dashboard page reads exists and has the
required columns (CSV/parquet) or keys (JSON)."""
import json

import pandas as pd
import pytest

from src import config as cfg

CSV_CONTRACTS = {
    "tbl_ch3_annual_totals.csv": {"year", "exports", "imports", "deficit", "export_to_import_ratio"},
    "tbl_ch3_zero_counts.csv": {"year", "flow", "zero_rate_pct"},
    "tbl_ch3_concentration.csv": {"flow", "scope", "hhi", "top_5_share_pct", "top_10_share_pct", "top_20_share_pct"},
    "tbl_ch3_top_partners_2024.csv": {"iso2", "partner_name", "imports", "exports"},
    "tbl_ch3_partner_asymmetry.csv": {"iso2", "partner_name", "cum_imports", "cum_exports", "classification"},
    "tbl_ch3_serbia_trajectory.csv": {"iso2", "year", "flow", "value_eur_thousands"},
    "tbl_ch3_serbia_monthly_breakpoint.csv": {"date", "value_eur_thousands", "mom_pct", "mom_z", "window"},
    "tbl_ch3_sector_composition.csv": {"year", "flow", "rank", "hs_section", "share_of_flow_pct"},
    "tbl_ch3_sector_event_response_ppshift.csv": {"hs_section", "flow", "event_label", "delta_pp", "z_score"},
    "tbl_ch5_export_spillover_summary.csv": {"flow", "n", "beta", "ci_95_low", "ci_95_high"},
    "tbl_ch5_parallel_trends.csv": {"group", "year", "mean_log_imports"},
    "tbl_ch5_leads_test.csv": set(),
    "tbl_ch5_event_study.csv": {"year", "beta", "ci_low_95", "ci_high_95"},
    "tbl_ch5_safeguards_summary.csv": set(),
    "cv_xgb_vs_ppml.csv": {"test_year", "ppml_rmse_eur_thousands", "xgb_rmse_eur_thousands", "xgb_r2", "ppml_r2"},
    "cv_xgb_vs_ppml_export.csv": {"test_year", "ppml_rmse_eur_thousands", "xgb_rmse_eur_thousands", "xgb_r2", "ppml_r2"},
    "tbl_ch4_shap_global.csv": {"feature", "mean_abs_shap"},
    "tbl_ch4_shap_global_export.csv": {"feature", "mean_abs_shap"},
    "tbl_ch4_ablation_cv.csv": {"layer", "fold_idx", "n_features", "rmse", "r2"},
    "tbl_ch4_ablation_holdout.csv": {"layer", "rmse", "r2"},
    "tbl_ch4_import_vs_export_prediction.csv": {"section", "metric", "import_value", "export_value"},
}

JSON_CONTRACTS = {
    "bootstrap_ci.json": {"beta_main", "bootstrap_ci_low", "bootstrap_ci_high", "exp_beta_minus_1_pct", "n_boot_completed"},
    "bootstrap_ci_export.json": {"beta_export", "bootstrap_ci_low", "bootstrap_ci_high", "exp_beta_minus_1_pct", "n_boot_completed"},
    "import_vs_export_did_comparison.json": {"beta_import", "beta_export", "import_ci_95_low", "import_ci_95_high", "export_ci_95_low", "export_ci_95_high"},
    "leads_test.json": {"joint_p_value", "joint_wald_stat", "df"},
    "holdout_2023_2024.json": {"xgboost"},
    "holdout_export_2023_2024.json": {"xgboost", "ppml_predictive_combined"},
    "dm_test.json": {"n", "dm_hln", "p_value", "interpretation"},
    "dm_test_export.json": {"n", "dm_hln", "p_value", "interpretation"},
    "persistence_baseline_export.json": {"cv", "holdout_combined"},
    "scenario_verdict.json": {"scenario", "improvement_pct", "overfit_flag_C", "r2_gap"},
    "export_modellability.json": {"export_zero_rate", "import_zero_rate", "var_log_ratio_exp_to_imp"},
}

NPY_FILES = ["bootstrap_draws.npy", "bootstrap_draws_export.npy"]

PARQUET_CONTRACTS = {
    "panel_bilateral.parquet": {"iso2", "year", "imports_eur_thousands"},
    "panel_bilateral_export.parquet": {"iso2", "year", "exports_eur_thousands"},
    "panel_sector.parquet": {"year", "hs_section", "imports_eur_thousands", "exports_eur_thousands"},
}


@pytest.mark.parametrize("name,required", CSV_CONTRACTS.items())
def test_csv_artefacts(name, required):
    p = cfg.TABLES / name
    if not p.exists():
        p = cfg.METRICS / name
    assert p.exists(), f"missing artefact: {name}"
    cols = set(pd.read_csv(p).columns)
    assert required <= cols, f"{name} missing columns: {required - cols}"


@pytest.mark.parametrize("name,keys", JSON_CONTRACTS.items())
def test_json_artefacts(name, keys):
    p = cfg.METRICS / name
    assert p.exists(), f"missing artefact: {name}"
    got = set(json.load(open(p)).keys())
    assert keys <= got, f"{name} missing keys: {keys - got}"


@pytest.mark.parametrize("name", NPY_FILES)
def test_npy_artefacts(name):
    assert (cfg.METRICS / name).exists(), f"missing artefact: {name}"


@pytest.mark.parametrize("name,required", PARQUET_CONTRACTS.items())
def test_parquet_artefacts(name, required):
    p = cfg.DATA_PROCESSED / name
    assert p.exists(), f"missing artefact: {name}"
    cols = set(pd.read_parquet(p).columns)
    assert required <= cols, f"{name} missing columns: {required - cols}"
