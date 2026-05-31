"""Phase 2 orchestrator — XGBoost + Optuna predictive pillar.

Runs end-to-end:

    python -m src.run_phase2

Steps (per implementation plan §5):
  1. Load Phase-1 bilateral panel.
  2. Build (X, y, meta) for XGBoost; report drop counts.
  3. Optuna 100-trial TPE search over cfg.OPTUNA_SEARCH_SPACE; objective is
     mean RMSE (level scale) across the 5 expanding-window folds.
  4. PPML-Predictive on the same 5 folds (head-to-head baseline).
  4b. PPML-Predictive APPENDIX CV on the natural 1,671-row set
      (no XGBoost-feasibility restriction).
  5. XGBoost with best params on the same 5 folds.
  5b. Pooled CV per-row prediction errors → cv_prediction_errors.csv
      (handed off to Phase 3 for the Diebold-Mariano test).
  6. Final 2023+2024 holdout: train on 2010-2022, score per year + combined
     (combined R² computed on POOLED predictions, not aggregated per year).
  7. Seed-stability check across {42, 142, 242} on the holdout.
  7.5. Direction-aware Scenario verdict (5 mutually-exclusive labels +
       independent overfit C flag) → scenario_verdict.json.
  8. Persist optuna_trials.csv, optuna_best_params.json,
     cv_xgb_vs_ppml.csv, holdout_2023_2024.json, seed_stability.json,
     cv_ppml_natural_rows.csv, cv_prediction_errors.csv,
     scenario_verdict.json, models/xgb_best.joblib.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import xgb_model as xgb


def main():
    print("\n[Phase 2] step 1/8 — load Phase-1 panel")
    panel = xgb.load_panel()
    print(f"  panel: {len(panel)} rows × {panel.shape[1]} cols")

    print("\n[Phase 2] step 2/8 — build XGBoost (X, y, meta)")
    X, y, meta = xgb.build_xgb_xy(panel)
    print(f"  XGBoost feeds: n={len(X)}, features={X.shape[1]} ({list(X.columns)})")
    print(f"  log-target: y = log1p(imports_eur_thousands), back-transform on score")

    print("\n[Phase 2] step 3/8 — Optuna 100-trial TPE search "
          "(objective: mean RMSE_levels across 5 CV folds)")
    t0 = time.time()
    study_res = xgb.optuna_search(
        X, y, meta,
        n_trials=cfg.OPTUNA_TRIALS,
        seed=cfg.SEED,
        folds=cfg.CV_FOLDS,
        persist_csv=cfg.METRICS / "optuna_trials.csv",
    )
    wall = time.time() - t0
    print(f"  trials completed: {study_res.n_completed} / {cfg.OPTUNA_TRIALS} "
          f"(failed: {study_res.n_failed})")
    print(f"  best mean fold RMSE (EUR thousands): {study_res.best_value:,.2f}")
    print(f"  best params: {study_res.best_params}")
    print(f"  wall time: {wall:,.1f}s")

    # Persist best params
    best_params_path = cfg.METRICS / "optuna_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "best_value_rmse_eur_thousands": study_res.best_value,
                "best_params": study_res.best_params,
                "n_trials_completed": study_res.n_completed,
                "n_trials_failed": study_res.n_failed,
                "wall_seconds": round(wall, 2),
                "search_space": cfg.OPTUNA_SEARCH_SPACE,
                "cv_folds": cfg.CV_FOLDS,
                "seed": cfg.SEED,
            },
            f, indent=2,
        )
    print(f"  → wrote {best_params_path}")

    print("\n[Phase 2] step 4/8 — PPML-Predictive on the same 5 CV folds "
          "(head-to-head baseline; identical-rows = XGBoost-feasible 1,559)")
    ppml_cv, ppml_cv_preds = xgb.ppml_predictive_cv(
        panel,
        train_test_pairs=cfg.CV_FOLDS,
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=True,
        return_predictions=True,
    )
    print(ppml_cv.to_string(index=False))

    # Step 4b — APPENDIX CV: PPML on its own 1,671-row natural feasible set
    print("\n[Phase 2] step 4b/8 — PPML-Predictive APPENDIX CV "
          "(natural rows; no XGBoost-feasibility restriction)")
    ppml_natural = xgb.ppml_predictive_cv(
        panel,
        train_test_pairs=cfg.CV_FOLDS,
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=False,
    )
    print(ppml_natural.to_string(index=False))
    nat_path = cfg.METRICS / "cv_ppml_natural_rows.csv"
    ppml_natural.to_csv(nat_path, index=False)
    print(f"  → wrote {nat_path}")
    print(f"  mean PPML-natural RMSE: "
          f"{ppml_natural['ppml_rmse_eur_thousands'].mean():,.2f}  "
          f"vs PPML-matched: {ppml_cv['ppml_rmse_eur_thousands'].mean():,.2f}")

    print("\n[Phase 2] step 5/8 — XGBoost (best params) on the same 5 CV folds")
    xgb_cv_df, xgb_cv_preds = xgb.xgb_cv(
        panel,
        best_params=study_res.best_params,
        train_test_pairs=cfg.CV_FOLDS,
        seed=cfg.SEED,
        return_predictions=True,
    )
    print(xgb_cv_df.to_string(index=False))

    # Merge CV side-by-side
    cv_df = ppml_cv.merge(xgb_cv_df, on=["last_train_year", "test_year", "n_train", "n_test"])
    cv_df["winner"] = np.where(
        cv_df["xgb_rmse_eur_thousands"] < cv_df["ppml_rmse_eur_thousands"], "xgb",
        np.where(cv_df["xgb_rmse_eur_thousands"] > cv_df["ppml_rmse_eur_thousands"], "ppml", "tie")
    )
    cv_path = cfg.METRICS / "cv_xgb_vs_ppml.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"\n  → wrote {cv_path}")
    print("  CV head-to-head:")
    print(cv_df.to_string(index=False))
    print(f"  mean PPML RMSE: {cv_df['ppml_rmse_eur_thousands'].mean():,.2f}  "
          f"R² mean: {cv_df['ppml_r2'].mean():.4f}")
    print(f"  mean XGB  RMSE: {cv_df['xgb_rmse_eur_thousands'].mean():,.2f}  "
          f"R² mean: {cv_df['xgb_r2'].mean():.4f}")

    # Step 5b — pooled CV prediction errors → cv_prediction_errors.csv (Phase 3 DM input)
    print("\n[Phase 2] step 5b/8 — pooled CV per-row prediction errors "
          "(handed off to Phase 3 Diebold-Mariano)")
    merge_keys = ["last_train_year", "test_year", "partner_id", "iso2",
                  "year", "imports_eur_thousands"]
    cv_pred_errors = ppml_cv_preds.merge(xgb_cv_preds, on=merge_keys, how="inner")
    cv_pred_errors["fold"] = cv_pred_errors["last_train_year"]
    cv_pred_errors = cv_pred_errors.rename(
        columns={"imports_eur_thousands": "y_true_eur_thousands"}
    )
    cv_pred_errors["sq_err_ppml"] = (
        cv_pred_errors["y_pred_ppml"] - cv_pred_errors["y_true_eur_thousands"]
    ) ** 2
    cv_pred_errors["sq_err_xgb"] = (
        cv_pred_errors["y_pred_xgb"] - cv_pred_errors["y_true_eur_thousands"]
    ) ** 2
    cv_pred_errors["loss_diff_ppml_minus_xgb"] = (
        cv_pred_errors["sq_err_ppml"] - cv_pred_errors["sq_err_xgb"]
    )
    # Order columns per plan §Change 5
    cv_pred_errors = cv_pred_errors[[
        "fold", "last_train_year", "test_year", "partner_id", "iso2", "year",
        "y_true_eur_thousands", "y_pred_ppml", "y_pred_xgb",
        "sq_err_ppml", "sq_err_xgb", "loss_diff_ppml_minus_xgb",
    ]]
    expected_n = int(cv_df["n_test"].sum())
    actual_n = len(cv_pred_errors)
    assert actual_n == expected_n, (
        f"cv_prediction_errors.csv has {actual_n} rows, "
        f"expected {expected_n} from cv_df['n_test'].sum()"
    )
    cv_pred_path = cfg.METRICS / "cv_prediction_errors.csv"
    cv_pred_errors.to_csv(cv_pred_path, index=False)
    mean_loss_diff = float(cv_pred_errors["loss_diff_ppml_minus_xgb"].mean())
    print(f"  rows: {actual_n} (expected from cv_df['n_test'].sum() = {expected_n}) ✓")
    print(f"  mean loss_diff_ppml_minus_xgb: {mean_loss_diff:+,.2f}  "
          f"(positive ⇒ PPML loses more squared error on average ⇒ XGB favoured)")
    print(f"  → wrote {cv_pred_path}")

    print("\n[Phase 2] step 6/8 — final holdout (train ≤ 2022; predict 2023, 2024)")
    holdout = xgb.fit_holdout(
        best_params=study_res.best_params,
        panel=panel,
        seed=cfg.SEED,
        save_model_to=cfg.MODELS / "xgb_best.joblib",
    )
    print(f"  XGBoost holdout (best params, seed={cfg.SEED}):")
    for yr, sc in holdout["per_year"].items():
        print(f"    {yr}: n={sc['n']}  RMSE={sc['rmse']:,.2f}  "
              f"MAE={sc['mae']:,.2f}  R²={sc['r2']:.4f}")
    if "combined" in holdout:
        sc = holdout["combined"]
        print(f"    combined: n={sc['n']}  RMSE={sc['rmse']:,.2f}  "
              f"MAE={sc['mae']:,.2f}  R²={sc['r2']:.4f}")

    # Holdout PPML for parity — capture per-row predictions for pooled R²
    last_tr, test_yrs = cfg.CV_HOLDOUT
    ppml_holdout_metrics, ppml_holdout_preds = xgb.ppml_predictive_cv(
        panel,
        train_test_pairs=[(last_tr, ty) for ty in test_yrs],
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=True,
        return_predictions=True,
    )
    print(f"  PPML-Predictive holdout (train ≤ {last_tr}, same regressors):")
    for _, r in ppml_holdout_metrics.iterrows():
        print(f"    {int(r['test_year'])}: n={int(r['n_test'])}  "
              f"RMSE={r['ppml_rmse_eur_thousands']:,.2f}  "
              f"MAE={r['ppml_mae']:,.2f}  R²={r['ppml_r2']:.4f}")

    # Pooled PPML holdout (combined R² requires the union of predictions —
    # cannot be aggregated from per-year R²)
    ppml_pooled_holdout = xgb.score_levels(
        ppml_holdout_preds["imports_eur_thousands"].values,
        ppml_holdout_preds["y_pred_ppml"].values,
    )
    print(f"    combined: n={ppml_pooled_holdout['n']}  "
          f"RMSE={ppml_pooled_holdout['rmse']:,.2f}  "
          f"MAE={ppml_pooled_holdout['mae']:,.2f}  "
          f"R²={ppml_pooled_holdout['r2']:.4f}")

    # Persist holdout JSON
    holdout_out = {
        "xgboost": holdout,
        "ppml_predictive_per_year": ppml_holdout_metrics.to_dict(orient="records"),
        "ppml_predictive_combined": ppml_pooled_holdout,
    }
    holdout_path = cfg.METRICS / "holdout_2023_2024.json"
    with open(holdout_path, "w") as f:
        json.dump(holdout_out, f, indent=2, default=str)
    print(f"  → wrote {holdout_path}")
    print(f"  → wrote {cfg.MODELS / 'xgb_best.joblib'}")

    print("\n[Phase 2] step 7/8 — seed-stability check {42, 142, 242} on the holdout")
    stab = xgb.seed_stability(
        best_params=study_res.best_params,
        panel=panel,
        seeds=(42, 142, 242),
    )
    print(f"  RMSE per seed (combined 2023+2024): "
          f"{stab['rmse_min']:,.2f} – {stab['rmse_max']:,.2f}  "
          f"(mean={stab['rmse_mean']:,.2f}, std={stab['rmse_std']:,.4f})")
    stab_path = cfg.METRICS / "seed_stability.json"
    with open(stab_path, "w") as f:
        json.dump(stab, f, indent=2, default=str)
    print(f"  → wrote {stab_path}")

    # ----------------------------------------------------------------------
    # Step 7.5 — direction-aware Scenario verdict (PROTOCOL_FREEZE §6)
    # 5 mutually-exclusive labels for improvement_pct + an INDEPENDENT
    # overfit C flag. Pooled PPML holdout RMSE/MAE/R² come from per-row
    # predictions captured in step 6 (NOT aggregated from per-year metrics).
    # ----------------------------------------------------------------------
    print("\n[Phase 2] step 7.5/8 — Scenario verdict (direction-aware, holdout-based)")
    ppml_holdout_combined_rmse = ppml_pooled_holdout["rmse"]
    ppml_holdout_combined_mae  = ppml_pooled_holdout["mae"]
    ppml_holdout_combined_r2   = ppml_pooled_holdout["r2"]
    xgb_holdout_rmse = holdout["combined"]["rmse"]
    xgb_holdout_r2   = holdout["combined"]["r2"]

    improvement_pct = 100.0 * (ppml_holdout_combined_rmse - xgb_holdout_rmse) / ppml_holdout_combined_rmse

    gap_info = xgb.xgb_train_test_r2(panel, study_res.best_params, seed=cfg.SEED)
    r2_gap = gap_info["r2_gap"]

    if improvement_pct >= 15.0:
        scenario = "A"
        scenario_description = (
            f"ML wins (XGB RMSE {improvement_pct:.2f}% lower than PPML, ≥15%)"
        )
    elif 5.0 <= improvement_pct < 15.0:
        scenario = "intermediate_xgb_gain"
        scenario_description = (
            f"intermediate XGB advantage ({improvement_pct:.2f}% in [5%, 15%))"
        )
    elif abs(improvement_pct) < 5.0:
        scenario = "B"
        scenario_description = (
            f"tie (|improvement| = {abs(improvement_pct):.2f}% < 5%)"
        )
    elif -15.0 < improvement_pct <= -5.0:
        scenario = "intermediate_ppml_gain"
        scenario_description = (
            f"intermediate PPML advantage "
            f"(PPML RMSE {-improvement_pct:.2f}% lower than XGB, in [5%, 15%))"
        )
    else:  # improvement_pct <= -15.0
        scenario = "ppml_wins_big"
        scenario_description = (
            f"PPML wins (PPML RMSE {-improvement_pct:.2f}% lower than XGB, ≥15%)"
        )

    overfit_flag_C = bool(r2_gap > 0.20)
    scenarios_applied = [scenario] + (["C"] if overfit_flag_C else [])

    verdict = {
        "improvement_pct": improvement_pct,
        "scenario": scenario,
        "scenario_description": scenario_description,
        "overfit_flag_C": overfit_flag_C,
        "scenarios_applied": scenarios_applied,
        "in_sample_r2": gap_info["in_sample_r2"],
        "oos_r2":       gap_info["oos_r2"],
        "r2_gap":       r2_gap,
        "xgb_holdout_rmse":  xgb_holdout_rmse,
        "xgb_holdout_r2":    xgb_holdout_r2,
        "ppml_holdout_rmse": ppml_holdout_combined_rmse,
        "ppml_holdout_mae":  ppml_holdout_combined_mae,
        "ppml_holdout_r2":   ppml_holdout_combined_r2,
    }
    print(f"  XGB  holdout RMSE: {xgb_holdout_rmse:,.2f}   R²: {xgb_holdout_r2:.4f}")
    print(f"  PPML holdout RMSE: {ppml_holdout_combined_rmse:,.2f}   "
          f"R²: {ppml_holdout_combined_r2:.4f}")
    print(f"  improvement_pct:   {improvement_pct:+.2f}%  →  Scenario {scenario}: "
          f"{scenario_description}")
    print(f"  in-sample R²: {gap_info['in_sample_r2']:.4f}  "
          f"OOS R²: {gap_info['oos_r2']:.4f}  gap: {r2_gap:.4f}")
    print(f"  Scenario C (overfit, R² gap > 0.20): {overfit_flag_C}")
    print(f"  scenarios_applied: {scenarios_applied}")
    verdict_path = cfg.METRICS / "scenario_verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"  → wrote {verdict_path}")

    print("\n[Phase 2] step 8/8 — done. Awaiting sign-off before Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
