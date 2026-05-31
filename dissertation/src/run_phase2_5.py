"""Phase 2.5 orchestrator - export-side predictive ML.

Mirrors run_phase2.py / run_phase3.py. Reuses xgb_model.py verbatim
via a three-column swap on the export panel; no Phase 0/1/1.5/1-5
module is modified.

Run from dissertation/:
    python -m src.run_phase2_5
"""
from __future__ import annotations

import json
import sys
import time

import pandas as pd

from . import config as cfg
from . import xgb_export


def main() -> int:
    t0 = time.time()

    print("[phase2.5] Step A - build export panel ...")
    panel_export = xgb_export.build_export_panel()
    print(f"[phase2.5]   panel rows: {len(panel_export)}")

    print("\n[phase2.5] Step B - Optuna 100-trial search (export) ...")
    t_opt = time.time()
    opt = xgb_export.run_export_optuna(panel_export)
    opt_wall = time.time() - t_opt
    print(f"[phase2.5]   best CV RMSE: {opt.best_value:,.2f}")
    print(f"[phase2.5]   best params : {opt.best_params}")
    print(f"[phase2.5]   trials      : {opt.n_completed} completed, {opt.n_failed} failed")
    print(f"[phase2.5]   wall time   : {opt_wall:.1f}s")
    optuna_record = {
        "best_value_rmse_eur_thousands": opt.best_value,
        "best_params": opt.best_params,
        "n_trials_completed": opt.n_completed,
        "n_trials_failed": opt.n_failed,
        "wall_seconds": float(opt_wall),
        "search_space": cfg.OPTUNA_SEARCH_SPACE,
        "cv_folds": [list(f) for f in cfg.CV_FOLDS],
        "seed": int(cfg.SEED),
    }
    with open(cfg.METRICS / "optuna_best_params_export.json", "w") as f:
        json.dump(optuna_record, f, indent=2)

    print("\n[phase2.5] Step C - CV head-to-head (PPML-export vs XGB-export) ...")
    h2h = xgb_export.run_export_cv_head_to_head(
        panel_export, opt.best_params, seed=cfg.SEED
    )
    per_fold = h2h["per_fold"].copy()
    mean_row = pd.DataFrame([{
        "last_train_year": "mean",
        "test_year": "mean",
        "n_train": per_fold["n_train"].sum(),
        "n_test": per_fold["n_test"].sum(),
        "ppml_rmse_eur_thousands": per_fold["ppml_rmse_eur_thousands"].mean(),
        "ppml_mae": per_fold["ppml_mae"].mean(),
        "ppml_r2": per_fold["ppml_r2"].mean(),
        "xgb_rmse_eur_thousands": per_fold["xgb_rmse_eur_thousands"].mean(),
        "xgb_mae": per_fold["xgb_mae"].mean(),
        "xgb_r2": per_fold["xgb_r2"].mean(),
        "winner": "n/a",
    }])
    pd.concat([per_fold, mean_row], ignore_index=True).to_csv(
        cfg.METRICS / "cv_xgb_vs_ppml_export.csv", index=False)
    h2h["per_row"].to_csv(
        cfg.METRICS / "cv_prediction_errors_export.csv", index=False)
    print(f"[phase2.5]   CV pool rows : {len(h2h['per_row'])}")
    for _, r in per_fold.iterrows():
        print(f"[phase2.5]     fold {r['last_train_year']}->{r['test_year']}: "
              f"XGB RMSE={r['xgb_rmse_eur_thousands']:>12,.1f} R2={r['xgb_r2']:+.4f}  "
              f"PPML RMSE={r['ppml_rmse_eur_thousands']:>12,.1f} R2={r['ppml_r2']:+.4f}  "
              f"winner={r['winner']}")

    print("\n[phase2.5] Step D - Holdout 2023+2024 ...")
    ho = xgb_export.run_export_holdout(
        panel_export, opt.best_params, seed=cfg.SEED
    )
    with open(cfg.METRICS / "holdout_export_2023_2024.json", "w") as f:
        json.dump(ho, f, indent=2)
    print(f"[phase2.5]   XGB combined RMSE: {ho['xgboost']['combined']['rmse']:,.2f}, "
          f"R2: {ho['xgboost']['combined']['r2']:+.4f}, "
          f"n={ho['xgboost']['combined']['n']}")
    print(f"[phase2.5]   PPML combined RMSE: {ho['ppml_predictive_combined']['rmse']:,.2f}, "
          f"R2: {ho['ppml_predictive_combined']['r2']:+.4f}")

    print("\n[phase2.5] Step E - Persistence baseline ...")
    persist = xgb_export.run_export_persistence_baseline(panel_export)
    with open(cfg.METRICS / "persistence_baseline_export.json", "w") as f:
        json.dump(persist, f, indent=2)
    print(f"[phase2.5]   CV pool n={persist['cv']['n']}: "
          f"RMSE={persist['cv']['rmse_eur_thousands']:,.2f}, "
          f"R2={persist['cv']['r2']:+.4f}")
    print(f"[phase2.5]   Holdout combined n={persist['holdout_combined']['n']}: "
          f"RMSE={persist['holdout_combined']['rmse_eur_thousands']:,.2f}, "
          f"R2={persist['holdout_combined']['r2']:+.4f}")

    print("\n[phase2.5] Step F - Diebold-Mariano (HLN, h=1) ...")
    dm = xgb_export.run_export_dm_test(
        cv_prediction_errors_export_path=cfg.METRICS / "cv_prediction_errors_export.csv"
    )
    with open(cfg.METRICS / "dm_test_export.json", "w") as f:
        json.dump(dm, f, indent=2)
    print(f"[phase2.5]   n={dm['n']}, DM_HLN={dm['dm_hln']:.4f}, p={dm['p_value']:.4f}")
    print(f"[phase2.5]   {dm['interpretation']}")

    print("\n[phase2.5] Step G - SHAP global ...")
    shap_summary = xgb_export.compute_export_shap_global(
        panel_export,
        booster_path=cfg.MODELS / "xgb_best_export.joblib",
    )
    print("[phase2.5]   top-5 features by mean |SHAP|:")
    print(shap_summary.head(5).to_string(index=False))

    print("\n[phase2.5] Step H - Import-vs-export comparison ...")
    panel_import = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    comp = xgb_export.build_import_vs_export_prediction_comparison(
        panel_import=panel_import,
        panel_export=panel_export,
        export_best_params=opt.best_params,
        export_persistence=persist,
    )
    print(f"[phase2.5]   comparison table rows: {len(comp)}")

    print("\n[phase2.5] Done in %.1fs." % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
