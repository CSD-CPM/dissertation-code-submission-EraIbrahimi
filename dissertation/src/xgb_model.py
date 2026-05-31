"""XGBoost predictive pillar (Phase 2).

Implementation plan §5:
- All 12 features in `cfg.FEATURE_ORDER` (no addition / no removal).
- Target = `imports_eur_thousands`. Fit in `np.log1p` space; back-transform
  with `np.expm1` before scoring. **Never score in log space.**
- Optuna TPE search over `cfg.OPTUNA_SEARCH_SPACE`, n = `cfg.OPTUNA_TRIALS`.
- Expanding-window CV: folds in `cfg.CV_FOLDS`; final 2023+2024 holdout.
- Cluster / fixed-effect handling is delegated to the locked PPML in
  `ppml.py` for the head-to-head; XGBoost itself takes the raw feature
  vector — partner identity is encoded via `partner_import_share_lag`,
  `lagged_imports_log1p`, `cefta_member`, structural dyad features, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

import json
import numpy as np
import pandas as pd

from . import config as cfg


# =============================================================================
# 1. Panel loader + feature matrix
# =============================================================================

def load_panel() -> pd.DataFrame:
    """Load the Phase-1 feature-ready bilateral panel."""
    p = cfg.DATA_PROCESSED / "panel_bilateral.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"panel_bilateral.parquet not found at {p}. "
            "Run `python -m src.run_phase1` first."
        )
    return pd.read_parquet(p)


def build_xgb_xy(
    panel: pd.DataFrame,
    feature_list: Iterable[str] = None,
    log_target: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build the XGBoost design matrix.

    Drops every row with a NaN in any of the 12 features. Logs the count and
    breakdown so PHASE2_REPORT can document it.

    Returns
    -------
    X : pd.DataFrame  (n × 12, columns in cfg.FEATURE_ORDER)
    y : np.ndarray    (n,)  -- np.log1p(imports_eur_thousands) if log_target
    meta : pd.DataFrame (n × 4)  -- iso2, year, partner_id, imports_eur_thousands
    """
    feature_list = list(feature_list or cfg.FEATURE_ORDER)
    needed = set(feature_list) | {"iso2", "year", "partner_id", "imports_eur_thousands"}
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"build_xgb_xy: panel missing columns: {missing}")

    pre = len(panel)
    sub = panel.dropna(subset=feature_list).copy()
    dropped = pre - len(sub)
    if dropped:
        print(f"[xgb] dropped {dropped} rows with NaN features "
              f"(usual: {(panel['year'] == min(cfg.YEARS)).sum()} lagged-2010 rows "
              f"+ partner-year rows with WDI gaps)")

    X = sub[feature_list].astype(float).reset_index(drop=True)
    raw_y = sub["imports_eur_thousands"].astype(float).values
    y = np.log1p(raw_y) if log_target else raw_y
    meta = sub[["iso2", "year", "partner_id", "imports_eur_thousands"]].reset_index(drop=True)
    return X, y, meta


# =============================================================================
# 2. CV fold iterator + holdout split
# =============================================================================

def expanding_cv_folds(
    meta: pd.DataFrame,
    folds: Iterable[Tuple[int, int]] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Yield (train_idx, test_idx, last_train_year, test_year) row-index arrays.

    For fold (last_train_year=t, test_year=t+1): train on year <= t,
    test on year == t+1.
    """
    folds = list(folds or cfg.CV_FOLDS)
    yrs = meta["year"].values
    for last_train_year, test_year in folds:
        train_mask = yrs <= last_train_year
        test_mask = yrs == test_year
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx, last_train_year, test_year


def holdout_split(
    meta: pd.DataFrame,
    last_train_year: int = None,
    test_years: Iterable[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Build train_idx + a dict of {year -> test_idx} for the final holdout.

    Defaults from cfg.CV_HOLDOUT = (2022, [2023, 2024]).
    """
    if last_train_year is None or test_years is None:
        last_train_year, test_years = cfg.CV_HOLDOUT
    yrs = meta["year"].values
    train_idx = np.where(yrs <= last_train_year)[0]
    test_idx_by_year = {y: np.where(yrs == y)[0] for y in test_years}
    return train_idx, test_idx_by_year


# =============================================================================
# 3. XGBoost fit / predict / score
# =============================================================================

def fit_xgb(X_train, y_train, params: dict, seed: int = cfg.SEED):
    """Fit an XGBoost regressor with the supplied params.

    `params` is expected to come from `cfg.OPTUNA_SEARCH_SPACE` — i.e. one
    value drawn from each discrete grid. The booster runs in the default
    `reg:squarederror` objective; y is already in log1p space when called
    from the orchestrator.
    """
    from xgboost import XGBRegressor
    full = dict(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=int(seed),
        n_jobs=-1,
        verbosity=0,
    )
    full.update(params)
    model = XGBRegressor(**full)
    model.fit(X_train, y_train)
    return model


def predict_levels(model, X) -> np.ndarray:
    """Predict on X and back-transform log1p → level (EUR thousands)."""
    p = np.asarray(model.predict(X), dtype=float)
    return np.expm1(p)


def score_levels(y_true_levels: np.ndarray, y_pred_levels: np.ndarray) -> dict:
    """RMSE + MAE + R² on the level scale (EUR thousands). NEVER on log scale.

    R² uses level-space sums of squares: 1 − SS_res / SS_tot, where
    SS_tot is computed against the y_true mean of THIS sample. When ss_tot
    is exactly zero (constant y), R² is NaN. R² may legitimately be
    negative when the model is worse than predicting the mean.
    """
    y_true = np.asarray(y_true_levels, dtype=float)
    y_pred = np.asarray(y_pred_levels, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}


def xgb_train_test_r2(
    panel: pd.DataFrame,
    best_params: dict,
    seed: int = cfg.SEED,
    last_train_year: int = None,
    test_years: Iterable[int] = None,
) -> dict:
    """In-sample (train) and OOS (holdout) R² for the XGBoost-best model.

    The gap = R²_train − R²_OOS is the Phase-2 §6 (Scenario C) overfit
    diagnostic. Both R²s are on the EUR-thousands level scale.
    """
    if last_train_year is None or test_years is None:
        last_train_year, test_years = cfg.CV_HOLDOUT

    X_full, y_full, meta = build_xgb_xy(panel)
    train_idx, test_idx_by_year = holdout_split(meta, last_train_year, test_years)
    model = fit_xgb(X_full.iloc[train_idx], y_full[train_idx], best_params, seed=seed)

    # In-sample (training) — score on level scale
    y_train_pred = predict_levels(model, X_full.iloc[train_idx])
    y_train_true = meta["imports_eur_thousands"].values[train_idx]
    in_sample = score_levels(y_train_true, y_train_pred)

    # Out-of-sample (combined holdout)
    test_idx_combined = np.concatenate(
        [test_idx_by_year[y] for y in test_years]
    )
    y_oos_pred = predict_levels(model, X_full.iloc[test_idx_combined])
    y_oos_true = meta["imports_eur_thousands"].values[test_idx_combined]
    oos = score_levels(y_oos_true, y_oos_pred)

    return {
        "in_sample_r2":   in_sample["r2"],
        "oos_r2":         oos["r2"],
        "r2_gap":         float(in_sample["r2"] - oos["r2"]),
        "in_sample_rmse": in_sample["rmse"],
        "oos_rmse":       oos["rmse"],
    }


# =============================================================================
# 4. Optuna search
# =============================================================================

def _suggest_from_space(trial, space: dict) -> dict:
    """Translate cfg.OPTUNA_SEARCH_SPACE (dict of lists) to suggest_categorical
    calls — locked search grid, no continuous widening."""
    params = {}
    for name, choices in space.items():
        params[name] = trial.suggest_categorical(name, choices)
    return params


@dataclass
class OptunaResult:
    study: object
    best_params: dict
    best_value: float
    trials_df: pd.DataFrame
    n_completed: int
    n_failed: int


def optuna_search(
    X_full,
    y_full,
    meta: pd.DataFrame,
    n_trials: int = None,
    seed: int = cfg.SEED,
    folds: Iterable[Tuple[int, int]] = None,
    persist_csv: Path = None,
) -> OptunaResult:
    """Bayesian search over cfg.OPTUNA_SEARCH_SPACE.

    Objective: mean RMSE on level scale across the 5 expanding-window folds.
    Sampler: TPESampler(seed=cfg.SEED). Lower is better.
    """
    import optuna
    from optuna.samplers import TPESampler

    n_trials = int(n_trials or cfg.OPTUNA_TRIALS)
    folds = list(folds or cfg.CV_FOLDS)
    fold_specs = list(expanding_cv_folds(meta, folds))

    # Quiet Optuna's own logging; keep our own prints
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = _suggest_from_space(trial, cfg.OPTUNA_SEARCH_SPACE)
        rmses = []
        for tr_idx, te_idx, _, _ in fold_specs:
            model = fit_xgb(X_full.iloc[tr_idx], y_full[tr_idx], params, seed=seed)
            y_pred_lvl = predict_levels(model, X_full.iloc[te_idx])
            y_true_lvl = meta["imports_eur_thousands"].values[te_idx]
            rmses.append(score_levels(y_true_lvl, y_pred_lvl)["rmse"])
        return float(np.mean(rmses))

    sampler = TPESampler(seed=int(seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    n_completed = int((trials_df["state"] == "COMPLETE").sum()) if "state" in trials_df else len(trials_df)
    n_failed = int((trials_df["state"] != "COMPLETE").sum()) if "state" in trials_df else 0

    if persist_csv is not None:
        Path(persist_csv).parent.mkdir(parents=True, exist_ok=True)
        trials_df.to_csv(persist_csv, index=False)

    return OptunaResult(
        study=study,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        trials_df=trials_df,
        n_completed=n_completed,
        n_failed=n_failed,
    )


# =============================================================================
# 5. Final 2023/24 holdout fit
# =============================================================================

def fit_holdout(
    best_params: dict,
    panel: pd.DataFrame,
    seed: int = cfg.SEED,
    last_train_year: int = None,
    test_years: Iterable[int] = None,
    save_model_to: Path = None,
) -> dict:
    """Train XGBoost on years <= last_train_year; predict each test year and
    the union; report level-scale RMSE/MAE per year and combined.
    """
    if last_train_year is None or test_years is None:
        last_train_year, test_years = cfg.CV_HOLDOUT
    test_years = list(test_years)

    X_full, y_full, meta = build_xgb_xy(panel)
    train_idx, test_idx_by_year = holdout_split(meta, last_train_year, test_years)

    model = fit_xgb(X_full.iloc[train_idx], y_full[train_idx], best_params, seed=seed)

    out = {
        "last_train_year": int(last_train_year),
        "test_years": [int(y) for y in test_years],
        "best_params": dict(best_params),
        "seed": int(seed),
        "per_year": {},
    }

    all_true, all_pred = [], []
    for y in test_years:
        idx = test_idx_by_year[y]
        y_true = meta["imports_eur_thousands"].values[idx]
        y_pred = predict_levels(model, X_full.iloc[idx])
        out["per_year"][str(y)] = {**score_levels(y_true, y_pred), "n": int(len(idx))}
        all_true.append(y_true)
        all_pred.append(y_pred)

    if all_true:
        out["combined"] = {
            **score_levels(np.concatenate(all_true), np.concatenate(all_pred)),
            "n": int(sum(len(a) for a in all_true)),
        }

    if save_model_to is not None:
        import joblib
        Path(save_model_to).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "feature_list": list(X_full.columns),
             "best_params": dict(best_params), "seed": int(seed),
             "last_train_year": int(last_train_year)},
            save_model_to,
        )

    return out


# =============================================================================
# 6. PPML-Predictive on the same CV folds (for head-to-head scoring)
# =============================================================================

def ppml_predictive_cv(
    panel: pd.DataFrame,
    train_test_pairs: Iterable[Tuple[int, int]] = None,
    regressors: Iterable[str] = None,
    reference_iso2: str = "DE",
    restrict_to_xgb_rows: bool = True,
    return_predictions: bool = False,
):
    """Refit PPML-Predictive on each CV fold (train = year <= t, test = t+1)
    and score on the level scale. If restrict_to_xgb_rows is True, both the
    train and test rows are first reduced to the XGBoost-feasible row set
    (12-feature dropna) so the head-to-head with XGBoost is on identical
    rows.

    Returns
    -------
    metrics_df : pd.DataFrame  (one row per fold:
        last_train_year, test_year, n_train, n_test,
        ppml_rmse_eur_thousands, ppml_mae, ppml_r2)
    predictions_df : pd.DataFrame  (only if `return_predictions=True`,
        one row per (test_row × fold):
        last_train_year, test_year, partner_id, iso2, year,
        imports_eur_thousands, y_pred_ppml). Required by the orchestrator
        for pooled holdout R² (cannot aggregate per-year R²) and for the
        Phase-3 DM cv_prediction_errors.csv artefact.
    """
    import statsmodels.api as sm

    train_test_pairs = list(train_test_pairs or cfg.CV_FOLDS)
    regressors = list(regressors or cfg.PPML_PREDICTIVE_REGRESSORS)

    if restrict_to_xgb_rows:
        # Drop rows the 12-feature XGBoost panel would also drop, so train and
        # test sets line up exactly between models.
        feasible_mask = panel[cfg.FEATURE_ORDER].notna().all(axis=1)
        panel = panel.loc[feasible_mask].copy()

    rows = []
    pred_frames = [] if return_predictions else None
    for last_train_year, test_year in train_test_pairs:
        train = panel[panel["year"] <= last_train_year].dropna(subset=regressors).copy()
        test = panel[panel["year"] == test_year].dropna(subset=regressors).copy()
        if len(train) == 0 or len(test) == 0:
            rows.append({"last_train_year": last_train_year, "test_year": test_year,
                         "n_train": len(train), "n_test": len(test),
                         "ppml_rmse_eur_thousands": np.nan, "ppml_mae": np.nan,
                         "ppml_r2": np.nan})
            continue

        # Build train design matrix
        X_obs_tr = train[regressors].astype(float)
        X_fe_tr = pd.get_dummies(train["iso2"], prefix="pFE", drop_first=False).astype(float)
        ref_col = f"pFE_{reference_iso2}"
        if ref_col in X_fe_tr.columns:
            X_fe_tr = X_fe_tr.drop(columns=[ref_col])
        X_tr = pd.concat([X_obs_tr.reset_index(drop=True),
                          X_fe_tr.reset_index(drop=True)], axis=1)
        X_tr = sm.add_constant(X_tr, has_constant="add")

        y_tr = train["imports_eur_thousands"].astype(float).values
        model = sm.GLM(y_tr, X_tr, family=sm.families.Poisson())
        try:
            res = model.fit(
                cov_type="cluster",
                cov_kwds={"groups": train["partner_id"].astype(int).values},
            )
        except Exception:
            res = model.fit()

        # Build test design matrix with SAME columns / order as train
        X_obs_te = test[regressors].astype(float)
        X_fe_te = pd.get_dummies(test["iso2"], prefix="pFE", drop_first=False).astype(float)
        # Add any train FE columns missing in test, drop any test-only columns
        for col in X_fe_tr.columns:
            if col not in X_fe_te.columns:
                X_fe_te[col] = 0.0
        X_fe_te = X_fe_te[X_fe_tr.columns]
        X_te = pd.concat([X_obs_te.reset_index(drop=True),
                          X_fe_te.reset_index(drop=True)], axis=1)
        X_te = sm.add_constant(X_te, has_constant="add")
        # Re-order to match training column order exactly
        X_te = X_te[X_tr.columns]

        y_pred_lvl = np.asarray(res.predict(X_te), dtype=float)
        y_true_lvl = test["imports_eur_thousands"].astype(float).values
        sc = score_levels(y_true_lvl, y_pred_lvl)
        rows.append({
            "last_train_year": int(last_train_year),
            "test_year": int(test_year),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "ppml_rmse_eur_thousands": sc["rmse"],
            "ppml_mae": sc["mae"],
            "ppml_r2": sc["r2"],
        })
        if return_predictions:
            pred_frames.append(pd.DataFrame({
                "last_train_year": int(last_train_year),
                "test_year":       int(test_year),
                "partner_id":      test["partner_id"].astype(int).values,
                "iso2":            test["iso2"].values,
                "year":            test["year"].astype(int).values,
                "imports_eur_thousands": y_true_lvl,
                "y_pred_ppml":     y_pred_lvl,
            }))

    metrics_df = pd.DataFrame(rows)
    if return_predictions:
        predictions_df = (pd.concat(pred_frames, ignore_index=True)
                          if pred_frames else
                          pd.DataFrame(columns=[
                              "last_train_year", "test_year", "partner_id",
                              "iso2", "year", "imports_eur_thousands",
                              "y_pred_ppml",
                          ]))
        return metrics_df, predictions_df
    return metrics_df


# =============================================================================
# 7. XGBoost on the same CV folds (paired with the PPML CV above)
# =============================================================================

def xgb_cv(
    panel: pd.DataFrame,
    best_params: dict,
    train_test_pairs: Iterable[Tuple[int, int]] = None,
    seed: int = cfg.SEED,
    return_predictions: bool = False,
):
    """Fit XGBoost with `best_params` on each CV fold and score on the level
    scale.

    Returns
    -------
    metrics_df : pd.DataFrame  (one row per fold:
        last_train_year, test_year, n_train, n_test,
        xgb_rmse_eur_thousands, xgb_mae, xgb_r2)
    predictions_df : pd.DataFrame  (only if `return_predictions=True`,
        same long-row schema as ppml_predictive_cv but with `y_pred_xgb`).
    """
    train_test_pairs = list(train_test_pairs or cfg.CV_FOLDS)
    X_full, y_full, meta = build_xgb_xy(panel)

    rows = []
    pred_frames = [] if return_predictions else None
    for tr_idx, te_idx, last_train_year, test_year in expanding_cv_folds(meta, train_test_pairs):
        model = fit_xgb(X_full.iloc[tr_idx], y_full[tr_idx], best_params, seed=seed)
        y_pred = predict_levels(model, X_full.iloc[te_idx])
        y_true = meta["imports_eur_thousands"].values[te_idx]
        sc = score_levels(y_true, y_pred)
        rows.append({
            "last_train_year": int(last_train_year),
            "test_year": int(test_year),
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "xgb_rmse_eur_thousands": sc["rmse"],
            "xgb_mae": sc["mae"],
            "xgb_r2": sc["r2"],
        })
        if return_predictions:
            te_meta = meta.iloc[te_idx]
            pred_frames.append(pd.DataFrame({
                "last_train_year": int(last_train_year),
                "test_year":       int(test_year),
                "partner_id":      te_meta["partner_id"].astype(int).values,
                "iso2":            te_meta["iso2"].values,
                "year":            te_meta["year"].astype(int).values,
                "imports_eur_thousands": y_true,
                "y_pred_xgb":      y_pred,
            }))

    metrics_df = pd.DataFrame(rows)
    if return_predictions:
        predictions_df = (pd.concat(pred_frames, ignore_index=True)
                          if pred_frames else
                          pd.DataFrame(columns=[
                              "last_train_year", "test_year", "partner_id",
                              "iso2", "year", "imports_eur_thousands",
                              "y_pred_xgb",
                          ]))
        return metrics_df, predictions_df
    return metrics_df


# =============================================================================
# 8. Seed-stability check (Phase 2 report §5)
# =============================================================================

def seed_stability(
    best_params: dict,
    panel: pd.DataFrame,
    seeds: Iterable[int] = (42, 142, 242),
    last_train_year: int = None,
    test_years: Iterable[int] = None,
) -> dict:
    """Re-fit the holdout XGBoost with each seed and report RMSE/MAE
    dispersion across seeds. Confirms results are not seed-dependent.
    """
    if last_train_year is None or test_years is None:
        last_train_year, test_years = cfg.CV_HOLDOUT

    rmses, maes = [], []
    per_seed = {}
    for s in seeds:
        h = fit_holdout(best_params, panel, seed=int(s),
                        last_train_year=last_train_year, test_years=test_years)
        rmses.append(h["combined"]["rmse"])
        maes.append(h["combined"]["mae"])
        per_seed[str(s)] = h["combined"]

    return {
        "seeds": [int(s) for s in seeds],
        "per_seed_combined": per_seed,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std":  float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "rmse_min":  float(np.min(rmses)),
        "rmse_max":  float(np.max(rmses)),
        "mae_mean":  float(np.mean(maes)),
        "mae_std":   float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
    }
