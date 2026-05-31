"""Interpretation pillar — SHAP, ablation, Diebold-Mariano.

PROTOCOL_FREEZE.md §6, §7, §13 are load-bearing for this module.

Critical binding rules respected here:

- SHAP values are kept in **model output space = log1p(EUR thousands)**.
  Never back-transform via `expm1` (the back-transform is non-linear in
  the sum of contributions). Every figure / column header / docstring
  states "SHAP value (model output units: log1p(EUR thousands))".
- SHAP values are described as **predictive contributions** or
  **associative importance**, never as **effects** (PROTOCOL_FREEZE
  §13 #2). The summary DataFrame's column is named `mean_shap_sign`,
  not "direction-of-effect".
- Ablation **always** uses the full 12-feature design matrix from
  `xgb_model.build_xgb_xy(panel)` and only **subsets columns** per
  layer. Calling `build_xgb_xy` with fewer features would dropna on a
  smaller column set and admit the 113 lagged-2010 rows back into
  L1/L2/L3, breaking the 1,559-row identical-rowset comparison vs L4.
- The canonical `models/xgb_best.joblib` is **read-only** in this
  module. Ablation fits **temporary, in-memory** boosters with the same
  `best_params` on different feature subsets — those are never saved.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from . import config as cfg
from . import xgb_model as xgb


# =============================================================================
# 1. Saved-model loader
# =============================================================================

def load_xgb_best() -> dict:
    """Load `models/xgb_best.joblib`.

    Returns the dict persisted by `xgb_model.fit_holdout`:
        {model, feature_list, best_params, seed, last_train_year}
    """
    import joblib
    p = cfg.MODELS / "xgb_best.joblib"
    if not p.exists():
        raise FileNotFoundError(
            f"xgb_best.joblib not found at {p}. "
            "Run `python -m src.run_phase2` first."
        )
    return joblib.load(p)


# =============================================================================
# 2. SHAP — global and Serbia-specific
# =============================================================================

def compute_shap(model, X) -> Tuple[np.ndarray, float]:
    """Run TreeExplainer on the given XGBoost model.

    Returns
    -------
    shap_values : np.ndarray of shape (n_rows, n_features)
        SHAP values in MODEL OUTPUT SPACE = log1p(EUR thousands).
    expected_value : float
        Scalar baseline for the regression. SHAP can return this as a
        scalar OR an array-like (e.g. ndarray of length 1, or a list);
        we coerce to a scalar via `np.asarray(...).reshape(-1)[0]`.

    The caller may use these to verify additivity:
        log1p_pred ≈ expected_value + shap_values.sum(axis=1)
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:
        # XGBoost wrapper sometimes confuses SHAP — retry with the booster
        explainer = shap.TreeExplainer(model.get_booster())
        shap_values = explainer.shap_values(X)

    # `expected_value` may be a scalar, a 0-d array, a 1-d array of length 1,
    # or a list. Coerce robustly.
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    return np.asarray(shap_values, dtype=float), expected_value


def shap_global_summary(shap_values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SHAP values across the panel into a per-feature summary.

    Returns a DataFrame sorted DESC by `mean_abs_shap` with columns:
        feature, mean_shap, mean_abs_shap, mean_shap_sign

    `mean_shap_sign` is 'positive' / 'negative' / 'zero' — never the
    word "effect" or "direction-of-effect" (PROTOCOL_FREEZE §13 #2).
    """
    if shap_values.shape[1] != X.shape[1]:
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} features but X has {X.shape[1]}"
        )
    rows = []
    for i, feature in enumerate(X.columns):
        col = shap_values[:, i]
        m = float(np.mean(col))
        rows.append({
            "feature":        feature,
            "mean_shap":      m,
            "mean_abs_shap":  float(np.mean(np.abs(col))),
            "mean_shap_sign": ("positive" if m > 0 else
                               "negative" if m < 0 else "zero"),
        })
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return out


def shap_for_serbia(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter SHAP values to Serbia (iso2 == 'XS') and return both views.

    Returns
    -------
    serbia_long : pd.DataFrame  (14 × 12 = 168 rows)
        columns: year, feature, shap_value
    serbia_wide : pd.DataFrame  (14 rows, 12 feature columns; index = year)

    The 2010 Serbia row is excluded from the XGBoost-feasible set because
    `lagged_imports_log1p` is NaN there. We assert that exactly 14 rows
    remain (years 2011-2024) so a feasibility-filter regression is caught
    immediately.
    """
    if not (len(X) == len(meta) == len(shap_values)):
        raise ValueError(
            f"Length mismatch: X={len(X)}, meta={len(meta)}, "
            f"shap_values={len(shap_values)}"
        )
    serbia_idx = np.where(meta["iso2"].values == "XS")[0]
    if len(serbia_idx) != 14:
        raise AssertionError(
            f"Expected 14 Serbia rows (2011-2024) in the XGBoost-feasible "
            f"panel; got {len(serbia_idx)}. "
            f"Years observed: {sorted(meta['year'].values[serbia_idx].tolist())}"
        )

    feats = list(X.columns)
    years = meta["year"].values[serbia_idx].astype(int)
    sv = shap_values[serbia_idx]                        # (14, 12)

    serbia_wide = pd.DataFrame(sv, columns=feats, index=pd.Index(years, name="year"))
    long_rows = []
    for r, y in enumerate(years):
        for c, f in enumerate(feats):
            long_rows.append({
                "year": int(y),
                "feature": f,
                "shap_value": float(sv[r, c]),
            })
    serbia_long = pd.DataFrame(long_rows)
    return serbia_long, serbia_wide


# =============================================================================
# 3. Ablation — CV and holdout, identical-rowset
# =============================================================================

def ablation_cv(
    panel: pd.DataFrame,
    best_params: dict,
    layers: dict = None,
    folds: Iterable[Tuple[int, int]] = None,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """4-layer ablation on the same 5 CV folds Phase 2 used.

    Identical-rowset construction (PROTOCOL_FREEZE-compliant):
        X_full, y, meta = xgb_model.build_xgb_xy(panel)   # 1,559 rows
        For each layer L: X_layer = X_full[layers[L]]      # column subset only

    Refits XGBoost with the SAME `best_params` per layer × fold (no
    Optuna re-run). The resulting boosters are temporary and never
    saved — `models/xgb_best.joblib` is untouched.

    Returns
    -------
    pd.DataFrame  (one row per (layer, fold) plus per-layer mean rows)
        Columns: layer, n_features, fold_idx, last_train_year, test_year,
                 n_train, n_test, rmse, mae, r2
    """
    layers = layers or cfg.ABLATION_LAYERS
    folds = list(folds or cfg.CV_FOLDS)

    X_full, y, meta = xgb.build_xgb_xy(panel)
    if len(X_full) != cfg.EXPECTED_N - 136:
        # 1,695 − 136 = 1,559; if it ever shifts we want to know loudly
        raise AssertionError(
            f"build_xgb_xy returned {len(X_full)} rows; expected 1,559 "
            f"(1,695 − 136 dropna)."
        )

    rows = []
    for layer_name, feats in layers.items():
        # Subset COLUMNS only — never re-call build_xgb_xy with fewer features
        X_layer = X_full[list(feats)]
        if X_layer.shape[1] != len(feats):
            raise AssertionError(
                f"Layer {layer_name} expected {len(feats)} feature columns "
                f"but only {X_layer.shape[1]} were resolved from X_full."
            )
        for fi, (tr_idx, te_idx, last_train_year, test_year) in enumerate(
            xgb.expanding_cv_folds(meta, folds)
        ):
            model_L = xgb.fit_xgb(X_layer.iloc[tr_idx], y[tr_idx], best_params, seed=seed)
            y_pred = xgb.predict_levels(model_L, X_layer.iloc[te_idx])
            y_true = meta["imports_eur_thousands"].values[te_idx]
            sc = xgb.score_levels(y_true, y_pred)
            rows.append({
                "layer":           layer_name,
                "n_features":      X_layer.shape[1],
                "fold_idx":        int(fi),
                "last_train_year": int(last_train_year),
                "test_year":       int(test_year),
                "n_train":         int(len(tr_idx)),
                "n_test":          int(len(te_idx)),
                "rmse":            sc["rmse"],
                "mae":             sc["mae"],
                "r2":              sc["r2"],
            })

    df = pd.DataFrame(rows)

    # Append per-layer means
    mean_rows = []
    for layer_name in layers:
        sub = df[df["layer"] == layer_name]
        mean_rows.append({
            "layer":           layer_name,
            "n_features":      int(sub["n_features"].iloc[0]),
            "fold_idx":        "mean",
            "last_train_year": np.nan,
            "test_year":       np.nan,
            "n_train":         np.nan,
            "n_test":          np.nan,
            "rmse":            float(sub["rmse"].mean()),
            "mae":             float(sub["mae"].mean()),
            "r2":              float(sub["r2"].mean()),
        })
    return pd.concat([df, pd.DataFrame(mean_rows)], ignore_index=True)


def ablation_holdout(
    panel: pd.DataFrame,
    best_params: dict,
    layers: dict = None,
    last_train_year: int = None,
    test_years: Iterable[int] = None,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """4-layer ablation on the 2023+2024 holdout.

    Same identical-rowset construction as `ablation_cv`. Combined RMSE /
    MAE / R² computed from POOLED predictions on the union of test years
    (cannot aggregate per-year R² — uses the per-year y_true means).

    Returns
    -------
    pd.DataFrame  (one row per layer)
        Columns: layer, n_features, n_train, n_test, rmse, mae, r2
    """
    layers = layers or cfg.ABLATION_LAYERS
    if last_train_year is None or test_years is None:
        last_train_year, test_years = cfg.CV_HOLDOUT
    test_years = list(test_years)

    X_full, y, meta = xgb.build_xgb_xy(panel)
    train_idx, test_idx_by_year = xgb.holdout_split(meta, last_train_year, test_years)
    test_idx_combined = np.concatenate([test_idx_by_year[ty] for ty in test_years])

    rows = []
    for layer_name, feats in layers.items():
        X_layer = X_full[list(feats)]
        model_L = xgb.fit_xgb(X_layer.iloc[train_idx], y[train_idx], best_params, seed=seed)
        y_pred = xgb.predict_levels(model_L, X_layer.iloc[test_idx_combined])
        y_true = meta["imports_eur_thousands"].values[test_idx_combined]
        sc = xgb.score_levels(y_true, y_pred)
        rows.append({
            "layer":      layer_name,
            "n_features": X_layer.shape[1],
            "n_train":    int(len(train_idx)),
            "n_test":     int(len(test_idx_combined)),
            "rmse":       sc["rmse"],
            "mae":        sc["mae"],
            "r2":         sc["r2"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 4. Diebold-Mariano with HLN small-sample correction
# =============================================================================

def diebold_mariano_hln(d: np.ndarray, h: int = 1) -> dict:
    """DM test on a row-level loss differential `d` with HLN correction.

    Sign convention:
        d[i] = squared_error_PPML[i] − squared_error_XGB[i]
        positive mean ⇒ PPML's squared error is larger ⇒ XGB favoured.

    Algorithm (Diebold-Mariano 1995, Harvey-Leybourne-Newbold 1997):
        n        = len(d)
        d̄        = d.mean()
        var_d    = d.var(ddof=1)
        DM       = d̄ / sqrt(var_d / n)
        HLN(h=1) = sqrt((n + 1 − 2·h + h·(h−1)/n) / n)
                 = sqrt((n − 1) / n)
        DM_HLN   = DM × HLN
        df       = n − 1
        p        = 2 · (1 − t.cdf(|DM_HLN|, df))   # two-sided

    Interpretation (α = 0.05):
        p < 0.05  AND  d̄ > 0   → "XGB significantly better"
        p < 0.05  AND  d̄ < 0   → "PPML significantly better"
        p ≥ 0.05                → "no significant difference at α=0.05"

    Returns a dict with all of the above plus interpretation string.
    """
    from scipy.stats import t as student_t

    d = np.asarray(d, dtype=float)
    n = int(len(d))
    if n < 3:
        raise ValueError(f"DM test needs at least 3 observations; got {n}")
    if not np.isfinite(d).all():
        raise ValueError("DM input contains NaN or inf values")

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 0:
        raise ValueError("DM input has zero variance — cannot test")

    dm = mean_d / float(np.sqrt(var_d / n))
    hln_factor = float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
    dm_hln = dm * hln_factor
    df = n - 1
    p_value = float(2.0 * (1.0 - student_t.cdf(abs(dm_hln), df=df)))

    if p_value < 0.05 and mean_d > 0:
        interpretation = "XGB significantly better"
    elif p_value < 0.05 and mean_d < 0:
        interpretation = "PPML significantly better"
    else:
        interpretation = "no significant difference at α=0.05"

    return {
        "n":              n,
        "h":              int(h),
        "mean_loss_diff": mean_d,
        "var_loss_diff":  var_d,
        "dm":             float(dm),
        "dm_hln":         float(dm_hln),
        "hln_factor":     hln_factor,
        "df":             int(df),
        "p_value":        p_value,
        "alpha":          0.05,
        "sign_convention": "loss_diff = sq_err_PPML − sq_err_XGB; "
                           "positive ⇒ XGB favoured",
        "interpretation": interpretation,
    }
