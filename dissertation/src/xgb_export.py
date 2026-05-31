"""Export-side predictive ML for Phase 2.5.

Methodological-transferability complement to the import-side
predictive pillar (Phase 2 v2 + Phase 3 SHAP / DM). XGBoost is reused
as the pre-specified ML benchmark; PPML remains the econometric
benchmark. The same Optuna search space, same 5 CV folds, same
2023-2024 holdout, and same seed=42 are applied to the export
target. A naive persistence baseline (exports_t = exports_{t-1}) is
added as a sanity floor.

This module does not modify any Phase 0/1/1.5/1-5 module. It reuses
xgb_model.build_xgb_xy, xgb_model.optuna_search,
xgb_model.ppml_predictive_cv, xgb_model.xgb_cv,
xgb_model.fit_holdout, xgb_model.xgb_train_test_r2 verbatim via a
three-column swap on a parallel export panel:

  imports_eur_thousands      <- exports_eur_thousands
  lagged_imports_log1p       <- lagged_exports_log1p (computed inline)
  partner_import_share_lag   <- partner_export_share_lag (computed inline)

Originals are preserved as *_actual columns for audit. Display labels
in SHAP tables, comparison tables, figures, and report prose all
carry export-side names; the column-swap is internal mechanics only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as cfg
from . import eda, xgb_model


EXPORT_LABEL_MAP = {
    "imports_eur_thousands":    "exports_eur_thousands",
    "lagged_imports_log1p":     "lagged_exports_log1p",
    "partner_import_share_lag": "partner_export_share_lag",
}


# ---------------------------------------------------------------------------
# Step 2 — build_export_panel
# ---------------------------------------------------------------------------

def build_export_panel() -> pd.DataFrame:
    """Build the 1,695-row export-side parallel panel.

    Loads the canonical import-side panel and joins ASK exports for
    the same 113 partners. Computes the two export-side lag features
    inline (mirroring features.add_lagged_features without modifying
    it). Swaps three flow-specific columns; preserves originals as
    *_actual; saves the result to panel_bilateral_export.parquet
    (gitignored, regenerable).
    """
    panel_imp = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    if len(panel_imp) != 1695:
        raise ValueError(f"import panel rows: {len(panel_imp)} != 1695")

    top_113 = sorted(panel_imp["iso2"].unique())
    ek = eda.parse_partner_full()
    ek_exp = ek[(ek["iso2"].isin(top_113)) & (ek["flow"] == "export")].copy()
    if len(ek_exp) != 113 * 15:
        raise ValueError(f"ASK export rows: {len(ek_exp)} != 1695")

    ek_exp = ek_exp[["iso2", "year", "value_eur_thousands"]].rename(
        columns={"value_eur_thousands": "exports_eur_thousands"}
    )

    ek_exp = ek_exp.sort_values(["iso2", "year"]).reset_index(drop=True)
    ek_exp["lagged_exports_log1p"] = np.log1p(
        ek_exp.groupby("iso2")["exports_eur_thousands"].shift(1).fillna(0)
    )
    ek_exp.loc[ek_exp["year"] == min(cfg.YEARS), "lagged_exports_log1p"] = np.nan
    ek_exp["prev_exports"] = ek_exp.groupby("iso2")["exports_eur_thousands"].shift(1)
    year_totals = ek_exp.groupby("year")["prev_exports"].transform("sum")
    ek_exp["partner_export_share_lag"] = ek_exp["prev_exports"] / year_totals
    ek_exp.loc[ek_exp["year"] == min(cfg.YEARS), "partner_export_share_lag"] = np.nan
    ek_exp = ek_exp.drop(columns=["prev_exports"])

    merged = panel_imp.merge(
        ek_exp[["iso2", "year", "exports_eur_thousands",
                "lagged_exports_log1p", "partner_export_share_lag"]],
        on=["iso2", "year"], how="left", validate="one_to_one",
    )
    if len(merged) != 1695:
        raise ValueError(f"merge changed row count: {len(merged)} != 1695")
    if merged["exports_eur_thousands"].isna().any():
        raise ValueError("exports_eur_thousands has NaNs after merge")

    swap = merged.copy()
    swap["imports_eur_thousands_actual"] = swap["imports_eur_thousands"]
    swap["lagged_imports_log1p_actual"] = swap["lagged_imports_log1p"]
    swap["partner_import_share_lag_actual"] = swap["partner_import_share_lag"]
    swap["imports_eur_thousands"] = swap["exports_eur_thousands"]
    swap["lagged_imports_log1p"] = swap["lagged_exports_log1p"]
    swap["partner_import_share_lag"] = swap["partner_export_share_lag"]

    # Hard audit assertions
    assert (swap["imports_eur_thousands"] == swap["exports_eur_thousands"]).all(), \
        "Swap audit: imports_eur_thousands != exports_eur_thousands"
    assert (swap["imports_eur_thousands_actual"] == merged["imports_eur_thousands"]).all(), \
        "Swap audit: imports_actual != original imports"
    eq = (swap["lagged_imports_log1p"].fillna(-999) ==
          swap["lagged_exports_log1p"].fillna(-999)).all()
    assert eq, "Swap audit: lagged_imports_log1p != lagged_exports_log1p"
    eq = (swap["partner_import_share_lag"].fillna(-999) ==
          swap["partner_export_share_lag"].fillna(-999)).all()
    assert eq, "Swap audit: partner_import_share_lag != partner_export_share_lag"

    for col in ("imports_eur_thousands", "lagged_imports_log1p",
                "partner_import_share_lag"):
        n_swap = swap[col].isna().sum()
        if col == "imports_eur_thousands":
            assert n_swap == 0, f"swap introduced {n_swap} NaNs in target"
        else:
            assert n_swap == 113, f"{col} has {n_swap} NaNs after swap (expected 113 = 2010 rows)"

    out_path = cfg.DATA_PROCESSED / "panel_bilateral_export.parquet"
    swap.to_parquet(out_path, index=False)
    return swap


# ---------------------------------------------------------------------------
# Step 3 — run_export_optuna
# ---------------------------------------------------------------------------

def run_export_optuna(panel_export: pd.DataFrame,
                       n_trials: int = None,
                       seed: int = None):
    """100-trial TPE search on the export-swapped panel."""
    n_trials = int(n_trials or cfg.OPTUNA_TRIALS)
    seed = int(seed if seed is not None else cfg.SEED)
    X, y, meta = xgb_model.build_xgb_xy(panel_export)
    result = xgb_model.optuna_search(
        X, y, meta,
        n_trials=n_trials,
        seed=seed,
        folds=cfg.CV_FOLDS,
        persist_csv=cfg.METRICS / "optuna_trials_export.csv",
    )
    return result


# ---------------------------------------------------------------------------
# Step 4 — run_export_cv_head_to_head
# ---------------------------------------------------------------------------

def run_export_cv_head_to_head(panel_export: pd.DataFrame,
                                 best_params: dict,
                                 seed: int = None) -> dict:
    """Paired CV: PPML-Predictive vs XGBoost on identical export-feasible rows."""
    seed = int(seed if seed is not None else cfg.SEED)

    ppml_metrics, ppml_pred = xgb_model.ppml_predictive_cv(
        panel_export,
        train_test_pairs=cfg.CV_FOLDS,
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=True,
        return_predictions=True,
    )
    xgb_metrics, xgb_pred = xgb_model.xgb_cv(
        panel_export,
        best_params=best_params,
        train_test_pairs=cfg.CV_FOLDS,
        seed=seed,
        return_predictions=True,
    )

    per_fold = ppml_metrics.merge(
        xgb_metrics[["last_train_year", "test_year",
                     "xgb_rmse_eur_thousands", "xgb_mae", "xgb_r2"]],
        on=["last_train_year", "test_year"], how="inner",
    )
    per_fold["winner"] = np.where(
        per_fold["xgb_rmse_eur_thousands"] < per_fold["ppml_rmse_eur_thousands"],
        "xgb", "ppml",
    )

    merged_pred = ppml_pred.merge(
        xgb_pred[["last_train_year", "test_year", "partner_id", "iso2", "year",
                  "y_pred_xgb"]],
        on=["last_train_year", "test_year", "partner_id", "iso2", "year"],
        how="inner",
    ).rename(columns={"imports_eur_thousands": "y_true_eur_thousands"})
    merged_pred["sq_err_ppml"] = (merged_pred["y_pred_ppml"] - merged_pred["y_true_eur_thousands"]) ** 2
    merged_pred["sq_err_xgb"] = (merged_pred["y_pred_xgb"] - merged_pred["y_true_eur_thousands"]) ** 2
    merged_pred["loss_diff_ppml_minus_xgb"] = merged_pred["sq_err_ppml"] - merged_pred["sq_err_xgb"]
    merged_pred.insert(0, "fold", merged_pred["last_train_year"])
    cols = ["fold", "last_train_year", "test_year", "partner_id", "iso2", "year",
            "y_true_eur_thousands", "y_pred_ppml", "y_pred_xgb",
            "sq_err_ppml", "sq_err_xgb", "loss_diff_ppml_minus_xgb"]
    merged_pred = merged_pred[cols]

    return {"per_fold": per_fold, "per_row": merged_pred}


# ---------------------------------------------------------------------------
# Step 5 — run_export_holdout
# ---------------------------------------------------------------------------

def run_export_holdout(panel_export: pd.DataFrame,
                        best_params: dict,
                        seed: int = None) -> dict:
    """Final 2023+2024 holdout for the export pillar."""
    seed = int(seed if seed is not None else cfg.SEED)

    booster_path = cfg.MODELS / "xgb_best_export.joblib"
    xgb_holdout = xgb_model.fit_holdout(
        best_params,
        panel_export,
        seed=seed,
        save_model_to=booster_path,
    )
    # Augment the saved bundle with export-side display labels so
    # downstream consumers do not accidentally surface the internal
    # import-side feature names.
    bundle = joblib.load(booster_path)
    bundle["display_feature_list"] = [
        EXPORT_LABEL_MAP.get(c, c) for c in bundle["feature_list"]
    ]
    joblib.dump(bundle, booster_path)

    ppml_metrics_2023, ppml_pred_2023 = xgb_model.ppml_predictive_cv(
        panel_export,
        train_test_pairs=[(2022, 2023)],
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=True,
        return_predictions=True,
    )
    ppml_metrics_2024, ppml_pred_2024 = xgb_model.ppml_predictive_cv(
        panel_export,
        train_test_pairs=[(2022, 2024)],
        regressors=cfg.PPML_PREDICTIVE_REGRESSORS,
        restrict_to_xgb_rows=True,
        return_predictions=True,
    )

    all_pred = pd.concat([ppml_pred_2023, ppml_pred_2024], ignore_index=True)
    sc = xgb_model.score_levels(
        all_pred["imports_eur_thousands"].values, all_pred["y_pred_ppml"].values
    )

    return {
        "xgboost": xgb_holdout,
        "ppml_predictive_per_year": (
            ppml_metrics_2023.to_dict(orient="records") +
            ppml_metrics_2024.to_dict(orient="records")
        ),
        "ppml_predictive_combined": {**sc},
    }


# ---------------------------------------------------------------------------
# Step 6 — run_export_persistence_baseline
# ---------------------------------------------------------------------------

def run_export_persistence_baseline(panel_export: pd.DataFrame) -> dict:
    """Persistence baseline: predicted exports_t = exports_{t-1}.

    Scored on the same row sets XGBoost uses: the 5 CV test folds and
    the 2023+2024 holdout. Back-transforms the swapped
    lagged_imports_log1p column via expm1 to recover the lagged
    export levels.
    """
    X, y, meta = xgb_model.build_xgb_xy(panel_export)
    lagged_levels = np.expm1(X["lagged_imports_log1p"].values)
    y_true_levels = meta["imports_eur_thousands"].values

    out = {}

    cv_test_idx = []
    for tr_idx, te_idx, _, _ in xgb_model.expanding_cv_folds(meta, cfg.CV_FOLDS):
        cv_test_idx.append(te_idx)
    cv_test_idx = np.unique(np.concatenate(cv_test_idx))
    sc = xgb_model.score_levels(
        y_true_levels[cv_test_idx], lagged_levels[cv_test_idx]
    )
    out["cv"] = {
        "n": int(len(cv_test_idx)),
        "rmse_eur_thousands": float(sc["rmse"]),
        "mae_eur_thousands": float(sc["mae"]),
        "r2": float(sc["r2"]),
    }

    _, test_idx_by_year = xgb_model.holdout_split(meta, *cfg.CV_HOLDOUT)
    holdout_combined_idx = np.concatenate(
        [test_idx_by_year[y] for y in cfg.CV_HOLDOUT[1]]
    )
    sc = xgb_model.score_levels(
        y_true_levels[holdout_combined_idx], lagged_levels[holdout_combined_idx]
    )
    out["holdout_combined"] = {
        "n": int(len(holdout_combined_idx)),
        "rmse_eur_thousands": float(sc["rmse"]),
        "mae_eur_thousands": float(sc["mae"]),
        "r2": float(sc["r2"]),
    }
    for yr in cfg.CV_HOLDOUT[1]:
        idx = test_idx_by_year[yr]
        sc = xgb_model.score_levels(
            y_true_levels[idx], lagged_levels[idx]
        )
        out[f"holdout_{yr}"] = {
            "n": int(len(idx)),
            "rmse_eur_thousands": float(sc["rmse"]),
            "mae_eur_thousands": float(sc["mae"]),
            "r2": float(sc["r2"]),
        }
    return out


# ---------------------------------------------------------------------------
# Step 7 — run_export_dm_test
# ---------------------------------------------------------------------------

def run_export_dm_test(cv_prediction_errors_export_path: Path,
                        alpha: float = 0.05) -> dict:
    """Diebold-Mariano test on pooled export CV errors, HLN-corrected at h=1."""
    from scipy.stats import t as student_t

    df = pd.read_csv(cv_prediction_errors_export_path)
    d = df["loss_diff_ppml_minus_xgb"].values
    n = int(len(d))
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    dm = mean_d / np.sqrt(var_d / n) if var_d > 0 else float("nan")
    hln_factor = float(np.sqrt((n - 1) / n))
    dm_hln = dm * hln_factor
    df_t = n - 1
    p_value = 2.0 * (1.0 - student_t.cdf(abs(dm_hln), df=df_t))

    interp = ("XGB significantly outperforms PPML" if (dm_hln > 0 and p_value < alpha) else
              ("PPML significantly outperforms XGB" if (dm_hln < 0 and p_value < alpha) else
               f"no significant difference at α={alpha}"))
    return {
        "n": n,
        "h": 1,
        "mean_loss_diff": mean_d,
        "var_loss_diff": var_d,
        "dm": float(dm),
        "dm_hln": float(dm_hln),
        "hln_factor": hln_factor,
        "df": int(df_t),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "sign_convention": "loss_diff = sq_err_PPML - sq_err_XGB; positive => XGB favoured",
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Step 8 — compute_export_shap_global
# ---------------------------------------------------------------------------

def compute_export_shap_global(panel_export: pd.DataFrame,
                                booster_path: Path,
                                out_stem: str = "fig_ch4_shap_bar_export",
                                table_stem: str = "tbl_ch4_shap_global_export"):
    """Global SHAP ranking on the export booster.

    Applies the export-side label translation before persisting the
    table and figure; SHAP values stay in log1p output space (same
    convention as Phase 3).
    """
    import shap
    import matplotlib.pyplot as plt

    bundle = joblib.load(booster_path)
    model = bundle["model"]

    X, y, meta = xgb_model.build_xgb_xy(panel_export)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    abs_shap = np.abs(shap_values)
    mean_abs = abs_shap.mean(axis=0)
    mean_shap = shap_values.mean(axis=0)
    sign = np.sign(mean_shap)
    sign_label = pd.Series(sign, index=X.columns).map(
        {1.0: "positive", -1.0: "negative", 0.0: "zero"}
    )

    display_feature = pd.Series(X.columns).replace(EXPORT_LABEL_MAP)

    summary = pd.DataFrame({
        "feature": display_feature.values,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_shap,
        "mean_shap_sign": sign_label.values,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    summary.to_csv(cfg.TABLES / f"{table_stem}.csv", index=False)
    with open(cfg.TABLES / f"{table_stem}.tex", "w") as f:
        f.write(summary.to_latex(index=False, float_format="%.4f", escape=True))

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 6))
    top = summary.head(12).iloc[::-1]
    colors = ["#c0392b" if s == "negative" else "#16a085" for s in top["mean_shap_sign"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value| (log1p output space)")
    ax.set_title("Export model - global SHAP feature ranking")
    fig.tight_layout()
    fig.savefig(cfg.FIGURES / f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(cfg.FIGURES / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    joblib.dump({
        "shap_values": shap_values,
        "feature_order": list(X.columns),
        "display_feature_order": list(display_feature.values),
        "expected_value": float(explainer.expected_value),
    }, cfg.MODELS / "shap_values_export.joblib")
    return summary


# ---------------------------------------------------------------------------
# Step 9 — build_import_vs_export_prediction_comparison
# ---------------------------------------------------------------------------

def _persistence_inline(panel: pd.DataFrame) -> dict:
    """Inline import-side persistence baseline for the comparison table."""
    X, y, meta = xgb_model.build_xgb_xy(panel)
    lagged_levels = np.expm1(X["lagged_imports_log1p"].values)
    y_true_levels = meta["imports_eur_thousands"].values
    cv_test_idx = []
    for tr_idx, te_idx, _, _ in xgb_model.expanding_cv_folds(meta, cfg.CV_FOLDS):
        cv_test_idx.append(te_idx)
    cv_test_idx = np.unique(np.concatenate(cv_test_idx))
    sc_cv = xgb_model.score_levels(
        y_true_levels[cv_test_idx], lagged_levels[cv_test_idx]
    )
    _, test_idx_by_year = xgb_model.holdout_split(meta, *cfg.CV_HOLDOUT)
    holdout_idx = np.concatenate(
        [test_idx_by_year[y] for y in cfg.CV_HOLDOUT[1]]
    )
    sc_ho = xgb_model.score_levels(
        y_true_levels[holdout_idx], lagged_levels[holdout_idx]
    )
    return {
        "cv": {"n": int(len(cv_test_idx)),
               "rmse_eur_thousands": float(sc_cv["rmse"]),
               "mae_eur_thousands": float(sc_cv["mae"]),
               "r2": float(sc_cv["r2"])},
        "holdout_combined": {"n": int(len(holdout_idx)),
                             "rmse_eur_thousands": float(sc_ho["rmse"]),
                             "mae_eur_thousands": float(sc_ho["mae"]),
                             "r2": float(sc_ho["r2"])},
    }


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def build_import_vs_export_prediction_comparison(
    panel_import: pd.DataFrame,
    panel_export: pd.DataFrame,
    export_best_params: dict,
    export_persistence: dict,
    out_stem: str = "tbl_ch4_import_vs_export_prediction",
) -> pd.DataFrame:
    """Build the long comparison table (level + normalised + statistical)."""
    cv_imp = pd.read_csv(cfg.METRICS / "cv_xgb_vs_ppml.csv")
    cv_pred_imp = pd.read_csv(cfg.METRICS / "cv_prediction_errors.csv")
    with open(cfg.METRICS / "holdout_2023_2024.json") as f:
        ho_imp = json.load(f)
    with open(cfg.METRICS / "dm_test.json") as f:
        dm_imp = json.load(f)
    with open(cfg.METRICS / "optuna_best_params.json") as f:
        opt_imp = json.load(f)

    cv_imp_mean = cv_imp[["ppml_rmse_eur_thousands", "ppml_mae", "ppml_r2",
                          "xgb_rmse_eur_thousands", "xgb_mae", "xgb_r2"]].mean()
    persist_imp = _persistence_inline(panel_import)

    cv_exp = pd.read_csv(cfg.METRICS / "cv_xgb_vs_ppml_export.csv")
    cv_pred_exp = pd.read_csv(cfg.METRICS / "cv_prediction_errors_export.csv")
    with open(cfg.METRICS / "holdout_export_2023_2024.json") as f:
        ho_exp = json.load(f)
    with open(cfg.METRICS / "dm_test_export.json") as f:
        dm_exp = json.load(f)
    with open(cfg.METRICS / "optuna_best_params_export.json") as f:
        opt_exp = json.load(f)

    cv_exp_folds = cv_exp.iloc[:5]
    cv_exp_mean = cv_exp_folds[["ppml_rmse_eur_thousands", "ppml_mae", "ppml_r2",
                                "xgb_rmse_eur_thousands", "xgb_mae", "xgb_r2"]].astype(float).mean()

    wape_cv_xgb_imp = _wape(cv_pred_imp["y_true_eur_thousands"].values,
                            cv_pred_imp["y_pred_xgb"].values)
    wape_cv_xgb_exp = _wape(cv_pred_exp["y_true_eur_thousands"].values,
                            cv_pred_exp["y_pred_xgb"].values)
    mean_y_cv_imp = float(cv_pred_imp["y_true_eur_thousands"].mean())
    mean_y_cv_exp = float(cv_pred_exp["y_true_eur_thousands"].mean())

    ho_xgb_imp = ho_imp["xgboost"]["combined"]
    ho_xgb_exp = ho_exp["xgboost"]["combined"]
    _, _, meta_imp_full = xgb_model.build_xgb_xy(panel_import)
    _, _, meta_exp_full = xgb_model.build_xgb_xy(panel_export)
    _, ho_idx_imp = xgb_model.holdout_split(meta_imp_full, *cfg.CV_HOLDOUT)
    _, ho_idx_exp = xgb_model.holdout_split(meta_exp_full, *cfg.CV_HOLDOUT)
    ho_combined_idx_imp = np.concatenate(list(ho_idx_imp.values()))
    ho_combined_idx_exp = np.concatenate(list(ho_idx_exp.values()))
    mean_y_ho_imp = float(meta_imp_full["imports_eur_thousands"].iloc[ho_combined_idx_imp].mean())
    mean_y_ho_exp = float(meta_exp_full["imports_eur_thousands"].iloc[ho_combined_idx_exp].mean())

    def _holdout_wape_from_saved(panel, booster_path):
        bundle = joblib.load(booster_path)
        model = bundle["model"]
        X, y, meta = xgb_model.build_xgb_xy(panel)
        _, test_idx_by_year = xgb_model.holdout_split(meta, *cfg.CV_HOLDOUT)
        ho_idx = np.concatenate(list(test_idx_by_year.values()))
        y_pred = xgb_model.predict_levels(model, X.iloc[ho_idx])
        y_true = meta["imports_eur_thousands"].values[ho_idx]
        return _wape(y_true, y_pred)

    wape_ho_xgb_imp = _holdout_wape_from_saved(
        panel_import, cfg.MODELS / "xgb_best.joblib"
    )
    wape_ho_xgb_exp = _holdout_wape_from_saved(
        panel_export, cfg.MODELS / "xgb_best_export.joblib"
    )

    rows = []

    def add(section, metric, imp, exp, note=""):
        rows.append({"section": section, "metric": metric,
                     "import_value": imp, "export_value": exp, "note": note})

    add("level CV (XGB)", "CV mean RMSE (EUR thousands)",
        float(cv_imp_mean["xgb_rmse_eur_thousands"]),
        float(cv_exp_mean["xgb_rmse_eur_thousands"]))
    add("level CV (XGB)", "CV mean MAE (EUR thousands)",
        float(cv_imp_mean["xgb_mae"]), float(cv_exp_mean["xgb_mae"]))
    add("level CV (XGB)", "CV mean R2",
        float(cv_imp_mean["xgb_r2"]), float(cv_exp_mean["xgb_r2"]))
    add("level CV (PPML)", "CV mean RMSE (EUR thousands)",
        float(cv_imp_mean["ppml_rmse_eur_thousands"]),
        float(cv_exp_mean["ppml_rmse_eur_thousands"]))
    add("level CV (PPML)", "CV mean R2",
        float(cv_imp_mean["ppml_r2"]), float(cv_exp_mean["ppml_r2"]))
    add("level holdout (XGB)", "Holdout combined RMSE",
        float(ho_xgb_imp["rmse"]), float(ho_xgb_exp["rmse"]))
    add("level holdout (XGB)", "Holdout combined R2",
        float(ho_xgb_imp["r2"]), float(ho_xgb_exp["r2"]))
    add("level holdout (PPML)", "Holdout combined RMSE",
        float(ho_imp["ppml_predictive_combined"]["rmse"]),
        float(ho_exp["ppml_predictive_combined"]["rmse"]))
    add("level holdout (PPML)", "Holdout combined R2",
        float(ho_imp["ppml_predictive_combined"]["r2"]),
        float(ho_exp["ppml_predictive_combined"]["r2"]))
    add("persistence", "CV pool RMSE",
        persist_imp["cv"]["rmse_eur_thousands"],
        export_persistence["cv"]["rmse_eur_thousands"])
    add("persistence", "CV pool R2",
        persist_imp["cv"]["r2"], export_persistence["cv"]["r2"])
    add("persistence", "Holdout combined RMSE",
        persist_imp["holdout_combined"]["rmse_eur_thousands"],
        export_persistence["holdout_combined"]["rmse_eur_thousands"])
    add("persistence", "Holdout combined R2",
        persist_imp["holdout_combined"]["r2"],
        export_persistence["holdout_combined"]["r2"])
    add("normalised CV (XGB)", "CV RMSE / mean(y_true)",
        float(cv_imp_mean["xgb_rmse_eur_thousands"]) / mean_y_cv_imp,
        float(cv_exp_mean["xgb_rmse_eur_thousands"]) / mean_y_cv_exp)
    add("normalised CV (XGB)", "CV MAE / mean(y_true)",
        float(cv_imp_mean["xgb_mae"]) / mean_y_cv_imp,
        float(cv_exp_mean["xgb_mae"]) / mean_y_cv_exp)
    add("normalised CV (XGB)", "WAPE_CV (pooled)",
        wape_cv_xgb_imp, wape_cv_xgb_exp)
    add("normalised holdout (XGB)", "Holdout combined WAPE",
        wape_ho_xgb_imp, wape_ho_xgb_exp)
    add("normalised holdout (XGB)", "Holdout RMSE / mean(y_true)",
        float(ho_xgb_imp["rmse"]) / mean_y_ho_imp,
        float(ho_xgb_exp["rmse"]) / mean_y_ho_exp)
    add("statistical DM", "DM test n", dm_imp["n"], dm_exp["n"])
    add("statistical DM", "DM_HLN", dm_imp["dm_hln"], dm_exp["dm_hln"])
    add("statistical DM", "DM p-value", dm_imp["p_value"], dm_exp["p_value"])

    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / f"{out_stem}.csv", index=False)
    with open(cfg.TABLES / f"{out_stem}.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=True))
    return df
