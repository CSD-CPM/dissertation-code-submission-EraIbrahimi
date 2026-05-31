"""Phase 3 orchestrator — interpretation pillar.

Runs end-to-end:

    python -m src.run_phase3

Steps:
  1. Capture pre-run SHA256 of `models/xgb_best.joblib` (read-only model).
  2. Load Phase-2 artefacts (panel, saved booster, pooled CV errors,
     persisted best params).
  3. Build (X, y, meta) — must be 1,559 rows / 12 features.
  4. SHAP global → table + beeswarm + bar chart (model output space:
     log1p(EUR thousands)).
  5. SHAP for Serbia (14 rows × 12 features) → long table + heatmap.
  6. 4-layer ablation CV (identical-rowset; X_full[layer_features] only).
  7. 4-layer ablation holdout → table + grouped bar chart.
  8. Diebold-Mariano HLN test on pooled CV errors (n = 558).
  9. (Optional, ≤ 50 MB) persist SHAP values bundle.
 10. Re-check SHA256 — must match the pre-run hash.
 11. Print summary; report is written separately.

PROTOCOL_FREEZE binding rules respected:
  - SHAP described as "predictive contributions" / "associative
    importance"; never "effects" anywhere.
  - SHAP values stay in log1p(EUR thousands); axis labels say so.
  - Ablation: X_full columns are subset; `xgb_model.build_xgb_xy` is
    called exactly once and its row set is reused for every layer.
  - The canonical saved model is never retrained or overwritten.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import xgb_model as xgb
from . import interpret as it


SHAP_LOG1P_LABEL = "SHAP value (model output units: log1p(EUR thousands))"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# pandas.to_latex(escape=True) escapes the table body, but the caption is
# written through VERBATIM. Captions containing underscores
# (best_params, sq_err_PPML, sq_err_XGB, ...) would otherwise emit
# uncompilable LaTeX. Escape them here before handing to pandas.
_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}",
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in s)


def _to_csv_and_tex(df: pd.DataFrame, path_csv: Path, path_tex: Path,
                   caption: str = None, label: str = None,
                   float_format: str = "%.4f", **to_latex_kwargs):
    """Persist a DataFrame as both .csv and .tex (one-line LaTeX include).

    Caption is escaped for LaTeX before being passed to pandas — pandas
    only escapes the body. Label is left as-is (it is a reference key,
    not rendered text).
    """
    df.to_csv(path_csv, index=to_latex_kwargs.pop("index_csv", False))
    tex_kwargs = dict(index=False, escape=True, float_format=float_format)
    tex_kwargs.update(to_latex_kwargs)
    if caption:
        tex_kwargs["caption"] = _latex_escape(caption)
    if label:
        tex_kwargs["label"] = label
    with open(path_tex, "w") as f:
        f.write(df.to_latex(**tex_kwargs))


def main():
    print("\n[Phase 3] step 1/11 — capture pre-run SHA256 of xgb_best.joblib")
    model_path = cfg.MODELS / "xgb_best.joblib"
    sha_before = _sha256(model_path)
    print(f"  SHA256 (before): {sha_before}")

    print("\n[Phase 3] step 2/11 — load Phase-2 artefacts")
    panel = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    bundle = it.load_xgb_best()
    cv_pred_errors = pd.read_csv(cfg.METRICS / "cv_prediction_errors.csv")
    best_params_meta = json.loads(
        (cfg.METRICS / "optuna_best_params.json").read_text()
    )
    cv_xgb_vs_ppml = pd.read_csv(cfg.METRICS / "cv_xgb_vs_ppml.csv")

    print(f"  panel:                  {len(panel)} rows × {panel.shape[1]} cols")
    print(f"  saved best_params:      {bundle['best_params']}")
    print(f"  optuna_best_params:     {best_params_meta['best_params']}")
    if bundle["best_params"] != best_params_meta["best_params"]:
        raise AssertionError(
            "Saved booster's best_params does not match optuna_best_params.json. "
            f"\n  joblib:  {bundle['best_params']}"
            f"\n  json:    {best_params_meta['best_params']}"
        )
    print("  best_params match ✓")
    print(f"  cv_prediction_errors:   {len(cv_pred_errors)} rows")

    print("\n[Phase 3] step 3/11 — build (X, y, meta)")
    X, y, meta = xgb.build_xgb_xy(panel)
    if len(X) != 1559:
        raise AssertionError(
            f"Expected 1,559 rows in X (Phase 2 v2); got {len(X)}."
        )
    print(f"  X.shape = {X.shape}; meta.shape = {meta.shape}")

    print("\n[Phase 3] step 4/11 — SHAP global "
          "(values in log1p(EUR thousands) — model output space)")
    t0 = time.time()
    shap_values, expected_value = it.compute_shap(bundle["model"], X)
    dt = time.time() - t0
    print(f"  shap_values.shape = {shap_values.shape}  ({dt:.1f}s)")
    if shap_values.shape != (1559, 12):
        raise AssertionError(
            f"shap_values.shape mismatch: got {shap_values.shape}, expected (1559, 12)."
        )

    # Additivity diagnostic: log1p_pred ≈ expected_value + shap_values.sum(axis=1)
    log1p_pred_from_shap = expected_value + shap_values.sum(axis=1)
    log1p_pred_from_model = np.asarray(bundle["model"].predict(X), dtype=float)
    add_max_dev = float(np.max(np.abs(log1p_pred_from_shap - log1p_pred_from_model)))
    print(f"  expected_value (log1p space)            = {expected_value:.6f}")
    print(f"  additivity max abs deviation (log1p)    = {add_max_dev:.2e}")
    if add_max_dev > 1e-3:
        raise AssertionError(
            f"SHAP additivity check failed: max abs deviation = {add_max_dev:.6f}; "
            "expected < 1e-3."
        )
    print("  additivity ✓")

    summary = it.shap_global_summary(shap_values, X)
    _to_csv_and_tex(
        summary,
        cfg.TABLES / "tbl_ch4_shap_global.csv",
        cfg.TABLES / "tbl_ch4_shap_global.tex",
        caption=("Global SHAP summary across the panel "
                 "(predictive contributions in log1p(EUR thousands)). "
                 "Sorted by mean(|SHAP|)."),
        label="tbl:ch4-shap-global",
    )
    print("  global summary (top 5 by mean |SHAP|):")
    print(summary.head(5).to_string(index=False))
    print(f"  → wrote {cfg.TABLES / 'tbl_ch4_shap_global.csv'}")

    # Render figures (beeswarm + bar)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    for kind, name in [(None, "fig_ch4_shap_beeswarm"),
                       ("bar", "fig_ch4_shap_bar")]:
        plt.figure()
        shap.summary_plot(
            shap_values, X,
            feature_names=list(X.columns),
            plot_type=kind, show=False,
        )
        ax = plt.gca()
        ax.set_xlabel(SHAP_LOG1P_LABEL)
        plt.title(("Predictive contributions" if kind is None else
                   "Mean absolute predictive contribution") +
                  " (model output: log1p(EUR thousands))")
        for ext in ("png", "pdf"):
            plt.savefig(cfg.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close()
    print(f"  → wrote fig_ch4_shap_beeswarm.{{png,pdf}} and fig_ch4_shap_bar.{{png,pdf}}")

    print("\n[Phase 3] step 5/11 — SHAP for Serbia "
          "(14 rows × 12 features; 2010 omitted from feasible set)")
    serbia_long, serbia_wide = it.shap_for_serbia(shap_values, X, meta)
    print(f"  serbia_long.shape = {serbia_long.shape}; "
          f"serbia_wide.shape = {serbia_wide.shape}")

    serbia_long.to_csv(cfg.TABLES / "tbl_ch4_serbia_shap.csv", index=False)
    with open(cfg.TABLES / "tbl_ch4_serbia_shap.tex", "w") as f:
        f.write(serbia_wide.to_latex(
            float_format="%.3f", escape=True, index=True,
            caption=_latex_escape(
                "Predictive contributions for Serbia (XS), 2011-2024. "
                "Cells in log1p(EUR thousands)."
            ),
            label="tbl:ch4-serbia-shap",
        ))
    print(f"  → wrote tbl_ch4_serbia_shap.{{csv,tex}}")

    # Heatmap
    import seaborn as sns
    plt.figure(figsize=(11, 5))
    abs_max = float(np.max(np.abs(serbia_wide.values)))
    sns.heatmap(
        serbia_wide.values,
        xticklabels=list(serbia_wide.columns),
        yticklabels=list(serbia_wide.index),
        cmap="RdBu_r",
        center=0,
        vmin=-abs_max, vmax=abs_max,
        annot=True, fmt=".2f",
        cbar_kws={"label": SHAP_LOG1P_LABEL},
    )
    plt.title("Serbia (XS): per-year predictive contributions per feature")
    plt.xlabel("feature")
    plt.ylabel("year")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(cfg.FIGURES / f"fig_ch4_serbia_shap_heatmap.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → wrote fig_ch4_serbia_shap_heatmap.{{png,pdf}}")

    print("\n[Phase 3] step 6/11 — 4-layer ablation CV "
          "(identical-rowset; X_full[layer_features] only)")
    ab_cv = it.ablation_cv(panel, bundle["best_params"])
    expected_counts = {"L1_structural": 3, "L2_policy": 7,
                       "L3_macro": 10, "L4_lagged": 12}
    for layer, expected in expected_counts.items():
        got = int(ab_cv.loc[ab_cv["layer"] == layer, "n_features"].iloc[0])
        if got != expected:
            raise AssertionError(
                f"Ablation layer {layer}: expected {expected} features, got {got}."
            )
    print("  per-layer feature counts ✓ (L1=3, L2=7, L3=10, L4=12)")
    means = ab_cv[ab_cv["fold_idx"] == "mean"][
        ["layer", "n_features", "rmse", "mae", "r2"]]
    print("  per-layer mean across 5 CV folds:")
    print(means.to_string(index=False))
    _to_csv_and_tex(
        ab_cv, cfg.TABLES / "tbl_ch4_ablation_cv.csv",
        cfg.TABLES / "tbl_ch4_ablation_cv.tex",
        caption=("4-layer ablation, mean across the 5 expanding-window CV "
                 "folds. Same best_params as the canonical model; "
                 "temporary in-memory refits per layer × fold."),
        label="tbl:ch4-ablation-cv",
        na_rep="—",
    )
    print(f"  → wrote tbl_ch4_ablation_cv.{{csv,tex}}")

    print("\n[Phase 3] step 7/11 — 4-layer ablation holdout (2023+2024)")
    ab_h = it.ablation_holdout(panel, bundle["best_params"])
    print(ab_h.to_string(index=False))
    _to_csv_and_tex(
        ab_h, cfg.TABLES / "tbl_ch4_ablation_holdout.csv",
        cfg.TABLES / "tbl_ch4_ablation_holdout.tex",
        caption=("4-layer ablation on the 2023+2024 holdout. RMSE / MAE / "
                 "R² computed from pooled per-row predictions on the union."),
        label="tbl:ch4-ablation-holdout",
    )
    print(f"  → wrote tbl_ch4_ablation_holdout.{{csv,tex}}")

    # Grouped bar: CV mean RMSE vs holdout combined RMSE per layer
    layer_order = list(cfg.ABLATION_LAYERS.keys())
    cv_rmse_by_layer = {row["layer"]: row["rmse"] for _, row in means.iterrows()}
    h_rmse_by_layer = {row["layer"]: row["rmse"] for _, row in ab_h.iterrows()}
    cv_vals = [cv_rmse_by_layer[l] for l in layer_order]
    h_vals  = [h_rmse_by_layer[l]  for l in layer_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(layer_order))
    w = 0.4
    ax.bar(xs - w/2, cv_vals, width=w, label="CV mean RMSE (5 folds)")
    ax.bar(xs + w/2, h_vals,  width=w, label="Holdout combined RMSE (2023+2024)")
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_order, rotation=20, ha="right")
    ax.set_ylabel("RMSE (EUR thousands)")
    ax.set_title("4-layer ablation lift: CV mean vs 2023+2024 holdout")
    ax.legend()
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(cfg.FIGURES / f"fig_ch4_ablation_lift.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → wrote fig_ch4_ablation_lift.{{png,pdf}}")

    print("\n[Phase 3] step 8/11 — Diebold-Mariano HLN test on pooled CV errors")
    expected_n = int(cv_xgb_vs_ppml["n_test"].sum())
    if len(cv_pred_errors) != expected_n:
        raise AssertionError(
            f"cv_prediction_errors.csv has {len(cv_pred_errors)} rows; "
            f"expected {expected_n} from cv_xgb_vs_ppml['n_test'].sum()."
        )
    d = cv_pred_errors["loss_diff_ppml_minus_xgb"].astype(float).values
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise AssertionError(
            f"DM input is degenerate (single-sign): pos={n_pos}, neg={n_neg}."
        )
    dm = it.diebold_mariano_hln(d, h=1)
    if not (0.0 < dm["p_value"] < 1.0):
        raise AssertionError(
            f"DM p_value is degenerate: {dm['p_value']}; check inputs."
        )
    print(f"  n = {dm['n']}  h = {dm['h']}")
    print(f"  mean_loss_diff (PPML − XGB) = {dm['mean_loss_diff']:+,.2f}")
    print(f"  DM         = {dm['dm']:+.4f}")
    print(f"  DM_HLN     = {dm['dm_hln']:+.4f}   "
          f"(HLN factor = {dm['hln_factor']:.6f})")
    print(f"  df         = {dm['df']}    p-value (two-sided) = {dm['p_value']:.6f}")
    print(f"  conclusion @ α=0.05: {dm['interpretation']}")

    dm_path = cfg.METRICS / "dm_test.json"
    with open(dm_path, "w") as f:
        json.dump(dm, f, indent=2)
    dm_table = pd.DataFrame([{
        "n":              dm["n"],
        "h":              dm["h"],
        "mean_loss_diff": dm["mean_loss_diff"],
        "dm":             dm["dm"],
        "dm_hln":         dm["dm_hln"],
        "df":             dm["df"],
        "p_value":        dm["p_value"],
        "interpretation": dm["interpretation"],
    }])
    _to_csv_and_tex(
        dm_table, cfg.TABLES / "tbl_ch4_dm_test.csv",
        cfg.TABLES / "tbl_ch4_dm_test.tex",
        caption=("Diebold-Mariano test with HLN small-sample correction "
                 "on pooled CV squared-error differential "
                 "(loss = sq_err_PPML − sq_err_XGB; positive ⇒ XGB favoured)."),
        label="tbl:ch4-dm-test",
        float_format="%.6f",
    )
    print(f"  → wrote {dm_path} and tbl_ch4_dm_test.{{csv,tex}}")

    print("\n[Phase 3] step 9/11 — persist SHAP values bundle (skip if > 50 MB)")
    import joblib, tempfile, os
    shap_bundle = {
        "shap_values":    shap_values,
        "expected_value": expected_value,
        "X_columns":      list(X.columns),
        "n_rows":         int(len(X)),
        "feature_list":   list(bundle["feature_list"]),
        "seed":           int(cfg.SEED),
    }
    shap_path = cfg.MODELS / "shap_values.joblib"
    joblib.dump(shap_bundle, shap_path)
    sz_mb = os.path.getsize(shap_path) / (1024 * 1024)
    if sz_mb > 50.0:
        os.remove(shap_path)
        print(f"  shap_values.joblib was {sz_mb:.1f} MB > 50 MB cap → deleted.")
    else:
        print(f"  → wrote {shap_path}  ({sz_mb:.2f} MB)")

    print("\n[Phase 3] step 10/11 — verify canonical model file is unchanged")
    sha_after = _sha256(model_path)
    print(f"  SHA256 (after):  {sha_after}")
    if sha_before != sha_after:
        raise AssertionError(
            "models/xgb_best.joblib changed during Phase 3 run.\n"
            f"  before: {sha_before}\n  after:  {sha_after}"
        )
    print("  SHA256 unchanged ✓ (canonical model not retrained or overwritten)")

    print("\n[Phase 3] step 11/11 — done. Awaiting sign-off before next workstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
