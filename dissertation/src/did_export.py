"""Export-side spillover DiD for Phase 1.5.

Single-spec causal estimate of whether Kosovo's exports to Serbia show
a Serbia-specific 2019 contraction under the same locked DiD design
used for the import side in Phase 4 (treated=XS, controls=AL/MK/ME,
treatment_years={2019}, partner+year FE, stratified pairs bootstrap
N=1000 with seed=42).

This module does not modify any Phase 1-5 module. It reuses
ppml.fit_ppml_did and ppml.did_pairs_bootstrap verbatim via a
column-swap trick: the underlying PPML code reads
imports_eur_thousands as the target column, so we build a 60-row
panel where imports_eur_thousands has been replaced by the true
exports_eur_thousands values, and pass that to the existing PPML
code. The original imports column is preserved as
imports_eur_thousands_actual for assertion-based audit. All downstream
artefact names, JSON keys, table columns, and prose sentences refer
to the outcome as exports (e.g. beta_export,
tbl_ch5_export_spillover_summary); the column-swap is purely
internal mechanics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import eda, ppml


DID_PARTNERS = ["XS", "AL", "MK", "ME"]


def load_did_export_panel() -> pd.DataFrame:
    """Build the 4-partner DiD panel carrying both flows on each row.

    Because the canonical panel_bilateral.parquet contains only
    imports_eur_thousands, this function builds the exports column
    itself by parsing the May 2026 ASK partner workbook via
    eda.parse_partner_full() and merging onto the 4-partner subset.
    Returns the 60-row merged frame; raises if any expected invariant
    is violated.
    """
    panel = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    sub = panel[panel["iso2"].isin(DID_PARTNERS)].copy()
    if len(sub) != 60:
        raise ValueError(
            f"DiD subset has {len(sub)} rows, expected 60 "
            f"(4 partners x 15 years). Partners present: "
            f"{sorted(sub['iso2'].unique())}"
        )

    ek = eda.parse_partner_full()
    ek_exp = ek[(ek["iso2"].isin(DID_PARTNERS)) & (ek["flow"] == "export")]
    ek_exp = ek_exp[["iso2", "year", "value_eur_thousands"]].rename(
        columns={"value_eur_thousands": "exports_eur_thousands"}
    )
    if len(ek_exp) != 60:
        raise ValueError(
            f"ASK export rows: {len(ek_exp)}, expected 60 "
            f"(4 partners x 15 years 2010-2024)"
        )

    merged = sub.merge(ek_exp, on=["iso2", "year"], how="left", validate="one_to_one")
    if merged["exports_eur_thousands"].isna().any():
        n_missing = int(merged["exports_eur_thousands"].isna().sum())
        bad = merged[merged["exports_eur_thousands"].isna()][["iso2", "year"]]
        raise ValueError(
            f"exports_eur_thousands has {n_missing} null rows after merge:\n{bad}"
        )
    if len(merged) != 60:
        raise ValueError(f"merge changed row count: {len(merged)} != 60")
    return merged


def _panel_with_exports_as_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Swap imports_eur_thousands with exports_eur_thousands.

    ppml.fit_ppml_did and ppml.did_pairs_bootstrap target the
    imports_eur_thousands column unconditionally. To run them on the
    export values without touching ppml.py, build a copy of the panel
    where the column has been swapped; the original imports column is
    preserved as imports_eur_thousands_actual.
    """
    if "exports_eur_thousands" not in panel.columns:
        raise ValueError(
            "exports_eur_thousands missing from panel - "
            "call load_did_export_panel() first"
        )
    p = panel.copy()
    p["imports_eur_thousands_actual"] = p["imports_eur_thousands"]
    p["imports_eur_thousands"] = p["exports_eur_thousands"]
    if not (p["imports_eur_thousands"] == panel["exports_eur_thousands"]).all():
        raise AssertionError("Column-swap failed: swapped imports != original exports")
    if not (p["imports_eur_thousands_actual"] == panel["imports_eur_thousands"]).all():
        raise AssertionError("Column-swap failed: imports_actual != original imports")
    return p


def smoke_test_bootstrap_export(panel: pd.DataFrame,
                                  n_boot: int = 40,
                                  seed: int = 42) -> dict:
    """40-draw smoke test before the canonical N=1000 run.

    Operational checks: bootstrap completes with fewer than 5% fit
    failures; bootstrap mean has same sign as main beta;
    |bootstrap_mean| within [0.25 * |beta|, 4 * |beta|]. Returns a
    diagnostic dict with pass_flag.
    """
    panel_swap = _panel_with_exports_as_target(panel)
    main = ppml.fit_ppml_did(
        panel_swap,
        treated_iso2="XS",
        control_iso2=["AL", "MK", "ME"],
        treatment_years=[2019],
    )
    names = list(main.model_result.model.exog_names)
    params = np.asarray(main.model_result.params)
    beta_main = float(params[names.index("serbia_x_post")])

    boot = ppml.did_pairs_bootstrap(
        panel_swap,
        n_boot=n_boot,
        seed=seed,
        treated_iso2="XS",
        control_iso2=["AL", "MK", "ME"],
        treatment_years=[2019],
    )

    bm = boot["mean_beta_did"]
    same_sign = (np.sign(bm) == np.sign(beta_main))
    in_band = (0.25 * abs(beta_main) <= abs(bm) <= 4.0 * abs(beta_main))
    fail_rate = boot["n_failed"] / boot["n_boot"]
    pass_flag = same_sign and in_band and (fail_rate < 0.05)

    return {
        "beta_main": beta_main,
        "bootstrap_mean": bm,
        "n_valid": boot["n_valid"],
        "n_failed": boot["n_failed"],
        "fail_rate": fail_rate,
        "same_sign": bool(same_sign),
        "in_magnitude_band": bool(in_band),
        "pass_flag": bool(pass_flag),
    }


def main_export_did_with_bootstrap(panel: pd.DataFrame,
                                    n_boot: int = 1000,
                                    seed: int = 42) -> dict:
    """Canonical export-side DiD: main fit + N=1000 stratified pairs bootstrap."""
    panel_swap = _panel_with_exports_as_target(panel)
    t0 = time.time()
    main = ppml.fit_ppml_did(
        panel_swap,
        treated_iso2="XS",
        control_iso2=["AL", "MK", "ME"],
        treatment_years=[2019],
    )
    names = list(main.model_result.model.exog_names)
    idx = names.index("serbia_x_post")
    params = np.asarray(main.model_result.params)
    bse = np.asarray(main.model_result.bse)
    beta_export = float(params[idx])
    se_cluster_placeholder = float(bse[idx])

    boot = ppml.did_pairs_bootstrap(
        panel_swap,
        n_boot=n_boot,
        seed=seed,
        treated_iso2="XS",
        control_iso2=["AL", "MK", "ME"],
        treatment_years=[2019],
    )
    wall = time.time() - t0

    exp_beta_minus_1_pct = 100.0 * (np.exp(beta_export) - 1.0)
    exp_ci_low_pct = 100.0 * (np.exp(boot["ci_95_lower"]) - 1.0)
    exp_ci_high_pct = 100.0 * (np.exp(boot["ci_95_upper"]) - 1.0)

    return {
        "beta_export": beta_export,
        "se_cluster_placeholder": se_cluster_placeholder,
        "n_boot": int(boot["n_boot"]),
        "n_boot_completed": int(boot["n_valid"]),
        "n_boot_failed": int(boot["n_failed"]),
        "bootstrap_mean": float(boot["mean_beta_did"]),
        "bootstrap_ci_low": float(boot["ci_95_lower"]),
        "bootstrap_ci_high": float(boot["ci_95_upper"]),
        "exp_beta_minus_1_pct": float(exp_beta_minus_1_pct),
        "exp_ci_low_pct": float(exp_ci_low_pct),
        "exp_ci_high_pct": float(exp_ci_high_pct),
        "wall_time_seconds": float(wall),
        "n_panel": int(len(panel)),
        "draws": boot["draws"],
    }


def import_vs_export_comparison(import_ci: dict, export_ci: dict) -> dict:
    """Side-by-side import-vs-export DiD comparison record.

    Reads the Phase 4 import-side CI (beta_main etc.) and the new
    Phase 1.5 export-side CI (beta_export etc.). The cis_overlap
    field is a descriptive heuristic; it is NOT a formal test of
    coefficient equality and is labelled as such downstream.
    """
    beta_import = float(import_ci["beta_main"])
    beta_export = float(export_ci["beta_export"])

    imp_lo = float(import_ci["bootstrap_ci_low"])
    imp_hi = float(import_ci["bootstrap_ci_high"])
    exp_lo = float(export_ci["bootstrap_ci_low"])
    exp_hi = float(export_ci["bootstrap_ci_high"])

    cis_overlap = not (imp_hi < exp_lo or exp_hi < imp_lo)
    ratio = beta_export / beta_import if beta_import != 0 else float("nan")
    delta_exp_minus_1_pct = (
        float(export_ci["exp_beta_minus_1_pct"]) -
        float(import_ci["exp_beta_minus_1_pct"])
    )

    return {
        "beta_import": beta_import,
        "beta_export": beta_export,
        "ratio": float(ratio),
        "delta_exp_minus_1_pct": float(delta_exp_minus_1_pct),
        "cis_overlap": bool(cis_overlap),
        "import_ci_95_low": imp_lo,
        "import_ci_95_high": imp_hi,
        "export_ci_95_low": exp_lo,
        "export_ci_95_high": exp_hi,
        "import_exp_minus_1_pct": float(import_ci["exp_beta_minus_1_pct"]),
        "export_exp_minus_1_pct": float(export_ci["exp_beta_minus_1_pct"]),
        "import_exp_ci_low_pct": float(import_ci["exp_ci_low_pct"]),
        "import_exp_ci_high_pct": float(import_ci["exp_ci_high_pct"]),
        "export_exp_ci_low_pct": float(export_ci["exp_ci_low_pct"]),
        "export_exp_ci_high_pct": float(export_ci["exp_ci_high_pct"]),
        "n_import_did": 60,
        "n_export_did": 60,
        "treatment_year": 2019,
        "treated_iso2": "XS",
        "control_iso2": ["AL", "MK", "ME"],
    }


def build_asymmetry_figure(panel: pd.DataFrame, comparison: dict,
                            out_stem: str = "fig_ch5_import_vs_export_asymmetry"):
    """Two-panel headline figure for the export-spillover chapter.

    Panel A: indexed trajectories (2017=100) for Serbia and the
    control mean, both flows. Panel B: forest plot of beta_import vs
    beta_export with 95% bootstrap CIs.
    """
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    df = panel[["iso2", "year", "imports_eur_thousands", "exports_eur_thousands"]].copy()
    serbia = df[df.iso2 == "XS"].set_index("year").sort_index()
    controls = df[df.iso2.isin(["AL", "MK", "ME"])].groupby("year")[
        ["imports_eur_thousands", "exports_eur_thousands"]
    ].mean().sort_index()

    def idx_to_2017(s):
        base = s.loc[2017]
        return (s / base) * 100.0 if base != 0 else s * 0.0

    serbia_imp_idx = idx_to_2017(serbia["imports_eur_thousands"])
    serbia_exp_idx = idx_to_2017(serbia["exports_eur_thousands"])
    ctrl_imp_idx = idx_to_2017(controls["imports_eur_thousands"])
    ctrl_exp_idx = idx_to_2017(controls["exports_eur_thousands"])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.plot(serbia_imp_idx.index, serbia_imp_idx.values, marker="o",
             linewidth=2.0, color="#c0392b", label="Serbia imports")
    axA.plot(serbia_exp_idx.index, serbia_exp_idx.values, marker="s",
             linewidth=2.0, color="#e67e22", label="Serbia exports")
    axA.plot(ctrl_imp_idx.index, ctrl_imp_idx.values, marker="o",
             linewidth=1.4, color="#2c3e50", linestyle="--",
             label="Control mean (AL/MK/ME) imports")
    axA.plot(ctrl_exp_idx.index, ctrl_exp_idx.values, marker="s",
             linewidth=1.4, color="#16a085", linestyle="--",
             label="Control mean (AL/MK/ME) exports")
    axA.axvline(2018.92, color="black", linewidth=0.8, linestyle=":")
    axA.axvline(2020.25, color="black", linewidth=0.8, linestyle=":")
    axA.axhline(100, color="grey", linewidth=0.5)
    axA.set_xlabel("Year")
    axA.set_ylabel("Indexed value (2017 = 100)")
    axA.set_title("(A) Indexed trade trajectories, 2010-2024")
    axA.legend(loc="upper left", fontsize=8, framealpha=0.85)

    rows = [
        ("Import-side beta", comparison["beta_import"],
         comparison["import_ci_95_low"], comparison["import_ci_95_high"]),
        ("Export-side beta", comparison["beta_export"],
         comparison["export_ci_95_low"], comparison["export_ci_95_high"]),
    ]
    ys = list(range(len(rows)))
    for y, (label, b, lo, hi) in zip(ys, rows):
        axB.plot([lo, hi], [y, y], color="#2c3e50", linewidth=2)
        axB.plot([b], [y], marker="o", color="#c0392b", markersize=9, zorder=5)
    axB.axvline(0, color="black", linewidth=0.8)
    axB.set_yticks(ys)
    axB.set_yticklabels([r[0] for r in rows])
    axB.set_xlabel("beta on serbia_x_post (log-link units)")
    axB.set_title("(B) Import vs export beta with 95% bootstrap CIs")
    axB.invert_yaxis()

    fig.tight_layout()
    fig.savefig(cfg.FIGURES / f"{out_stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(cfg.FIGURES / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_summary_table(import_ci: dict, export_ci: dict,
                          out_stem: str = "tbl_ch5_export_spillover_summary"):
    """Per-flow summary row table (CSV + .tex)."""
    rows = [
        {
            "flow": "import",
            "n": 60,
            "beta": import_ci["beta_main"],
            "bootstrap_mean": import_ci["bootstrap_mean"],
            "ci_95_low": import_ci["bootstrap_ci_low"],
            "ci_95_high": import_ci["bootstrap_ci_high"],
            "exp_beta_minus_1_pct": import_ci["exp_beta_minus_1_pct"],
            "ci_low_pct": import_ci["exp_ci_low_pct"],
            "ci_high_pct": import_ci["exp_ci_high_pct"],
        },
        {
            "flow": "export",
            "n": 60,
            "beta": export_ci["beta_export"],
            "bootstrap_mean": export_ci["bootstrap_mean"],
            "ci_95_low": export_ci["bootstrap_ci_low"],
            "ci_95_high": export_ci["bootstrap_ci_high"],
            "exp_beta_minus_1_pct": export_ci["exp_beta_minus_1_pct"],
            "ci_low_pct": export_ci["exp_ci_low_pct"],
            "ci_high_pct": export_ci["exp_ci_high_pct"],
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(cfg.TABLES / f"{out_stem}.csv", index=False)
    with open(cfg.TABLES / f"{out_stem}.tex", "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f", escape=True))
