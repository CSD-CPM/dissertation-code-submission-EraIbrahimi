"""Phase 4 orchestrator — DiD safeguards + 1,000-draw bootstrap.

Runs end-to-end:

    python -m src.run_phase4

Steps:
  1.  Capture pre-run SHA-256 of `models/xgb_best.joblib` (read-only
      invariant — Phase 4 must not touch the booster).
  2.  Load DiD-feasible panel (4 partners × 15 years = 60 rows).
  3.  Programmatic smoke test: n_boot=40 with stdout-capture; assert
      "cluster-robust SEs failed" not in captured output.
  4.  Main DiD + 1,000-draw stratified partner-pairs bootstrap.
  5.  Parallel trends data + figure (pre-2018 only).
  6.  Leads test (Wald joint, lead_years 2014–2017).
  7.  Event study (full leads + lags 2011–2024).
  8.  Placebo at fake_treatment_year=2014.
  9.  Sensitivity #1: window restricted to 2015–2021.
 10.  Sensitivity #2: + ln_partner_gdp control variant (substitute for
      the original "drop Kosovo-GDP-control" sensitivity).
 11.  Sensitivity #3: treatment_years={2018, 2019, 2020}.
 12.  Build §6 safeguards summary table.
 13.  SHAP-vs-DiD consistency paragraph (sign + ranking only).
 14.  Re-check SHA-256 of models/xgb_best.joblib; assert unchanged.

PROTOCOL_FREEZE binding rules (§13) respected throughout: DiD prose
hedged ("the point estimate indicates", "consistent with"); SHAP
language stays as "predictive contribution" / "associative
importance" — never "effect".
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import did_safeguards as ds
from . import ppml


# ---------------------------------------------------------------------------
# Reused helpers (intentional duplication from run_phase3.py — small, low-risk;
# avoids restructuring xgb_model.py / interpret.py to expose them).
# ---------------------------------------------------------------------------

_LATEX_SPECIAL_CHARS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#":  r"\#",                "_": r"\_", "{": r"\{", "}": r"\}",
    "~":  r"\textasciitilde{}",  "^": r"\textasciicircum{}",
}


def _latex_escape(s: str) -> str:
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in s)


def _to_csv_and_tex(df: pd.DataFrame, path_csv: Path, path_tex: Path,
                   caption: str = None, label: str = None,
                   float_format: str = "%.4f", **to_latex_kwargs):
    df.to_csv(path_csv, index=to_latex_kwargs.pop("index_csv", False))
    tex_kwargs = dict(index=False, escape=True, float_format=float_format)
    tex_kwargs.update(to_latex_kwargs)
    if caption:
        tex_kwargs["caption"] = _latex_escape(caption)
    if label:
        tex_kwargs["label"] = label
    with open(path_tex, "w") as f:
        f.write(df.to_latex(**tex_kwargs))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matplotlib_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "savefig.dpi":     300,
        "figure.dpi":      300,
    })
    import seaborn as sns
    sns.set_style("whitegrid")
    return plt, sns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plt, sns = _matplotlib_setup()

    print("\n[Phase 4] step 1/14 — capture pre-run SHA-256 of xgb_best.joblib")
    booster_path = cfg.MODELS / "xgb_best.joblib"
    sha_before = _sha256(booster_path)
    print(f"  SHA-256 (before): {sha_before}")

    print("\n[Phase 4] step 2/14 — load DiD-feasible panel")
    panel = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    did_panel = ds.load_did_panel(panel)
    print(f"  did_panel: {len(did_panel)} rows  partners={sorted(did_panel['iso2'].unique())}")

    print("\n[Phase 4] step 3/14 — programmatic smoke test (n_boot=40)")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        smoke = ppml.did_pairs_bootstrap(
            did_panel, n_boot=40, seed=cfg.SEED,
            treated_iso2=cfg.DID_TREATED_ISO2,
            control_iso2=cfg.DID_CONTROL_ISO2,
            treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
        )
    captured = buf.getvalue()
    if captured.strip():
        print("  [smoke captured stdout]")
        print(captured)
    assert "cluster-robust SEs failed" not in captured, (
        "Cluster-robust fallback observed during smoke test — bootstrap fix "
        "regressed; do not proceed to the 1,000-draw run."
    )
    smoke_mean = smoke["mean_beta_did"]
    print(f"  smoke n_valid = {smoke['n_valid']}, n_failed = {smoke['n_failed']}")
    print(f"  smoke mean = {smoke_mean:+.4f}")
    # Sign + magnitude assertions vs main β
    main_fit = ppml.fit_ppml_did(
        did_panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )
    names = list(main_fit.model_result.model.exog_names)
    beta_main = float(np.asarray(main_fit.model_result.params)[names.index("serbia_x_post")])
    print(f"  main β = {beta_main:+.4f}")
    assert np.sign(smoke_mean) == np.sign(beta_main), (
        f"Smoke bootstrap mean has wrong sign: {smoke_mean:+.4f} vs main β = {beta_main:+.4f}"
    )
    ratio = abs(smoke_mean) / abs(beta_main)
    assert 0.25 <= ratio <= 4.0, (
        f"Smoke bootstrap mean magnitude off: |bm|/|β| = {ratio:.3f} not in [0.25, 4.0]"
    )
    print(f"  sign + magnitude checks ✓ ({ratio:.3f}× of main β)")

    print("\n[Phase 4] step 4/14 — main DiD + 1,000-draw stratified pairs bootstrap")
    t0 = time.time()
    main_result = ds.main_did_with_bootstrap(
        did_panel, n_boot=cfg.BOOTSTRAP_N, seed=cfg.SEED,
    )
    wall = time.time() - t0
    draws = main_result.pop("draws")  # don't dump 1000 floats into JSON
    print(f"  bootstrap completed in {wall:.1f}s")
    print(f"  n_boot_completed = {main_result['n_boot_completed']} / {main_result['n_boot']}")
    print(f"  n_boot_failed    = {main_result['n_boot_failed']}")
    print(f"  β_main           = {main_result['beta_main']:+.4f}")
    print(f"  bootstrap_mean   = {main_result['bootstrap_mean']:+.4f}")
    print(f"  bootstrap 95% CI = [{main_result['bootstrap_ci_low']:+.4f}, {main_result['bootstrap_ci_high']:+.4f}]")
    print(f"  exp(β)−1         = {main_result['exp_beta_minus_1_pct']:+.2f}%")
    print(f"  exp(CI)−1        = [{main_result['exp_ci_low_pct']:+.2f}%, {main_result['exp_ci_high_pct']:+.2f}%]")

    np.save(cfg.METRICS / "bootstrap_draws.npy", draws)
    with open(cfg.METRICS / "bootstrap_ci.json", "w") as f:
        json.dump(main_result, f, indent=2, default=str)
    print(f"  → wrote {cfg.METRICS / 'bootstrap_draws.npy'}")
    print(f"  → wrote {cfg.METRICS / 'bootstrap_ci.json'}")

    # Optional histogram
    valid_draws = draws[~np.isnan(draws)]
    if len(valid_draws) >= 100:
        plt.figure(figsize=(7, 4))
        plt.hist(valid_draws, bins=40, edgecolor="white")
        plt.axvline(main_result["beta_main"], color="black", linestyle="-",
                    linewidth=1.2, label="β_main")
        plt.axvline(main_result["bootstrap_ci_low"], color="C3", linestyle="--",
                    linewidth=1.0, label="2.5 / 97.5 %")
        plt.axvline(main_result["bootstrap_ci_high"], color="C3", linestyle="--",
                    linewidth=1.0)
        plt.xlabel("β_DiD on serbia_x_post")
        plt.ylabel("draws")
        plt.title(f"Bootstrap distribution (n={int(main_result['n_boot_completed'])})")
        plt.legend()
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(cfg.FIGURES / f"fig_ch5_bootstrap_distribution.{ext}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  → wrote fig_ch5_bootstrap_distribution.{{png,pdf}}")

    print("\n[Phase 4] step 5/14 — parallel trends (pre-2018)")
    pt = ds.parallel_trends_data(panel)
    _to_csv_and_tex(
        pt, cfg.TABLES / "tbl_ch5_parallel_trends.csv",
        cfg.TABLES / "tbl_ch5_parallel_trends.tex",
        caption=("Pre-treatment (2010-2017) per-year mean of "
                 "log1p(imports_eur_thousands) for Serbia (XS), each "
                 "control partner (AL, MK, ME), and the equally-weighted "
                 "control mean."),
        label="tbl:ch5-parallel-trends",
        float_format="%.3f",
    )
    print(f"  → wrote tbl_ch5_parallel_trends.{{csv,tex}}")

    fig, ax = plt.subplots(figsize=(8, 5))
    pt_main = pt[pt["group"].isin([cfg.DID_TREATED_ISO2, "control_mean(AL+MK+ME)"])]
    for g, sub in pt_main.groupby("group"):
        sub_sorted = sub.sort_values("year")
        lw = 2.0 if g == cfg.DID_TREATED_ISO2 else 1.6
        ax.plot(sub_sorted["year"], sub_sorted["mean_log_imports"],
                marker="o", linewidth=lw, label=g)
    # Faint individual control lines for context
    for g, sub in pt[pt["group"].isin(cfg.DID_CONTROL_ISO2)].groupby("group"):
        sub_sorted = sub.sort_values("year")
        ax.plot(sub_sorted["year"], sub_sorted["mean_log_imports"],
                linestyle=":", alpha=0.55, linewidth=1.0, label=g)
    ax.axvline(2018, color="C3", linestyle="--", linewidth=1.0, alpha=0.8,
               label="tariff onset (2018)")
    ax.set_xlabel("year")
    ax.set_ylabel("mean log1p(imports_eur_thousands)")
    ax.set_title("Pre-treatment trajectories: Serbia vs control group")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(cfg.FIGURES / f"fig_ch5_parallel_trends.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  → wrote fig_ch5_parallel_trends.{{png,pdf}}")

    print("\n[Phase 4] step 6/14 — leads test (Wald joint, 2014-2017)")
    lt = ds.leads_test(did_panel)
    print(f"  joint Wald = {lt['joint_wald_stat']:.4f}  "
          f"p = {lt['joint_p_value']:.6f}  df = {lt['df']}")
    print(f"  per-lead β: {lt['per_lead_coefs']}")
    print(f"  interpretation: {lt['interpretation']}")
    leads_table = pd.DataFrame([
        {"year": y, "beta": lt["per_lead_coefs"][y], "se": lt["per_lead_se"][y],
         "t_stat": lt["per_lead_coefs"][y] / lt["per_lead_se"][y]
                   if lt["per_lead_se"][y] else np.nan}
        for y in sorted(lt["per_lead_coefs"])
    ])
    _to_csv_and_tex(
        leads_table, cfg.TABLES / "tbl_ch5_leads_test.csv",
        cfg.TABLES / "tbl_ch5_leads_test.tex",
        caption=("Pre-treatment leads test: Serbia x year_t coefficients "
                 "for t in {2014, 2015, 2016, 2017} alongside the locked "
                 "main treatment indicator. Cluster-robust SEs at "
                 "partner level."),
        label="tbl:ch5-leads-test",
    )
    with open(cfg.METRICS / "leads_test.json", "w") as f:
        json.dump({k: (v if not isinstance(v, dict) else
                       {str(kk): vv for kk, vv in v.items()})
                   for k, v in lt.items()}, f, indent=2, default=str)
    print(f"  → wrote tbl_ch5_leads_test.{{csv,tex}} and leads_test.json")

    print("\n[Phase 4] step 7/14 — event study (2011-2024 ex 2010)")
    es_df, es_status = ds.event_study(did_panel)
    print(f"  status: {es_status}")
    if es_status["status"] == "ok":
        print(es_df[["year", "beta", "se", "t_stat"]].to_string(index=False))
    _to_csv_and_tex(
        es_df, cfg.TABLES / "tbl_ch5_event_study.csv",
        cfg.TABLES / "tbl_ch5_event_study.tex",
        caption=("Event study: Serbia x year_t coefficients for every "
                 "year in 2011-2024 (reference year 2010). Single PPML "
                 "fit; if it does not converge, this table reports the "
                 "failure status only."),
        label="tbl:ch5-event-study",
    )
    if es_status["status"] == "ok":
        fig, ax = plt.subplots(figsize=(9, 5))
        sub = es_df.dropna(subset=["beta"]).sort_values("year")
        ax.errorbar(sub["year"], sub["beta"],
                    yerr=[sub["beta"] - sub["ci_low_95"], sub["ci_high_95"] - sub["beta"]],
                    fmt="o", capsize=3, linewidth=1.4, markersize=5,
                    label="β_t (95% CI)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(2018, color="C3", linestyle="--", linewidth=1.0, alpha=0.8,
                   label="tariff onset (2018)")
        ax.set_xlabel("year")
        ax.set_ylabel("β on Serbia x year_t (log-link units)")
        ax.set_title("Event study: Serbia x year_t coefficients")
        ax.legend()
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(cfg.FIGURES / f"fig_ch5_event_study.{ext}",
                        dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  → wrote fig_ch5_event_study.{{png,pdf}}")
    else:
        print("  event-study figure skipped (refit did not converge)")
    print(f"  → wrote tbl_ch5_event_study.{{csv,tex}}")

    print("\n[Phase 4] step 8/14 — placebo (fake_treatment_year=2014)")
    pb = ds.placebo_did(did_panel, fake_treatment_year=2014)
    for k, v in pb.items():
        print(f"  {k}: {v}")
    with open(cfg.METRICS / "placebo.json", "w") as f:
        json.dump(pb, f, indent=2, default=str)
    print(f"  → wrote placebo.json")

    print("\n[Phase 4] step 9/14 — sensitivity #1: window 2015-2021")
    sw = ds.sensitivity_window(did_panel, year_range=(2015, 2021))
    for k, v in sw.items():
        print(f"  {k}: {v}")
    with open(cfg.METRICS / "sensitivity_window.json", "w") as f:
        json.dump(sw, f, indent=2, default=str)
    print(f"  → wrote sensitivity_window.json")

    print("\n[Phase 4] step 10/14 — sensitivity #2: + ln_partner_gdp control variant (substitute)")
    sg = ds.sensitivity_partner_gdp_control(did_panel)
    for k, v in sg.items():
        print(f"  {k}: {v}")
    with open(cfg.METRICS / "sensitivity_partner_gdp.json", "w") as f:
        json.dump(sg, f, indent=2, default=str)
    print(f"  → wrote sensitivity_partner_gdp.json")

    print("\n[Phase 4] step 11/14 — sensitivity #3: treatment_years={2018,2019,2020}")
    sts = ds.sensitivity_treatment_window(did_panel)
    for k, v in sts.items():
        print(f"  {k}: {v}")
    with open(cfg.METRICS / "sensitivity_treatment_window.json", "w") as f:
        json.dump(sts, f, indent=2, default=str)
    print(f"  → wrote sensitivity_treatment_window.json")

    print("\n[Phase 4] step 12/14 — build §6 safeguards summary table")
    main_b = main_result["beta_main"]
    main_se = main_result["se_cluster_placeholder"]
    summary_rows = [
        {"spec": "Main (treatment_years={2019})",
         "beta": main_b, "se": main_se, "t": main_b / main_se if main_se else np.nan,
         "n": 60, "notes": "headline; bootstrap CI is canonical inference"},
        {"spec": f"Placebo (fake treatment_year={pb['fake_year']})",
         "beta": pb["beta"], "se": pb["se"], "t": pb["t"], "n": pb["n"],
         "notes": pb["interpretation"]},
        {"spec": f"Sensitivity: window {sw['year_range'][0]}-{sw['year_range'][1]}",
         "beta": sw.get("beta", np.nan), "se": sw.get("se", np.nan),
         "t": sw.get("t", np.nan), "n": sw.get("n", np.nan),
         "notes": f"status={sw.get('status')}"},
        {"spec": "Sensitivity: + ln_partner_gdp control",
         "beta": sg["beta_with_gdp"], "se": sg["se"], "t": sg["t"], "n": sg["n"],
         "notes": "substitute for Kosovo-GDP-control sensitivity (see §6 footnote)"},
        {"spec": "Sensitivity: treatment_years={2018,2019,2020}",
         "beta": sts["beta_2018_19_20"], "se": sts["se_2018_19_20"],
         "t": sts["t_2018_19_20"], "n": sts["n"],
         "notes": "window-robust check"},
    ]
    summary = pd.DataFrame(summary_rows)
    _to_csv_and_tex(
        summary, cfg.TABLES / "tbl_ch5_safeguards_summary.csv",
        cfg.TABLES / "tbl_ch5_safeguards_summary.tex",
        caption=("Safeguards summary: main DiD plus placebo and three "
                 "sensitivity refits. Cluster-robust SEs at partner level "
                 "(placeholder; the bootstrap CI on the main spec is the "
                 "canonical inference statement)."),
        label="tbl:ch5-safeguards-summary",
        float_format="%.4f",
    )
    print(summary.to_string(index=False))
    print(f"  → wrote tbl_ch5_safeguards_summary.{{csv,tex}}")

    print("\n[Phase 4] step 13/14 — SHAP-vs-DiD consistency paragraph")
    consistency = ds.shap_vs_did_consistency(
        cfg.TABLES / "tbl_ch4_serbia_shap.csv",
        main_result,
    )
    print(consistency)
    (cfg.METRICS / "shap_vs_did_consistency.txt").write_text(consistency)
    print(f"  → wrote shap_vs_did_consistency.txt")

    print("\n[Phase 4] step 14/14 — verify canonical model file is unchanged")
    sha_after = _sha256(booster_path)
    print(f"  SHA-256 (after):  {sha_after}")
    if sha_before != sha_after:
        raise AssertionError(
            "models/xgb_best.joblib changed during Phase 4 run.\n"
            f"  before: {sha_before}\n  after:  {sha_after}"
        )
    print("  SHA-256 unchanged ✓")

    print("\n[Phase 4] done. Awaiting sign-off before next workstream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
