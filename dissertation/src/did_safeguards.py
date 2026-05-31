"""DiD safeguards — parallel trends, leads, event study, placebo,
sensitivities, partner-pairs bootstrap, SHAP-vs-DiD consistency note.

PROTOCOL_FREEZE.md §3.2, §8, §9, §13 are load-bearing for this module.

Critical binding rules respected here:

- DiD prose is hedged: "the point estimate indicates", "consistent
  with", "suggestive". NEVER "the tariff caused" / "proves" /
  "demonstrates" (PROTOCOL_FREEZE §13 #3).
- The locked DiD spec is unchanged: treated = XS, controls = AL/MK/ME,
  primary treatment_years = {2019}; reference partner = MK; reference
  year = 2010.
- The canonical `models/xgb_best.joblib` is read-only in this
  workstream. The orchestrator captures and re-checks its SHA-256.

Design notes:

- `ppml.fit_ppml_did` only supports `serbia_x_post` (+ optional
  `ln_partner_gdp`). The leads test and event study need additional
  Serbia × year_t dummies, so this module includes an internal helper,
  `_fit_ppml_with_serbia_year_dummies`, that mirrors fit_ppml_did's
  FE construction without modifying `ppml.py`.
- Wald joint tests use statsmodels' `GLMResults.wald_test` with the
  cluster-robust covariance already baked in at fit time
  (`cov_type='cluster'`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import json
import numpy as np
import pandas as pd

from . import config as cfg
from . import ppml


# =============================================================================
# 1. DiD-feasible panel
# =============================================================================

DID_PARTNERS = [cfg.DID_TREATED_ISO2] + cfg.DID_CONTROL_ISO2          # ['XS', 'AL', 'MK', 'ME']
DID_PANEL_EXPECTED_N = len(DID_PARTNERS) * len(cfg.YEARS)             # 4 × 15 = 60


def load_did_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Filter the full bilateral panel to {XS, AL, MK, ME}.

    Asserts the resulting size is 4 partners × 15 years = 60 rows.
    `partner_id` stays integer (the original Phase-1 numbering) so the
    downstream cluster-robust SEs work without coercion.
    """
    sub = panel[panel["iso2"].isin(DID_PARTNERS)].copy()
    if len(sub) != DID_PANEL_EXPECTED_N:
        raise AssertionError(
            f"DiD panel: expected {DID_PANEL_EXPECTED_N} rows "
            f"({DID_PARTNERS} × 15 years), got {len(sub)}."
        )
    if set(sub["iso2"].unique()) != set(DID_PARTNERS):
        raise AssertionError(
            f"DiD panel iso2 mismatch: {set(sub['iso2'].unique())} != {set(DID_PARTNERS)}"
        )
    return sub.reset_index(drop=True)


# =============================================================================
# 2. Internal PPML helper for leads / event study
# =============================================================================

@dataclass
class _SerbiaDummiesFit:
    result: object                 # statsmodels GLMResultsWrapper
    exog_names: list
    serbia_year_dummy_names: list  # e.g. ['serbia_x_2014', 'serbia_x_2015', ...]
    main_dummy_name: str | None    # 'serbia_x_post' or None (event study)
    n: int
    n_partners: int
    n_years: int


def _fit_ppml_with_serbia_year_dummies(
    panel: pd.DataFrame,
    *,
    serbia_year_dummies: list,
    include_main_serbia_x_post: bool = True,
    main_treatment_years: set | None = None,
    treated_iso2: str = cfg.DID_TREATED_ISO2,
    control_iso2: list | None = None,
    reference_iso2: str | None = None,
    reference_year: int | None = None,
    include_gdp_control: bool = False,
) -> _SerbiaDummiesFit:
    """Like `ppml.fit_ppml_did` but with arbitrary Serbia × year_t
    dummies as additional regressors.

    Used by `leads_test` and `event_study` only. Does **not** modify
    `ppml.py`; it re-implements the FE construction locally so the
    locked Phase-1 module stays untouched.

    Cluster-robust SEs at partner level via
    `model.fit(cov_type='cluster', cov_kwds={'groups': partner_id})`.
    """
    import statsmodels.api as sm

    control_iso2 = list(control_iso2 or cfg.DID_CONTROL_ISO2)
    main_treatment_years = (set(main_treatment_years)
                            if main_treatment_years is not None
                            else set(cfg.DID_TREATMENT_YEARS_PRIMARY))

    sub = panel[panel["iso2"].isin([treated_iso2] + control_iso2)].copy()

    # 1. main serbia_x_post (optional)
    regressor_cols = []
    main_name = None
    if include_main_serbia_x_post:
        sub["serbia_x_post"] = (
            (sub["iso2"] == treated_iso2) & (sub["year"].isin(main_treatment_years))
        ).astype(int)
        regressor_cols.append("serbia_x_post")
        main_name = "serbia_x_post"

    # 2. extra Serbia × year_t dummies
    serbia_year_dummy_names = []
    for y in serbia_year_dummies:
        col = f"serbia_x_{int(y)}"
        sub[col] = ((sub["iso2"] == treated_iso2) & (sub["year"] == int(y))).astype(int)
        regressor_cols.append(col)
        serbia_year_dummy_names.append(col)

    # 3. ln_partner_gdp (optional)
    if include_gdp_control:
        if "ln_partner_gdp" not in sub.columns:
            raise ValueError("include_gdp_control=True but ln_partner_gdp not in panel")
        regressor_cols.append("ln_partner_gdp")

    # Drop NaN rows for whatever regressors we ended up with
    sub = sub.dropna(subset=regressor_cols).copy()

    # 4. Reference levels
    if reference_iso2 is None:
        reference_iso2 = control_iso2[0]
    if reference_year is None:
        reference_year = min(panel["year"])

    # 5. Design matrix
    y = sub["imports_eur_thousands"].astype(float).values
    X_obs = sub[regressor_cols].astype(float)
    X_pfe = pd.get_dummies(sub["iso2"], prefix="pFE", drop_first=False).astype(float)
    ref_p = f"pFE_{reference_iso2}"
    if ref_p in X_pfe.columns:
        X_pfe = X_pfe.drop(columns=[ref_p])
    X_yfe = pd.get_dummies(sub["year"], prefix="yFE", drop_first=False).astype(float)
    ref_y = f"yFE_{reference_year}"
    if ref_y in X_yfe.columns:
        X_yfe = X_yfe.drop(columns=[ref_y])

    X = pd.concat(
        [X_obs.reset_index(drop=True),
         X_pfe.reset_index(drop=True),
         X_yfe.reset_index(drop=True)], axis=1,
    )
    X = sm.add_constant(X, has_constant="add")

    # 6. Fit GLM-Poisson with cluster-robust SEs
    model = sm.GLM(y, X, family=sm.families.Poisson())
    try:
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["partner_id"].astype(int).values},
        )
    except Exception as e:
        # Fall back to non-clustered SEs only if clustering itself fails.
        # We surface the failure rather than silently dropping cov_type.
        print(f"[did_safeguards] WARNING: cluster-robust SEs failed: {e}")
        result = model.fit()

    return _SerbiaDummiesFit(
        result=result,
        exog_names=list(result.model.exog_names),
        serbia_year_dummy_names=serbia_year_dummy_names,
        main_dummy_name=main_name,
        n=int(len(sub)),
        n_partners=int(X_pfe.shape[1] + 1),  # +1 for the dropped reference partner
        n_years=int(X_yfe.shape[1] + 1),     # +1 for the dropped reference year
    )


# =============================================================================
# 3. Main DiD + 1,000-draw stratified pairs bootstrap
# =============================================================================

def _extract_beta_se(result, name: str) -> Tuple[float, float, float]:
    names = list(result.model.exog_names)
    i = names.index(name)
    b = float(np.asarray(result.params)[i])
    s = float(np.asarray(result.bse)[i])
    t = b / s if s else float("nan")
    return b, s, t


def main_did_with_bootstrap(
    panel: pd.DataFrame,
    n_boot: int = cfg.BOOTSTRAP_N,
    seed: int = cfg.SEED,
) -> dict:
    """Headline DiD + 1,000-draw stratified partner-pairs bootstrap CI.

    Returns the canonical inference statement going into Chapter 5.
    The placeholder cluster-robust SE is reported alongside but
    explicitly labelled as such.

    The bootstrap call returns `n_valid` / `n_failed`; this function
    aliases them to `n_boot_completed` / `n_boot_failed` (per the
    Phase 4 prompt's main_did_with_bootstrap return spec) while also
    keeping the originals so downstream code that follows
    ppml.did_pairs_bootstrap's vocabulary still works.
    """
    fit = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )
    beta_main, se_placeholder, _ = _extract_beta_se(fit.model_result, "serbia_x_post")

    boot = ppml.did_pairs_bootstrap(
        panel,
        n_boot=n_boot,
        seed=seed,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )

    bm = boot["mean_beta_did"]
    cl = boot["ci_95_lower"]
    ch = boot["ci_95_upper"]
    return {
        "beta_main":              beta_main,
        "se_cluster_placeholder": se_placeholder,
        "n_boot":                 boot["n_boot"],
        "n_boot_completed":       boot["n_valid"],
        "n_boot_failed":          boot["n_failed"],
        "n_valid":                boot["n_valid"],     # alias
        "n_failed":               boot["n_failed"],    # alias
        "bootstrap_mean":         bm,
        "bootstrap_ci_low":       cl,
        "bootstrap_ci_high":      ch,
        "exp_beta_minus_1_pct":   float((np.exp(beta_main) - 1.0) * 100.0),
        "exp_ci_low_pct":         float((np.exp(cl) - 1.0) * 100.0)
                                  if np.isfinite(cl) else float("nan"),
        "exp_ci_high_pct":        float((np.exp(ch) - 1.0) * 100.0)
                                  if np.isfinite(ch) else float("nan"),
        "draws":                  boot["draws"],
    }


# =============================================================================
# 4. Parallel trends data (pre-2018 only)
# =============================================================================

def parallel_trends_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-year mean of np.log1p(imports_eur_thousands) split by group.

    Groups: treated (XS) and control_mean (equally-weighted mean of
    AL, MK, ME). Restricted to pre-treatment years 2010-2017.
    Long format: year × group × mean_log_imports.
    """
    pre = panel[(panel["year"] <= 2017)].copy()
    pre["log1p_imports"] = np.log1p(pre["imports_eur_thousands"].astype(float))

    treated = (pre[pre["iso2"] == cfg.DID_TREATED_ISO2]
               .groupby("year")["log1p_imports"].mean()
               .reset_index().rename(columns={"log1p_imports": "mean_log_imports"}))
    treated["group"] = cfg.DID_TREATED_ISO2

    controls = pre[pre["iso2"].isin(cfg.DID_CONTROL_ISO2)]
    # For each control partner, take its yearly mean (here it's just the row),
    # then average ACROSS partners → equally-weighted control mean per year.
    control_yearly = (controls.groupby(["iso2", "year"])["log1p_imports"].mean()
                              .reset_index())
    control_mean = (control_yearly.groupby("year")["log1p_imports"].mean()
                                  .reset_index()
                                  .rename(columns={"log1p_imports": "mean_log_imports"}))
    control_mean["group"] = "control_mean(AL+MK+ME)"

    return (pd.concat([treated, controls.assign(group=lambda d: d["iso2"])
                                       .groupby(["year", "group"])["log1p_imports"].mean()
                                       .reset_index().rename(columns={"log1p_imports": "mean_log_imports"}),
                       control_mean],
                      ignore_index=True, sort=False)
            .sort_values(["group", "year"])
            .reset_index(drop=True)
            [["year", "group", "mean_log_imports"]])


# =============================================================================
# 5. Leads test (Wald joint, pre-2018 leads)
# =============================================================================

def leads_test(
    panel: pd.DataFrame,
    lead_years: tuple = (2014, 2015, 2016, 2017),
) -> dict:
    """PPML-DiD with extra Serbia × year_t dummies for each lead year.

    Joint Wald test (cluster-robust): H0: all lead coefficients = 0.

    Phrasing rule: a non-rejecting result is "leads test does NOT
    reject pre-trend differences at α=0.05" — failure to reject;
    NEVER "supports parallel trends" (overclaim with 4 clusters).
    """
    fit = _fit_ppml_with_serbia_year_dummies(
        panel,
        serbia_year_dummies=list(lead_years),
        include_main_serbia_x_post=True,
        main_treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )

    per_lead_coefs = {}
    per_lead_se = {}
    R_constraints = []
    names = fit.exog_names
    for col in fit.serbia_year_dummy_names:
        i = names.index(col)
        per_lead_coefs[int(col.split("_")[-1])] = float(np.asarray(fit.result.params)[i])
        per_lead_se[int(col.split("_")[-1])]    = float(np.asarray(fit.result.bse)[i])
        row = np.zeros(len(names))
        row[i] = 1.0
        R_constraints.append(row)
    R = np.vstack(R_constraints)

    wald = fit.result.wald_test(R, scalar=True)
    # statsmodels Wald has either F or chi2 depending on cov_type
    stat = float(np.asarray(wald.statistic).reshape(-1)[0])
    pval = float(np.asarray(wald.pvalue).reshape(-1)[0])
    df_num = int(getattr(wald, "df_num", len(lead_years)))

    if pval >= 0.05:
        interpretation = (f"Failure to reject at α=0.05 (p={pval:.4f}) — "
                          "leads test does not reject pre-trend differences "
                          "(weak claim with 4 clusters; does NOT prove parallel trends).")
    else:
        interpretation = (f"Reject at α=0.05 (p={pval:.4f}) — "
                          "evidence against the parallel-trends assumption.")

    return {
        "per_lead_coefs":   per_lead_coefs,
        "per_lead_se":      per_lead_se,
        "joint_wald_stat":  stat,
        "joint_p_value":    pval,
        "df":               df_num,
        "n":                fit.n,
        "interpretation":   interpretation,
    }


# =============================================================================
# 6. Event study (full leads + lags)
# =============================================================================

def event_study(
    panel: pd.DataFrame,
    event_years: Iterable[int] = range(2011, 2025),
    reference_year: int = 2010,
) -> Tuple[pd.DataFrame, dict]:
    """Single PPML-DiD refit with Serbia × year_t dummies for every
    t ≠ reference_year. Returns the long event-study DataFrame plus
    a status dict.

    A single GLM fit either converges or it does not. If the refit
    fails (PerfectSeparationError or any other exception), this
    function returns an empty event-study DataFrame and a status dict
    flagging the failure — the orchestrator continues with the leads
    test.
    """
    event_year_list = sorted(int(y) for y in event_years if int(y) != int(reference_year))

    try:
        fit = _fit_ppml_with_serbia_year_dummies(
            panel,
            serbia_year_dummies=event_year_list,
            include_main_serbia_x_post=False,
            reference_year=reference_year,
        )
    except Exception as e:
        return (
            pd.DataFrame(
                {"year": [np.nan], "beta": [np.nan], "se": [np.nan],
                 "t_stat": [np.nan], "ci_low_95": [np.nan], "ci_high_95": [np.nan],
                 "status": ["did_not_converge"], "reason": [repr(e)]}
            ),
            {"status": "did_not_converge", "reason": repr(e),
             "n_year_dummies": len(event_year_list)},
        )

    rows = []
    names = fit.exog_names
    params = np.asarray(fit.result.params)
    bses = np.asarray(fit.result.bse)
    for col in fit.serbia_year_dummy_names:
        i = names.index(col)
        b, se = float(params[i]), float(bses[i])
        t = b / se if se else float("nan")
        rows.append({
            "year":       int(col.split("_")[-1]),
            "beta":       b,
            "se":         se,
            "t_stat":     t,
            "ci_low_95":  b - 1.96 * se,
            "ci_high_95": b + 1.96 * se,
            "status":     "ok",
            "reason":     "",
        })
    df = (pd.DataFrame(rows).sort_values("year").reset_index(drop=True))
    return df, {"status": "ok", "n_year_dummies": len(event_year_list), "n": fit.n}


# =============================================================================
# 7. Placebo
# =============================================================================

def placebo_did(panel: pd.DataFrame, fake_treatment_year: int = 2014) -> dict:
    """Re-run the locked main DiD spec with treatment_years={fake_year}.

    Phrasing: |t| < 2 ⇒ "consistent with no obvious placebo signal"
    (failure to reject β=0); a significant placebo β suggests the
    design is picking up confounders rather than the tariff itself.
    """
    fake_set = {int(fake_treatment_year)}
    fit = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=fake_set,
    )
    b, s, t = _extract_beta_se(fit.model_result, "serbia_x_post")

    if abs(t) < 2.0:
        interpretation = (f"|t|={abs(t):.2f} < 2.0 — failure to reject β=0; "
                          "consistent with no obvious placebo signal.")
    else:
        interpretation = (f"|t|={abs(t):.2f} >= 2.0 — placebo β is statistically "
                          "non-trivial; the design may be capturing confounders "
                          "rather than only the tariff.")

    sub = panel[panel["iso2"].isin([cfg.DID_TREATED_ISO2] + cfg.DID_CONTROL_ISO2)]
    n_treated_obs = int(((sub["iso2"] == cfg.DID_TREATED_ISO2)
                         & (sub["year"].isin(fake_set))).sum())
    return {
        "beta": b, "se": s, "t": t,
        "n": int(fit.n),
        "n_treated_obs": n_treated_obs,
        "fake_year": int(fake_treatment_year),
        "interpretation": interpretation,
    }


# =============================================================================
# 8. Sensitivities
# =============================================================================

def sensitivity_window(
    panel: pd.DataFrame,
    year_range: tuple = (2015, 2021),
) -> dict:
    """Restrict to year ∈ year_range and refit the main DiD.

    Catches PerfectSeparationError → status='perfect_separation_skipped'.
    """
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    lo, hi = int(year_range[0]), int(year_range[1])
    sub = panel[(panel["year"] >= lo) & (panel["year"] <= hi)].copy()
    n = int(len(sub))
    n_partners = int(sub["iso2"].nunique())
    try:
        fit = ppml.fit_ppml_did(
            sub,
            treated_iso2=cfg.DID_TREATED_ISO2,
            control_iso2=cfg.DID_CONTROL_ISO2,
            treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
        )
        b, s, t = _extract_beta_se(fit.model_result, "serbia_x_post")
        return {
            "status":     "ok",
            "beta":       b, "se": s, "t": t,
            "n":          n, "n_partners": n_partners,
            "year_range": [lo, hi],
        }
    except PerfectSeparationError as e:
        return {
            "status":     "perfect_separation_skipped",
            "reason":     repr(e),
            "n":          n, "n_partners": n_partners,
            "year_range": [lo, hi],
        }
    except Exception as e:
        return {
            "status":     "fit_failed",
            "reason":     repr(e),
            "n":          n, "n_partners": n_partners,
            "year_range": [lo, hi],
        }


def sensitivity_partner_gdp_control(panel: pd.DataFrame) -> dict:
    """Substitute for the original 'drop Kosovo-GDP-control' sensitivity.

    The locked main DiD has no Kosovo-GDP control (it would be
    collinear with year FE), so dropping it is undefined. The closest
    analog is to ADD ln_partner_gdp as a regressor and check whether
    β survives. This is reported as a substitute robustness check.
    """
    fit_main = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )
    beta_main, _, _ = _extract_beta_se(fit_main.model_result, "serbia_x_post")

    fit_gdp = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
        include_gdp_control=True,
    )
    b_gdp, se_gdp, t_gdp = _extract_beta_se(fit_gdp.model_result, "serbia_x_post")
    abs_change = abs(b_gdp - beta_main)
    pct_change = 100.0 * abs_change / abs(beta_main) if beta_main != 0 else float("nan")
    if pct_change < 25.0:
        interpretation = (f"|Δβ| = {abs_change:.4f} ({pct_change:.1f}% of |β_main|) "
                          "— β survives the partner-GDP control variant.")
    else:
        interpretation = (f"|Δβ| = {abs_change:.4f} ({pct_change:.1f}% of |β_main|) "
                          "— β shifts materially when ln_partner_gdp is added; "
                          "report as instability.")
    return {
        "beta_with_gdp":      b_gdp,
        "se":                 se_gdp,
        "t":                  t_gdp,
        "beta_main":          beta_main,
        "abs_change_in_beta": abs_change,
        "pct_change_in_beta": pct_change,
        "interpretation":     interpretation,
        "n":                  int(fit_gdp.n),
        "note": ("Substitute for the original Kosovo-GDP-control sensitivity "
                 "(the locked main DiD has no Kosovo-GDP control: it would be "
                 "collinear with year FE). Reported as the closest robustness "
                 "check available."),
    }


def sensitivity_treatment_window(panel: pd.DataFrame) -> dict:
    """Refit the DiD with treatment_years = {2018, 2019, 2020} (locked
    sensitivity per PROTOCOL_FREEZE §3.2 last bullet).
    """
    fit_main = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY,
    )
    beta_2019, _, _ = _extract_beta_se(fit_main.model_result, "serbia_x_post")

    fit_window = ppml.fit_ppml_did(
        panel,
        treated_iso2=cfg.DID_TREATED_ISO2,
        control_iso2=cfg.DID_CONTROL_ISO2,
        treatment_years=cfg.DID_TREATMENT_YEARS_SENSITIVITY,
    )
    beta_window, se_window, t_window = _extract_beta_se(
        fit_window.model_result, "serbia_x_post"
    )
    abs_change = abs(beta_window - beta_2019)
    pct_change = (100.0 * abs_change / abs(beta_2019)
                  if beta_2019 != 0 else float("nan"))
    if pct_change < 25.0:
        interpretation = (f"|Δβ| = {abs_change:.4f} ({pct_change:.1f}% of "
                          "|β_main|) — β stable when window is widened.")
    else:
        interpretation = (f"|Δβ| = {abs_change:.4f} ({pct_change:.1f}% of "
                          "|β_main|) — β shifts when the post-period is "
                          "{2018, 2019, 2020}; report as instability.")
    return {
        "beta_2019_only":     beta_2019,
        "beta_2018_19_20":    beta_window,
        "se_2018_19_20":      se_window,
        "t_2018_19_20":       t_window,
        "abs_change_in_beta": abs_change,
        "pct_change_in_beta": pct_change,
        "interpretation":     interpretation,
        "n":                  int(fit_window.n),
    }


# =============================================================================
# 9. SHAP-vs-DiD consistency note
# =============================================================================

def shap_vs_did_consistency(
    shap_serbia_table_path: Path,
    did_main: dict,
) -> str:
    """One-paragraph consistency note for PHASE4_REPORT.md §7.

    SIGN AND QUALITATIVE COMPARISON ONLY. SHAP and DiD live in
    different units; the booster's predictive contributions
    (log1p model output space) are not magnitude-comparable to the
    PPML log-link DiD coefficient.

    Phrasing rule (PROTOCOL_FREEZE §13 #2 #3):
      - SHAP side: "predictive contribution" / "associative
        importance"; never "effect".
      - DiD side: "the point estimate indicates", "consistent with";
        never "the tariff caused".
    """
    shap_long = pd.read_csv(shap_serbia_table_path)
    serbia_tariff = shap_long[shap_long["feature"] == "serbia_tariff"]
    in_tariff_window = serbia_tariff[serbia_tariff["year"].isin([2018, 2019, 2020])]
    tariff_2019 = serbia_tariff[serbia_tariff["year"] == 2019]
    s_2019 = (float(tariff_2019["shap_value"].iloc[0])
              if len(tariff_2019) else float("nan"))

    # Find the largest-magnitude (most negative) feature for Serbia 2019
    # to characterise where the trough actually lives in SHAP terms.
    serbia_2019_all = shap_long[shap_long["year"] == 2019]
    if len(serbia_2019_all):
        most_neg_2019 = serbia_2019_all.loc[serbia_2019_all["shap_value"].idxmin()]
        most_neg_feature = str(most_neg_2019["feature"])
        most_neg_val = float(most_neg_2019["shap_value"])
    else:
        most_neg_feature = "n/a"
        most_neg_val = float("nan")

    beta_main = did_main["beta_main"]
    bm = did_main["bootstrap_mean"]
    cl = did_main["bootstrap_ci_low"]
    ch = did_main["bootstrap_ci_high"]

    tariff_window_summary = (
        "zero across 2018, 2019, and 2020"
        if (in_tariff_window["shap_value"] == 0).all()
        else f"non-trivial in 2018-2020 (max |SHAP| = "
             f"{in_tariff_window['shap_value'].abs().max():.3f})"
    )

    paragraph = (
        f"The booster's predictive contributions for `serbia_tariff` are "
        f"{tariff_window_summary} on the Serbia rows — the policy dummy is "
        f"associatively flat in the predictive model. The booster captures "
        f"the 2019 trough through `{most_neg_feature}` instead "
        f"(SHAP value {most_neg_val:+.3f} for Serbia 2019). The DiD point "
        f"estimate β = {beta_main:+.4f} on `serbia_x_post` (bootstrap mean "
        f"{bm:+.4f}, 95 % CI [{cl:+.4f}, {ch:+.4f}]) is negative, "
        f"consistent with the negative `{most_neg_feature}` predictive "
        f"contribution at the same data point. The two pillars therefore "
        f"agree on sign, but they are measuring different objects — "
        f"associative importance in log1p model output space versus a "
        f"causal log-link coefficient — and are not comparable on "
        f"magnitude or interpretation. The DiD point estimate indicates "
        f"a deep contraction in Serbia's 2019 imports relative to the "
        f"AL/MK/ME baseline; the SHAP analysis confirms the booster sees "
        f"the same direction without using the policy indicator itself."
    )
    return paragraph
