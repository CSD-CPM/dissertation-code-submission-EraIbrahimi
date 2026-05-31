"""Both PPML specifications.

Implementation plan §4:
- fit_ppml_predictive(panel)  -> partner FE + observable regressors, no year FE
- fit_ppml_did(panel)         -> partner FE + year FE + serbia_x_post

Both use statsmodels.GLM(y, X, family=sm.families.Poisson()) with
cluster-robust standard errors at the partner level. Zeros in imports are
retained (that is the whole point of PPML).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class PPMLResult:
    spec_name: str
    model_result: object
    n: int
    n_partner_fe: int
    n_year_fe: int
    zero_rows: int
    feature_list: list
    clustered: bool


def _build_partner_dummies(panel: pd.DataFrame, reference_iso2: str) -> pd.DataFrame:
    """One-hot partner dummies, dropping the reference level."""
    d = pd.get_dummies(panel["iso2"], prefix="pFE", drop_first=False).astype(float)
    ref_col = f"pFE_{reference_iso2}"
    if ref_col in d.columns:
        d = d.drop(columns=[ref_col])
    return d


def _build_year_dummies(panel: pd.DataFrame, reference_year: int) -> pd.DataFrame:
    d = pd.get_dummies(panel["year"], prefix="yFE", drop_first=False).astype(float)
    ref_col = f"yFE_{reference_year}"
    if ref_col in d.columns:
        d = d.drop(columns=[ref_col])
    return d


def fit_ppml_predictive(
    panel: pd.DataFrame,
    regressors: Iterable[str] = None,
    reference_iso2: str = "DE",
) -> PPMLResult:
    """Plan §4.1: partner FE, observable regressors, no year FE.

    Target: imports_eur_thousands (levels, not log — zeros retained natively).
    Cluster-robust SEs at partner level.
    """
    import statsmodels.api as sm

    regressors = list(regressors or cfg.PPML_PREDICTIVE_REGRESSORS)
    needed = set(regressors) | {"iso2", "year", "imports_eur_thousands", "partner_id"}
    missing = needed - set(panel.columns)
    if missing:
        raise ValueError(f"PPML-Predictive missing columns: {missing}")

    # Any rows with NaN in regressors get dropped with a log; PPML-Poisson
    # itself cannot handle NaNs, but zeros in y must stay.
    sub = panel.dropna(subset=list(regressors)).copy()
    n_dropped = len(panel) - len(sub)
    if n_dropped:
        print(f"[ppml-predictive] dropped {n_dropped} rows with NaN regressors "
              f"(usually 2010 lagged rows or missing WDI)")

    y = sub["imports_eur_thousands"].astype(float).values
    X_obs = sub[regressors].astype(float)
    X_fe = _build_partner_dummies(sub, reference_iso2=reference_iso2)
    X = pd.concat([X_obs.reset_index(drop=True), X_fe.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Poisson())
    # statsmodels GLM exposes cluster-robust SEs through fit(cov_type=...);
    # GLMResults has no .get_robustcov_results() (that is OLS-only).
    clustered = True
    try:
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["partner_id"].astype(int).values},
        )
    except Exception as e:
        print(f"[ppml-predictive] WARNING: cluster-robust SEs failed: {e}")
        result = model.fit()
        clustered = False

    return PPMLResult(
        spec_name="PPML-Predictive",
        model_result=result,
        n=int(len(sub)),
        n_partner_fe=int(X_fe.shape[1]),
        n_year_fe=0,
        zero_rows=int((sub["imports_eur_thousands"] == 0).sum()),
        feature_list=regressors,
        clustered=clustered,
    )


def fit_ppml_did(
    panel: pd.DataFrame,
    treated_iso2: str = cfg.DID_TREATED_ISO2,
    control_iso2: Iterable[str] = None,
    treatment_years: Iterable[int] = None,
    include_gdp_control: bool = False,
    reference_iso2: str | None = None,
    reference_year: int = None,
) -> PPMLResult:
    """Plan §4.2: partner FE + year FE + serbia_x_post interaction.

    Default treatment = {2019} (primary). Sensitivity refits should pass
    treatment_years={2018,2019,2020}. include_gdp_control=True adds
    ln_partner_gdp as a regressor (sensitivity 2).
    """
    import statsmodels.api as sm

    control_iso2 = list(control_iso2 or cfg.DID_CONTROL_ISO2)
    treatment_years = set(treatment_years or cfg.DID_TREATMENT_YEARS_PRIMARY)

    # Restrict to Serbia + controls
    sub = panel[panel["iso2"].isin([treated_iso2] + control_iso2)].copy()
    # Build treatment dummy
    sub["serbia_x_post"] = (
        (sub["iso2"] == treated_iso2) & (sub["year"].isin(treatment_years))
    ).astype(int)

    regressors = ["serbia_x_post"]
    if include_gdp_control:
        if "ln_partner_gdp" not in sub.columns:
            raise ValueError("include_gdp_control=True but ln_partner_gdp not in panel")
        regressors.append("ln_partner_gdp")

    sub = sub.dropna(subset=regressors).copy()
    if reference_iso2 is None:
        reference_iso2 = control_iso2[0]
    if reference_year is None:
        reference_year = min(panel["year"])

    y = sub["imports_eur_thousands"].astype(float).values
    X_obs = sub[regressors].astype(float)
    X_pfe = _build_partner_dummies(sub, reference_iso2=reference_iso2)
    X_yfe = _build_year_dummies(sub, reference_year=reference_year)
    X = pd.concat(
        [X_obs.reset_index(drop=True), X_pfe.reset_index(drop=True),
         X_yfe.reset_index(drop=True)], axis=1,
    )
    X = sm.add_constant(X, has_constant="add")

    model = sm.GLM(y, X, family=sm.families.Poisson())
    clustered = True
    try:
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["partner_id"].astype(int).values},
        )
    except Exception as e:
        print(f"[ppml-did] WARNING: cluster-robust SEs failed: {e}")
        result = model.fit()
        clustered = False

    return PPMLResult(
        spec_name=f"PPML-DiD(treat={sorted(treatment_years)})",
        model_result=result,
        n=int(len(sub)),
        n_partner_fe=int(X_pfe.shape[1]),
        n_year_fe=int(X_yfe.shape[1]),
        zero_rows=int((sub["imports_eur_thousands"] == 0).sum()),
        feature_list=regressors,
        clustered=clustered,
    )


# =============================================================================
# Pairs bootstrap — N = 1,000 (locked)
# =============================================================================

def did_pairs_bootstrap(
    panel: pd.DataFrame,
    n_boot: int = cfg.BOOTSTRAP_N,
    seed: int = cfg.SEED,
    **did_kwargs,
) -> dict:
    """Stratified pairs bootstrap for the Serbia DiD.

    With only 4 partners (XS + AL/MK/ME), naive resampling that drops
    the treated cluster destroys identification. We resample CONTROLS
    only with replacement and always preserve Serbia. partner_id stays
    integer so cluster-robust SEs work in every draw.

    Signature is preserved; existing return keys (n_boot, n_valid,
    mean_beta_did, ci_95_lower, ci_95_upper, draws) are preserved; the
    return dict is extended with `n_failed` for diagnostic use.
    """
    rng = np.random.default_rng(seed)
    treated = did_kwargs.get("treated_iso2", cfg.DID_TREATED_ISO2)
    controls = list(did_kwargs.get("control_iso2", cfg.DID_CONTROL_ISO2))

    # Untouched treated rows — Serbia stays as Serbia in every draw.
    treated_panel = panel[panel["iso2"] == treated].copy()
    if len(treated_panel) == 0:
        raise ValueError(f"Bootstrap: no rows for treated iso2={treated!r}")

    # Allocate a unique integer partner_id namespace far above any real id
    # so duplicated controls don't collide with the canonical numbering.
    PID_BASE = 9000

    draws = np.full(n_boot, np.nan)
    n_failed = 0
    for b in range(n_boot):
        sampled_controls = rng.choice(controls, size=len(controls), replace=True)
        boot_frames = [treated_panel]
        boot_control_iso2 = []
        for copy_idx, c in enumerate(sampled_controls):
            cdup = panel[panel["iso2"] == c].copy()
            new_iso2 = f"{c}_b{copy_idx}"
            new_pid = PID_BASE + b * 100 + copy_idx  # unique INT per draw, per copy
            cdup["iso2"] = new_iso2
            cdup["partner_id"] = int(new_pid)
            boot_frames.append(cdup)
            boot_control_iso2.append(new_iso2)
        boot_panel = pd.concat(boot_frames, ignore_index=True)

        # Always pin reference_iso2 to a name guaranteed to exist in the
        # bootstrap panel. If a caller passes a stale reference_iso2 via
        # did_kwargs (e.g. an original-named partner), the dummy column
        # would not be dropped and partner FE would become collinear.
        boot_kwargs = {**did_kwargs,
                       "treated_iso2":   treated,            # untouched 'XS'
                       "control_iso2":   boot_control_iso2,  # renamed copies
                       "reference_iso2": boot_control_iso2[0]}
        try:
            r = fit_ppml_did(boot_panel, **boot_kwargs)
            params = r.model_result.params
            names = (getattr(r.model_result, "model").exog_names
                     if hasattr(r.model_result, "model") else None)
            if names is None:
                names = list(getattr(params, "index", []))
            idx = names.index("serbia_x_post")
            draws[b] = float(np.asarray(params)[idx])
        except Exception:
            n_failed += 1

    valid = draws[~np.isnan(draws)]
    return {
        "n_boot": int(n_boot),
        "n_valid": int(len(valid)),
        "n_failed": int(n_failed),
        "mean_beta_did": float(np.mean(valid)) if len(valid) else float("nan"),
        "ci_95_lower": float(np.percentile(valid, 2.5)) if len(valid) else float("nan"),
        "ci_95_upper": float(np.percentile(valid, 97.5)) if len(valid) else float("nan"),
        "draws": draws,
    }
