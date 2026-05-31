"""Scenario engine — three frozen scenarios + slider helper.

Reads the saved XGBoost booster (`models/xgb_best.joblib`) once,
runs `model.predict` on manipulated feature vectors. **No `.fit()`
anywhere.** Predictions are back-transformed via `np.expm1` so all
return values are on the EUR-thousands level scale.

PROTOCOL_FREEZE.md §13 #7 binding rule: every scenario output is
described as a *conditional prediction under manipulated inputs*,
never an *effect*. The `caveat` field of every return dict carries
the §13 #7 string verbatim.

The three frozen scenarios (Era's locked set, PROTOCOL_FREEZE §11):

    1. 20% Serbia tariff — toggle serbia_tariff = 1 on the XS row.
       The dummy was trained on the 2018-2020 100% regime; the
       booster cannot distinguish 20% from 100%. The "20%" label is
       the policy story; the model manipulation is the same binary
       toggle.
    2. Kosovo GDP +5% — multiply Kosovo's GDP by 1.05 for the base
       year (ln_kosovo_gdp shifts by log(1.05) ≈ 0.0488 across all
       partner rows, which share the same Kosovo macro by construction).
    3. Hypothetical Turkey FTA — toggle cefta_member = 1 on the TR
       row. Turkey is NOT a CEFTA member; this is a proxy for an
       FTA-like preferential trade agreement, not a forecast that
       Turkey will join CEFTA.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as cfg


CAVEAT_BASE = (
    "Conditional prediction under manipulated inputs; "
    "not a causal estimate."
)
CAVEAT_SERBIA_TARIFF = (
    f"{CAVEAT_BASE} The serbia_tariff dummy is binary, so the booster "
    "cannot distinguish 20% from 100%."
)
CAVEAT_TURKEY_FTA = (
    f"{CAVEAT_BASE} Turkey is not a CEFTA member; the cefta_member toggle "
    "is a proxy for an FTA-like preferential trade agreement, not a "
    "forecast that Turkey will join CEFTA."
)


# =============================================================================
# 1. Booster loader
# =============================================================================

def load_xgb_for_scenarios() -> dict:
    """Load `models/xgb_best.joblib` (same bundle Phase 3 uses).

    Returns the dict persisted by `xgb_model.fit_holdout`:
    {model, feature_list, best_params, seed, last_train_year}.
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
# 2. Base feature matrix + baseline predictions
# =============================================================================

def base_feature_matrix(panel: pd.DataFrame, base_year: int = 2024) -> pd.DataFrame:
    """Restrict to base_year rows that survive the 12-feature dropna.

    Returns a per-partner DataFrame with the 12-column XGBoost feature
    matrix plus iso2 / partner_name / year / partner_id / actual
    imports. The exact row count is reported (verified at runtime, not
    hard-coded; expected ~107 for 2024 = 113 partners − partial-WDI
    drops, but the function does not assert any specific count).
    """
    base = panel[panel["year"] == int(base_year)].copy()
    feasible_mask = base[cfg.FEATURE_ORDER].notna().all(axis=1)
    sub = base.loc[feasible_mask].reset_index(drop=True).copy()
    return sub


def predict_baseline(panel: pd.DataFrame, base_year: int = 2024) -> pd.DataFrame:
    """Run the saved XGBoost on the unmodified base_year feature matrix.

    Returns: iso2, partner_name, year, imports_actual,
    imports_predicted_baseline, residual.
    """
    bundle = load_xgb_for_scenarios()
    model = bundle["model"]

    sub = base_feature_matrix(panel, base_year)
    X = sub[cfg.FEATURE_ORDER].astype(float)
    pred_log1p = np.asarray(model.predict(X), dtype=float)
    pred_lvl = np.expm1(pred_log1p)
    return pd.DataFrame({
        "iso2":          sub["iso2"].values,
        "partner_name":  sub["partner_name"].values,
        "year":          sub["year"].astype(int).values,
        "imports_actual": sub["imports_eur_thousands"].astype(float).values,
        "imports_predicted_baseline": pred_lvl,
        "residual":      sub["imports_eur_thousands"].astype(float).values - pred_lvl,
    })


def _predict_levels(model, X: pd.DataFrame) -> np.ndarray:
    """Back-transform booster log1p output to level scale (EUR thousands)."""
    return np.expm1(np.asarray(model.predict(X), dtype=float))


def _build_per_partner_table(
    sub: pd.DataFrame,
    baseline_pred: np.ndarray,
    scenario_pred: np.ndarray,
) -> pd.DataFrame:
    delta = scenario_pred - baseline_pred
    pct = np.where(np.abs(baseline_pred) > 1e-12,
                   100.0 * delta / baseline_pred,
                   np.nan)
    return pd.DataFrame({
        "iso2":          sub["iso2"].values,
        "partner_name":  sub["partner_name"].values,
        "imports_actual": sub["imports_eur_thousands"].astype(float).values,
        "baseline_pred": baseline_pred,
        "scenario_pred": scenario_pred,
        "delta_eur_thousands": delta,
        "delta_pct":     pct,
    })


def _top10_by_abs_delta(per_partner: pd.DataFrame) -> pd.DataFrame:
    return (per_partner.assign(_abs=per_partner["delta_eur_thousands"].abs())
                       .sort_values("_abs", ascending=False)
                       .drop(columns=["_abs"])
                       .head(10)
                       .reset_index(drop=True))


def _serialise(d: dict) -> dict:
    """JSON-serialisable copy: convert DataFrames → list-of-records,
    numpy arrays → lists, numpy scalars → native Python."""
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[k] = v.to_dict(orient="records")
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def _persist(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialise(payload), f, indent=2, default=str)


# =============================================================================
# 3. The three frozen scenarios (PROTOCOL_FREEZE §11)
# =============================================================================

def scenario_serbia_tariff(panel: pd.DataFrame, base_year: int = 2024) -> dict:
    """Scenario #1: toggle serbia_tariff = 1 on the XS row.

    The booster's serbia_tariff is a binary dummy learned on the
    2018-2020 100% regime. The "20%" label is policy framing; the
    manipulation is the binary toggle. Caveat string in the return
    dict makes this explicit.
    """
    bundle = load_xgb_for_scenarios()
    model = bundle["model"]
    sub = base_feature_matrix(panel, base_year)

    X_base = sub[cfg.FEATURE_ORDER].astype(float).copy()
    baseline_pred = _predict_levels(model, X_base)

    X_scn = X_base.copy()
    serbia_mask = (sub["iso2"].values == "XS")
    if not serbia_mask.any():
        raise ValueError(
            f"Scenario serbia_tariff: no XS row in base_year={base_year} "
            "after the 12-feature dropna; cannot run the scenario."
        )
    X_scn.loc[serbia_mask, "serbia_tariff"] = 1.0
    scenario_pred = _predict_levels(model, X_scn)

    per_partner = _build_per_partner_table(sub, baseline_pred, scenario_pred)
    top10 = _top10_by_abs_delta(per_partner)

    payload = {
        "scenario_id":   "serbia_tariff_20pct_proxy",
        "scenario_label": "20% Serbia tariff (binary proxy — booster cannot distinguish 20% from 100%)",
        "base_year":     int(base_year),
        "n_partners":    int(len(sub)),
        "feature_changed": "serbia_tariff",
        "manipulation":  "serbia_tariff = 1 on XS row only",
        "per_partner":   per_partner,
        "summary_top10": top10,
        "caveat":        CAVEAT_SERBIA_TARIFF,
    }
    _persist(cfg.METRICS / "scenario_serbia_tariff.json", payload)
    return payload


def scenario_kosovo_gdp_plus5(panel: pd.DataFrame, base_year: int = 2024) -> dict:
    """Scenario #2: Kosovo GDP +5%.

    `ln_kosovo_gdp` varies only by year (Kosovo macro is the same
    across all 113 partner rows for a given year). +5% on the level
    shifts ln_kosovo_gdp by log(1.05) ≈ 0.04879 across every partner
    row simultaneously.
    """
    bundle = load_xgb_for_scenarios()
    model = bundle["model"]
    sub = base_feature_matrix(panel, base_year)

    X_base = sub[cfg.FEATURE_ORDER].astype(float).copy()
    baseline_pred = _predict_levels(model, X_base)

    X_scn = X_base.copy()
    shift = float(np.log(1.05))   # ≈ 0.04879
    X_scn["ln_kosovo_gdp"] = X_scn["ln_kosovo_gdp"] + shift
    scenario_pred = _predict_levels(model, X_scn)

    per_partner = _build_per_partner_table(sub, baseline_pred, scenario_pred)
    top10 = _top10_by_abs_delta(per_partner)

    payload = {
        "scenario_id":   "kosovo_gdp_plus5",
        "scenario_label": "Kosovo GDP +5%",
        "base_year":     int(base_year),
        "n_partners":    int(len(sub)),
        "feature_changed": "ln_kosovo_gdp",
        "manipulation":  f"ln_kosovo_gdp += log(1.05) ≈ {shift:.6f} on every row",
        "per_partner":   per_partner,
        "summary_top10": top10,
        "caveat":        CAVEAT_BASE,
    }
    _persist(cfg.METRICS / "scenario_kosovo_gdp_plus5.json", payload)
    return payload


def scenario_turkey_fta(panel: pd.DataFrame, base_year: int = 2024) -> dict:
    """Scenario #3: hypothetical Turkey FTA.

    Toggle cefta_member = 1 on the TR row only. Turkey is NOT a CEFTA
    member; this is a proxy for an FTA-like preferential agreement,
    not a forecast.
    """
    bundle = load_xgb_for_scenarios()
    model = bundle["model"]
    sub = base_feature_matrix(panel, base_year)

    X_base = sub[cfg.FEATURE_ORDER].astype(float).copy()
    baseline_pred = _predict_levels(model, X_base)

    X_scn = X_base.copy()
    tr_mask = (sub["iso2"].values == "TR")
    if not tr_mask.any():
        raise ValueError(
            f"Scenario turkey_fta: no TR row in base_year={base_year} "
            "after the 12-feature dropna; cannot run the scenario."
        )
    X_scn.loc[tr_mask, "cefta_member"] = 1.0
    scenario_pred = _predict_levels(model, X_scn)

    per_partner = _build_per_partner_table(sub, baseline_pred, scenario_pred)
    top10 = _top10_by_abs_delta(per_partner)

    payload = {
        "scenario_id":   "turkey_fta_proxy",
        "scenario_label": "Hypothetical Turkey FTA (cefta_member toggled on TR — proxy)",
        "base_year":     int(base_year),
        "n_partners":    int(len(sub)),
        "feature_changed": "cefta_member",
        "manipulation":  "cefta_member = 1 on TR row only",
        "per_partner":   per_partner,
        "summary_top10": top10,
        "caveat":        CAVEAT_TURKEY_FTA,
    }
    _persist(cfg.METRICS / "scenario_turkey_fta.json", payload)
    return payload


# =============================================================================
# 4. Generic slider helper for the dashboard
# =============================================================================

def slider_scenario(
    panel: pd.DataFrame,
    base_year: int = 2024,
    *,
    serbia_tariff_xs: bool = False,
    saa_in_force_overrides: dict | None = None,
    covid_overrides: dict | None = None,
    cefta_member_overrides: dict | None = None,
    ln_kosovo_gdp_shift: float = 0.0,
    ln_partner_gdp_overrides: dict | None = None,
) -> dict:
    """User-driven slider variant.

    Accepts any subset of the binary policy toggles plus continuous
    overrides for the macro features. The dashboard wraps this in
    `@st.cache_data` keyed on the call args. Caveat string is the
    base §13 #7 wording (the slider page repeats it visually).

    Override semantics:
      - `serbia_tariff_xs`: True flips serbia_tariff = 1 on the XS row.
      - `*_overrides` dicts map iso2 → 0/1 to flip the corresponding
        binary feature on that partner's row.
      - `ln_kosovo_gdp_shift`: additive shift in log space, applied to
        every row (Kosovo macro is panel-wide).
      - `ln_partner_gdp_overrides`: iso2 → new ln_partner_gdp value
        (replaces, not shifts).
    """
    bundle = load_xgb_for_scenarios()
    model = bundle["model"]
    sub = base_feature_matrix(panel, base_year)
    X_base = sub[cfg.FEATURE_ORDER].astype(float).copy()
    baseline_pred = _predict_levels(model, X_base)

    X_scn = X_base.copy()

    if serbia_tariff_xs:
        X_scn.loc[sub["iso2"].values == "XS", "serbia_tariff"] = 1.0

    for col, overrides in (
        ("saa_in_force", saa_in_force_overrides),
        ("covid",        covid_overrides),
        ("cefta_member", cefta_member_overrides),
    ):
        if not overrides:
            continue
        for iso2, val in overrides.items():
            mask = (sub["iso2"].values == iso2)
            if mask.any():
                X_scn.loc[mask, col] = float(val)

    if ln_kosovo_gdp_shift:
        X_scn["ln_kosovo_gdp"] = X_scn["ln_kosovo_gdp"] + float(ln_kosovo_gdp_shift)

    if ln_partner_gdp_overrides:
        for iso2, val in ln_partner_gdp_overrides.items():
            mask = (sub["iso2"].values == iso2)
            if mask.any():
                X_scn.loc[mask, "ln_partner_gdp"] = float(val)

    scenario_pred = _predict_levels(model, X_scn)
    per_partner = _build_per_partner_table(sub, baseline_pred, scenario_pred)
    top10 = _top10_by_abs_delta(per_partner)

    return {
        "scenario_id":  "slider_custom",
        "base_year":    int(base_year),
        "n_partners":   int(len(sub)),
        "per_partner":  per_partner,
        "summary_top10": top10,
        "caveat":       CAVEAT_BASE,
    }


# =============================================================================
# 5. What-if simulator (dashboard interactive panel)
# =============================================================================

# The export panel stores exports in place of imports in the lag column; the
# lag feature name is therefore the same for both flows.
WHATIF_LAG_FEATURE = "lagged_imports_log1p"

_FLOW_PANEL = {
    "import": "panel_bilateral.parquet",
    "export": "panel_bilateral_export.parquet",
}
_FLOW_BOOSTER = {
    "import": "xgb_best.joblib",
    "export": "xgb_best_export.joblib",
}
_FLOW_ACTUAL_COL = {
    "import": "imports_eur_thousands",
    "export": "exports_eur_thousands",
}


def load_panel_for_flow(flow: str) -> pd.DataFrame:
    """Load the bilateral panel for the given flow direction.

    Parameters
    ----------
    flow : {"import", "export"}
    """
    if flow not in _FLOW_PANEL:
        raise ValueError(f"flow must be 'import' or 'export', got {flow!r}")
    return pd.read_parquet(cfg.DATA_PROCESSED / _FLOW_PANEL[flow])


def _load_booster(flow: str) -> dict:
    """Load the saved XGBoost bundle for the given flow direction."""
    import joblib
    if flow not in _FLOW_BOOSTER:
        raise ValueError(f"flow must be 'import' or 'export', got {flow!r}")
    p = cfg.MODELS / _FLOW_BOOSTER[flow]
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found at {p}. "
            "Run the corresponding training phase first."
        )
    return joblib.load(p)


def _build_whatif_per_partner(
    sub: pd.DataFrame,
    flow: str,
    baseline_pred: np.ndarray,
    scenario_pred: np.ndarray,
) -> pd.DataFrame:
    """Per-partner results table with a flow column and a flow-neutral actual_trade column."""
    actual_col = _FLOW_ACTUAL_COL[flow]
    delta = scenario_pred - baseline_pred
    pct = np.where(np.abs(baseline_pred) > 1e-12,
                   100.0 * delta / baseline_pred,
                   np.nan)
    return pd.DataFrame({
        "flow":               flow,
        "iso2":               sub["iso2"].values,
        "partner_name":       sub["partner_name"].values,
        "actual_trade":       sub[actual_col].astype(float).values,
        "baseline_pred":      baseline_pred,
        "scenario_pred":      scenario_pred,
        "delta_eur_thousands": delta,
        "delta_pct":          pct,
    })


def whatif_scenario(
    *,
    flow: str = "import",
    base_year: int = 2024,
    partner_iso2: str | None = None,
    partner_gdp_pct: float = 0.0,
    lagged_trade_pct: float = 0.0,
    panel: pd.DataFrame | None = None,
) -> dict:
    """Interactive what-if helper for the dashboard simulator panel.

    Shifts ``ln_partner_gdp`` and/or the lagged-trade feature for a
    selected partner (or all partners) and re-runs the saved booster.
    No re-fitting occurs.

    Parameters
    ----------
    flow : {"import", "export"}
    base_year : int
        Year whose feature matrix is used as the counterfactual baseline.
    partner_iso2 : str or None
        ISO2 of the partner to perturb. ``None`` applies both shocks to
        every partner row simultaneously.
    partner_gdp_pct : float
        Percentage change in the partner's GDP level, e.g. 10.0 → +10%.
        The feature shift is ``np.log1p(partner_gdp_pct / 100)``.
    lagged_trade_pct : float
        Percentage change in lagged trade, e.g. 20.0 → +20%.
    panel : pd.DataFrame or None
        Pre-loaded panel (skips the parquet read when called in a loop).

    Returns
    -------
    dict with keys: scenario_id, flow, base_year, partner_iso2,
    partner_gdp_pct, lagged_trade_pct, n_partners, per_partner,
    summary_top10, caveat.
    """
    if flow not in {"import", "export"}:
        raise ValueError(f"flow must be 'import' or 'export', got {flow!r}")
    if partner_gdp_pct <= -100.0:
        raise ValueError(
            f"partner_gdp_pct must be > -100 (implies non-negative level); "
            f"got {partner_gdp_pct}"
        )
    if lagged_trade_pct <= -100.0:
        raise ValueError(
            f"lagged_trade_pct must be > -100 (implies non-negative level); "
            f"got {lagged_trade_pct}"
        )

    if panel is None:
        panel = load_panel_for_flow(flow)

    bundle = _load_booster(flow)
    model = bundle["model"]

    sub = base_feature_matrix(panel, base_year)
    X = sub[cfg.FEATURE_ORDER].astype(float).copy()
    baseline_pred = _predict_levels(model, X)

    if partner_iso2 is not None:
        partner_mask = sub["iso2"].values == partner_iso2
        if not partner_mask.any():
            raise ValueError(
                f"Partner {partner_iso2!r} not found in {flow} panel for "
                f"base_year={base_year} after the 12-feature dropna."
            )
    else:
        partner_mask = np.ones(len(sub), dtype=bool)

    X_scn = X.copy()

    if partner_gdp_pct != 0.0:
        X_scn.loc[partner_mask, "ln_partner_gdp"] += np.log1p(partner_gdp_pct / 100.0)

    if lagged_trade_pct != 0.0:
        old = X_scn.loc[partner_mask, WHATIF_LAG_FEATURE].values
        X_scn.loc[partner_mask, WHATIF_LAG_FEATURE] = np.log1p(
            np.expm1(old) * (1.0 + lagged_trade_pct / 100.0)
        )

    scenario_pred = _predict_levels(model, X_scn)
    per_partner = _build_whatif_per_partner(sub, flow, baseline_pred, scenario_pred)
    top10 = _top10_by_abs_delta(per_partner)

    return {
        "scenario_id":      "whatif_custom",
        "flow":             flow,
        "base_year":        int(base_year),
        "partner_iso2":     partner_iso2,
        "partner_gdp_pct":  float(partner_gdp_pct),
        "lagged_trade_pct": float(lagged_trade_pct),
        "n_partners":       int(len(sub)),
        "per_partner":      per_partner,
        "summary_top10":    top10,
        "caveat":           CAVEAT_BASE,
    }
