"""Phase 4 (causal-diagnostics pillar) artefact integrity."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as cfg


def test_phase4_did_panel_60_rows():
    panel = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    did_panel = panel[panel["iso2"].isin(["XS", "AL", "MK", "ME"])]
    assert len(did_panel) == 60, (
        f"DiD panel: expected 60 rows (4 partners × 15 years); "
        f"got {len(did_panel)}."
    )
    assert set(did_panel["iso2"].unique()) == {"XS", "AL", "MK", "ME"}, (
        "DiD panel partner set must be exactly {XS, AL, MK, ME}."
    )


def test_phase4_bootstrap_completed():
    payload = json.loads((cfg.METRICS / "bootstrap_ci.json").read_text())
    n_completed = int(payload["n_boot_completed"])
    assert n_completed >= 950, (
        f"Bootstrap n_completed: expected ≥ 950 / 1000; got {n_completed}."
    )


def test_phase4_bootstrap_ci_finite():
    payload = json.loads((cfg.METRICS / "bootstrap_ci.json").read_text())
    lo = float(payload["bootstrap_ci_low"])
    hi = float(payload["bootstrap_ci_high"])
    assert np.isfinite(lo), f"bootstrap_ci_low not finite: {lo}"
    assert np.isfinite(hi), f"bootstrap_ci_high not finite: {hi}"
    assert hi > lo, f"CI half-width must be positive; got [{lo}, {hi}]"


def test_phase4_safeguards_present():
    required = [
        "placebo.json",
        "sensitivity_window.json",
        "sensitivity_partner_gdp.json",
        "sensitivity_treatment_window.json",
        "leads_test.json",
    ]
    missing = [f for f in required if not (cfg.METRICS / f).exists()]
    assert not missing, (
        f"Phase 4 safeguards JSONs missing: {missing}"
    )
