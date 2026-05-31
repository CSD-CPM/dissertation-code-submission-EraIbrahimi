"""Phase 1 panel integrity invariants.

Reads the committed parquet only — no model retraining.
"""
from __future__ import annotations

import pandas as pd

from src import config as cfg


def _load_panel() -> pd.DataFrame:
    return pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")


def test_phase1_panel_size():
    panel = _load_panel()
    assert len(panel) == 1695, (
        f"Phase 1 panel size invariant: expected 1,695 rows "
        f"(113 partners × 15 years), got {len(panel)}."
    )


def test_phase1_zeros_preserved():
    panel = _load_panel()
    n_zero = int((panel["imports_eur_thousands"] == 0).sum())
    # Zero rows should be ≈ 35 (≈ 2 %); allow ±5 for re-run drift.
    assert 30 <= n_zero <= 40, (
        f"Phase 1 zero rows: expected ~35 (range [30, 40] for re-run drift), "
        f"got {n_zero}."
    )


def test_phase1_panel_balance():
    panel = _load_panel()
    n_partners = panel["iso2"].nunique()
    n_years = panel["year"].nunique()
    assert n_partners == 113, (
        f"Phase 1 partner count: expected 113, got {n_partners}."
    )
    assert n_years == 15, (
        f"Phase 1 year count: expected 15 (2010-2024), got {n_years}."
    )
    # Strict balance: every (iso2, year) pair must be present exactly once.
    pairs = panel.groupby(["iso2", "year"]).size()
    assert (pairs == 1).all(), (
        "Phase 1 panel is not balanced — some (iso2, year) pairs are "
        f"duplicated or missing. Pair-count distribution: {pairs.value_counts().to_dict()}"
    )


def test_phase1_feature_count():
    panel = _load_panel()
    missing = [f for f in cfg.FEATURE_ORDER if f not in panel.columns]
    assert not missing, (
        f"Phase 1 panel missing locked features: {missing}. "
        f"Expected all 12 in cfg.FEATURE_ORDER: {cfg.FEATURE_ORDER}."
    )
