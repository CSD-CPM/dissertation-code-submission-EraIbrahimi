"""Phase 3 (interpretation pillar) artefact integrity."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as cfg


def test_phase3_shap_global():
    df = pd.read_csv(cfg.TABLES / "tbl_ch4_shap_global.csv")
    assert len(df) == 12, (
        f"SHAP global ranking: expected 12 rows (one per feature), "
        f"got {len(df)}."
    )
    # Sorted descending by mean_abs_shap?
    msa = df["mean_abs_shap"].astype(float).values
    assert (np.diff(msa) <= 0).all(), (
        "tbl_ch4_shap_global.csv must be sorted by mean_abs_shap "
        "DESCENDING."
    )
    # mean_shap_sign must be one of {positive, negative, zero}
    assert set(df["mean_shap_sign"].unique()) <= {"positive", "negative", "zero"}


def test_phase3_serbia_shap_long_168_rows():
    df = pd.read_csv(cfg.TABLES / "tbl_ch4_serbia_shap.csv")
    # 14 Serbia rows (years 2011-2024) × 12 features = 168 rows
    assert len(df) == 168, (
        f"tbl_ch4_serbia_shap.csv: expected 168 rows (14 years × 12 "
        f"features); got {len(df)}."
    )
    n_years = df["year"].nunique()
    n_feats = df["feature"].nunique()
    assert n_years == 14, f"expected 14 distinct years; got {n_years}"
    assert n_feats == 12, f"expected 12 distinct features; got {n_feats}"


def test_phase3_ablation_layers():
    df = pd.read_csv(cfg.TABLES / "tbl_ch4_ablation_cv.csv")
    # 4 layers × 5 folds = 20 fold-level rows + 4 per-layer mean rows = 24
    assert len(df) == 24, (
        f"tbl_ch4_ablation_cv.csv: expected 24 rows (4 layers × 5 "
        f"folds + 4 mean rows); got {len(df)}."
    )
    layers = sorted(df["layer"].unique())
    assert layers == ["L1_structural", "L2_policy", "L3_macro", "L4_lagged"], (
        f"Unexpected ablation layer set: {layers}"
    )
    # Per-layer feature counts: L1=3, L2=7, L3=10, L4=12
    expected_counts = {
        "L1_structural": 3, "L2_policy": 7,
        "L3_macro": 10, "L4_lagged": 12,
    }
    for layer, expected in expected_counts.items():
        got = int(df.loc[df["layer"] == layer, "n_features"].iloc[0])
        assert got == expected, (
            f"Ablation layer {layer}: expected {expected} features, "
            f"got {got}."
        )


def test_phase3_dm_test():
    payload = json.loads((cfg.METRICS / "dm_test.json").read_text())
    assert payload["n"] == 558, (
        f"DM test: expected n=558 (pooled CV errors), got {payload['n']}."
    )
    for k in ("dm", "dm_hln", "p_value", "df", "mean_loss_diff"):
        assert k in payload, f"dm_test.json missing key: {k!r}"
    p = float(payload["p_value"])
    assert 0.0 <= p <= 1.0, f"DM p-value out of [0,1]: {p}"
    dm_hln = float(payload["dm_hln"])
    assert np.isfinite(dm_hln), f"DM_HLN must be finite; got {dm_hln}."
