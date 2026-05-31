import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import scenarios as sc


def test_baseline_invariance_matches_predict_baseline():
    panel = sc.load_panel_for_flow("import")
    out = sc.whatif_scenario(flow="import", base_year=2024,
                             partner_gdp_pct=0.0, lagged_trade_pct=0.0,
                             panel=panel)
    per = out["per_partner"]
    assert np.allclose(per["scenario_pred"], per["baseline_pred"])
    base = sc.predict_baseline(panel, 2024).sort_values("iso2").reset_index(drop=True)
    got = per.sort_values("iso2").reset_index(drop=True)
    assert np.allclose(got["baseline_pred"], base["imports_predicted_baseline"])
    assert (per["delta_eur_thousands"] == 0).all()


def test_partner_gdp_transform_and_recomputation():
    panel = sc.load_panel_for_flow("import")
    model = sc._load_booster("import")["model"]
    sub = sc.base_feature_matrix(panel, 2024)
    assert (sub["iso2"] == "DE").any()
    X = sub[cfg.FEATURE_ORDER].astype(float).copy()
    X.loc[sub["iso2"].values == "DE", "ln_partner_gdp"] += np.log1p(0.10)
    expected = sc._predict_levels(model, X)[sub["iso2"].values == "DE"][0]
    out = sc.whatif_scenario(flow="import", base_year=2024,
                             partner_iso2="DE", partner_gdp_pct=10.0, panel=panel)
    row = out["per_partner"]
    row = row[row["iso2"] == "DE"].iloc[0]
    assert row["scenario_pred"] == pytest.approx(expected, rel=1e-9)


def test_lagged_trade_transform_exact():
    panel = sc.load_panel_for_flow("import")
    model = sc._load_booster("import")["model"]
    sub = sc.base_feature_matrix(panel, 2024)
    X = sub[cfg.FEATURE_ORDER].astype(float).copy()
    m = sub["iso2"].values == "DE"
    old = X.loc[m, sc.WHATIF_LAG_FEATURE].values
    X.loc[m, sc.WHATIF_LAG_FEATURE] = np.log1p(np.expm1(old) * (1.0 + 0.20))
    expected = sc._predict_levels(model, X)[m][0]
    out = sc.whatif_scenario(flow="import", base_year=2024,
                             partner_iso2="DE", lagged_trade_pct=20.0, panel=panel)
    row = out["per_partner"]
    row = row[row["iso2"] == "DE"].iloc[0]
    assert row["scenario_pred"] == pytest.approx(expected, rel=1e-9)


def test_responsiveness_no_sign_assertion():
    out = sc.whatif_scenario(flow="import", base_year=2024, partner_gdp_pct=5.0)
    deltas = out["per_partner"]["delta_eur_thousands"].abs()
    assert (deltas > 1e-9).sum() >= 1


def test_export_flow_uses_export_target_and_booster():
    out = sc.whatif_scenario(flow="export", base_year=2024,
                             partner_iso2="DE", lagged_trade_pct=10.0)
    per = out["per_partner"]
    assert (per["flow"] == "export").all()
    panel = sc.load_panel_for_flow("export")
    de_exp = panel[(panel["iso2"] == "DE") & (panel["year"] == 2024)]["exports_eur_thousands"].iloc[0]
    assert per[per["iso2"] == "DE"]["actual_trade"].iloc[0] == pytest.approx(de_exp)


def test_invalid_pct_raises():
    with pytest.raises(ValueError):
        sc.whatif_scenario(flow="import", lagged_trade_pct=-100.0)
    with pytest.raises(ValueError):
        sc.whatif_scenario(flow="import", partner_gdp_pct=-100.0)
    with pytest.raises(ValueError):
        sc.whatif_scenario(flow="nope")
