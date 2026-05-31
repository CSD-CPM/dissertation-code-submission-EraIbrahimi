"""What-if simulator and policy-scenario cards. The what-if runs inference on a
saved booster (no training); the levers are model-responsive predictors (partner
GDP, last-year trade), not policy instruments. Outputs are conditional
predictions, never causal effects."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lib import theme, flow as flowmod  # noqa: E402
from lib.loaders import load_json, M  # noqa: E402

from src import scenarios as sc  # noqa: E402

theme.inject()
st.title("Simulator — conditional predictions under manipulated inputs")
theme.intro_card(
    "How to use this",
    "Pick a partner and a flow, then nudge the two predictors the model actually "
    "responds to - partner GDP and last-year trade - and watch the booster's "
    "prediction move. These are <b>model-responsive predictors, not policy "
    "levers</b>: outputs are conditional predictions, never causal effects.",
)
st.warning(
    "All outputs are CONDITIONAL PREDICTIONS under manipulated inputs - model "
    "behaviour, not causal estimates. The booster is non-monotonic, so the "
    "direction of a partner's response is the model's, not an assumed rule."
)

flow = flowmod.flow_toggle()
panel = sc.load_panel_for_flow(flow)
feasible = sc.base_feature_matrix(panel, 2024)
partners = sorted(feasible["iso2"].unique())
default_index = partners.index("XS") if "XS" in partners else 0

col1, col2, col3 = st.columns(3)
partner = col1.selectbox("Partner", partners, index=default_index)
gdp_pct = col2.slider("Partner GDP change (%)", -50, 50, 0, 5)
lag_pct = col3.slider("Last-year trade change (%)", -90, 100, 0, 10)

result = sc.whatif_scenario(flow=flow, base_year=2024, partner_iso2=partner,
                            partner_gdp_pct=float(gdp_pct),
                            lagged_trade_pct=float(lag_pct), panel=panel)
per_partner = result["per_partner"]
row = per_partner[per_partner["iso2"] == partner].iloc[0]
baseline_value = float(row["baseline_pred"])
scenario_value = float(row["scenario_pred"])
change_pct = (scenario_value - baseline_value) / baseline_value * 100 if baseline_value else float("nan")

bar = go.Figure(go.Bar(x=["baseline", "scenario"], y=[baseline_value, scenario_value],
                       marker_color=[flowmod.FLOW_COLOR[flow], "#2f9e44"]))
bar.update_layout(height=340, yaxis=dict(title="predicted trade (EUR k)"),
                  margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(bar, use_container_width=True)
st.markdown(
    f"**Readout:** with partner GDP {gdp_pct:+d}% and last-year trade {lag_pct:+d}%, "
    f"the model predicts {flowmod.FLOW_LABEL[flow].lower()} for "
    f"**{row['partner_name']}** of **EUR {scenario_value:,.0f} k** versus a baseline "
    f"of **EUR {baseline_value:,.0f} k** - a change of **{change_pct:+.1f}%**."
)
st.caption(f"*{result['caveat']}*")

with st.expander("All partners - top 10 by absolute change"):
    st.dataframe(result["summary_top10"], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("## Policy-toggle scenarios — and why the model barely reacts")
st.caption(
    "The frozen policy scenarios. The booster never splits on the policy dummies, "
    "so these typically move the prediction by about zero - an honest finding, not "
    "a bug: the model encodes the Serbia collapse through fundamentals (partner "
    "GDP, lagged trade), not the tariff flag."
)


def policy_card(name: str, title: str) -> None:
    payload = load_json(M(f"scenario_{name}.json"))
    per = pd.DataFrame(payload["per_partner"])
    deltas = per["delta_eur_thousands"].astype(float)
    nonzero = int((deltas.abs() > 1e-6).sum())
    st.markdown(f"### {title}")
    theme.kpi_row([
        ("feature changed", payload["feature_changed"]),
        ("non-zero change rows", f"{nonzero} / {payload['n_partners']}"),
        ("max |change| (EUR k)", f"{deltas.abs().max():,.1f}"),
    ])
    if nonzero == 0:
        st.info("The booster's prediction is unchanged by this toggle - it never "
                "split on this policy feature. The 2019 collapse is captured "
                "indirectly through partner GDP and lagged trade instead.")
    st.caption(f"*{payload['caveat']}*")


policy_card("serbia_tariff", "Scenario 1 — Re-imposed Serbia tariff (binary proxy)")
policy_card("kosovo_gdp_plus5", "Scenario 2 — Kosovo GDP +5 %")
policy_card("turkey_fta", "Scenario 3 — Hypothetical Turkey FTA (proxy)")
