"""Causal analysis: the PPML difference-in-differences estimate of the
tariff-attributable change. The flow toggle scopes the headline and bootstrap
distribution; the import-vs-export asymmetry and the import-side safeguards are
always shown. "Effect" refers only to beta_DiD; the prose is hedged."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lib import theme, flow as flowmod  # noqa: E402
from lib.loaders import load_csv, load_json, load_npy, T, M  # noqa: E402

theme.inject()
st.title("Causal: PPML-DiD + safeguards")
theme.intro_card(
    "What this page shows",
    "The difference-in-differences estimate of the tariff-attributable change "
    "for the selected flow, the import-vs-export asymmetry, and the import-side "
    "safeguards. 'Effect' refers only to beta_DiD; the prose is hedged.",
)

flow = flowmod.flow_toggle()

if flow == "import":
    ci = load_json(M("bootstrap_ci.json"))
    beta = ci["beta_main"]
    draws = load_npy(M("bootstrap_draws.npy"))
else:
    ci = load_json(M("bootstrap_ci_export.json"))
    beta = ci["beta_export"]
    draws = load_npy(M("bootstrap_draws_export.npy"))

st.markdown("Does the tariff episode leave a measurable, Serbia-specific mark once partner and year effects are absorbed? The difference-in-differences coefficient answers that for the selected flow.")

st.markdown(f"## Headline DiD — {flowmod.FLOW_LABEL[flow]} (1,000-draw pairs bootstrap)")
theme.kpi_row([
    ("beta on serbia_x_post", f"{beta:+.4f}"),
    ("95 % CI (beta)", f"[{ci['bootstrap_ci_low']:+.4f}, {ci['bootstrap_ci_high']:+.4f}]"),
    ("exp(beta) - 1", f"{ci['exp_beta_minus_1_pct']:+.2f} %"),
    ("Bootstrap valid", f"{ci['n_boot_completed']} / 1,000", f"{ci['n_boot_failed']} failed"),
])
st.caption(
    "*The point estimate indicates* a contraction relative to the AL/MK/ME "
    "fixed-effects baseline. The bootstrap percentile interval is the canonical "
    "inference statement (G = 4 clusters, so the cluster-robust SE is not used "
    "for inference)."
)

st.markdown("With only four partner clusters, inference rests on the bootstrap distribution rather than the cluster-robust standard error.")

st.markdown("## Bootstrap distribution")
valid = draws[~np.isnan(draws)]
hist = px.histogram(valid, nbins=40, height=360, labels={"value": "beta draw"})
hist.add_vline(x=beta, line_color="black", annotation_text=f"beta = {beta:+.4f}")
hist.add_vline(x=ci["bootstrap_ci_low"], line_dash="dash", line_color="crimson")
hist.add_vline(x=ci["bootstrap_ci_high"], line_dash="dash", line_color="crimson")
hist.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(hist, use_container_width=True)

st.markdown("Placing the two flows side by side is the crux of the study: the shock is sharp on imports and essentially absent on exports.")

st.markdown("## Import-vs-export asymmetry")
summary = load_csv(T("tbl_ch5_export_spillover_summary.csv"))
forest = go.Figure()
for _, row in summary.iterrows():
    key = str(row["flow"]).lower()[:6]
    forest.add_trace(go.Scatter(
        x=[row["beta"]], y=[str(row["flow"]).title()], mode="markers",
        marker=dict(size=12, color=flowmod.FLOW_COLOR.get(key, "#555")),
        error_x=dict(type="data", symmetric=False,
                     array=[row["ci_95_high"] - row["beta"]],
                     arrayminus=[row["beta"] - row["ci_95_low"]]),
        showlegend=False))
forest.add_vline(x=0, line_color="black", line_width=0.8)
forest.update_layout(height=240, xaxis=dict(title="beta on serbia_x_post (log-link units)"),
                     margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(forest, use_container_width=True)
st.dataframe(summary, use_container_width=True, hide_index=True)
st.caption(
    "Import-side beta is approximately -4.09 (about -98 %) versus an export-side "
    "beta near zero; the two bootstrap CIs do not overlap. A descriptive "
    "diagnostic, not a formal test of coefficient equality."
)

st.markdown("The headline only stands if it survives scrutiny. These checks are run on the import outcome, where the effect is identified.")

st.markdown("## Safeguards — import-only (not refit on exports)")
st.caption("Parallel trends, the leads test, the event study and sensitivity "
           "refits apply to the import outcome only.")
trends = load_csv(T("tbl_ch5_parallel_trends.csv"))
trends_fig = px.line(trends.sort_values(["group", "year"]), x="year",
                     y="mean_log_imports", color="group", markers=True, height=380)
trends_fig.add_vline(x=2018, line_dash="dash", line_color="rgba(220,40,60,0.6)")
trends_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(trends_fig, use_container_width=True)

leads = load_json(M("leads_test.json"))
st.markdown(
    f"**Leads test (joint Wald, 2014-2017):** p = {leads['joint_p_value']:.4f}, "
    f"Wald = {leads['joint_wald_stat']:.2f}, df = {leads['df']}. With G = 4 "
    "clusters the joint statistic is inflated; the *direction* (Serbia drifts "
    "upward pre-tariff) is the substantive finding."
)

event_study = load_csv(T("tbl_ch5_event_study.csv")).copy()
event_study["year"] = event_study["year"].astype(int)
event_study = event_study.sort_values("year")
event_fig = go.Figure(go.Scatter(
    x=event_study["year"], y=event_study["beta"], mode="markers+lines",
    error_y=dict(type="data",
                 array=(event_study["ci_high_95"] - event_study["beta"]).values,
                 arrayminus=(event_study["beta"] - event_study["ci_low_95"]).values)))
event_fig.add_hline(y=0, line_color="black", line_width=0.8)
event_fig.add_vline(x=2018, line_dash="dash", line_color="rgba(220,40,60,0.6)")
event_fig.update_layout(height=420, xaxis=dict(title="year", dtick=2),
                        yaxis=dict(title="beta on Serbia x year_t"),
                        margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(event_fig, use_container_width=True)

st.markdown("**Safeguards summary**")
st.dataframe(load_csv(T("tbl_ch5_safeguards_summary.csv")),
             use_container_width=True, hide_index=True)
