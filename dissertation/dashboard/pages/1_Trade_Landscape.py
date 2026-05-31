"""Trade Landscape: the full descriptive picture of Kosovo's external trade,
2010-2024 — scale and balance, who Kosovo trades with, sparsity and
concentration, sectoral composition, the Serbia tariff anatomy, and
data-reliability diagnostics."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from lib import theme, flow as flowmod  # noqa: E402
from lib.loaders import load_csv, load_json, load_parquet, T, M, F, P  # noqa: E402
from src import config as cfg  # noqa: E402

theme.inject()
st.title("Trade Landscape")

theme.intro_card(
    "What this page shows",
    "The full descriptive picture of Kosovo's external trade, 2010-2024: scale "
    "and balance, who Kosovo trades with, how concentrated and sparse it is, what "
    "it trades, and the Serbia tariff anatomy. Descriptive evidence only; the "
    "feature engineering and the causal and predictive analysis follow on the "
    "later pages.",
)

# ---------------------------------------------------------------------------
# Scale & balance
# ---------------------------------------------------------------------------
annual = load_csv(T("tbl_ch3_annual_totals.csv"))
row_2024 = annual[annual["year"] == 2024].iloc[0]
row_2010 = annual[annual["year"] == 2010].iloc[0]

st.markdown(
    "Kosovo is a small, highly open economy whose external trade has expanded "
    "rapidly but stayed structurally imbalanced. The 2024 figures set the scale."
)

st.markdown("## Trade scale and balance, 2024")
theme.kpi_row([
    ("Imports", f"EUR {row_2024['imports'] / 1000:,.0f} M", "Total goods imports, 2024"),
    ("Exports", f"EUR {row_2024['exports'] / 1000:,.0f} M", "Total goods exports, 2024"),
    ("Trade deficit", f"EUR {row_2024['deficit'] / 1000:,.0f} M", "Imports minus exports, 2024"),
    ("Exports / imports", f"{row_2024['export_to_import_ratio']:.2f}", "Export-to-import coverage ratio"),
])

st.markdown(
    "Tracing the two flows across the full panel shows steady growth on both "
    "sides and a gap between them that widens throughout."
)

st.markdown("## Imports, exports and the trade deficit")
fig = go.Figure()
fig.add_trace(go.Bar(x=annual["year"], y=annual["deficit"], name="Deficit",
                     marker_color="#c0ccd9"))
fig.add_trace(go.Scatter(x=annual["year"], y=annual["imports"], name="Imports",
                         line=dict(color=flowmod.FLOW_COLOR["import"], width=3)))
fig.add_trace(go.Scatter(x=annual["year"], y=annual["exports"], name="Exports",
                         line=dict(color=flowmod.FLOW_COLOR["export"], width=3)))
fig.update_layout(height=420, yaxis=dict(title="EUR thousands"),
                  hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

imports_growth = row_2024["imports"] / row_2010["imports"]
exports_growth = row_2024["exports"] / row_2010["exports"]
st.markdown(
    f"Imports grew about {imports_growth:.1f}x and exports about {exports_growth:.1f}x "
    f"between 2010 and 2024, while the trade deficit widened from "
    f"EUR {row_2010['deficit'] / 1000:,.0f} M to EUR {row_2024['deficit'] / 1000:,.0f} M. "
    "The export-to-import ratio stayed within a narrow 11-18 % band throughout, so "
    "export capacity scaled roughly in step with imports without closing the "
    "structural gap. Kosovo's export base is more concentrated and sparser than its "
    "import base - detail on the Trade Landscape page."
)

# ---------------------------------------------------------------------------
# Partners
# ---------------------------------------------------------------------------
st.markdown(
    "That growth is spread unevenly across partners: trade is concentrated among "
    "a handful of economies, and the partner mix differs sharply between the two flows."
)

st.markdown("## Who Kosovo trades with, 2024")
top = load_csv(T("tbl_ch3_top_partners_2024.csv"))
left, right = st.columns(2)
top_imports = top.sort_values("imports", ascending=False).head(8)
fig_imports = px.bar(top_imports.iloc[::-1], x="imports", y="partner_name",
                     orientation="h", height=360,
                     labels={"imports": "2024 imports (EUR k)", "partner_name": ""})
fig_imports.update_traces(marker_color=flowmod.FLOW_COLOR["import"])
fig_imports.update_layout(margin=dict(l=10, r=10, t=34, b=10), title="Top import partners")
left.plotly_chart(fig_imports, use_container_width=True)
top_exports = top.sort_values("exports", ascending=False).head(8)
fig_exports = px.bar(top_exports.iloc[::-1], x="exports", y="partner_name",
                     orientation="h", height=360,
                     labels={"exports": "2024 exports (EUR k)", "partner_name": ""})
fig_exports.update_traces(marker_color=flowmod.FLOW_COLOR["export"])
fig_exports.update_layout(margin=dict(l=10, r=10, t=34, b=10), title="Top export partners")
right.plotly_chart(fig_exports, use_container_width=True)

# ---------------------------------------------------------------------------
# Structure — sparsity, concentration, asymmetry
# ---------------------------------------------------------------------------
st.markdown(
    "How dense is Kosovo's bilateral trade matrix? The export side is markedly "
    "sparser than the import side — in any given year roughly half of potential "
    "export relationships are inactive, against under a third on the import side."
)

st.markdown("## Zero counts and sparsity")
zero_counts = load_csv(T("tbl_ch3_zero_counts.csv"))
fig_zero = px.line(zero_counts.sort_values("year"), x="year", y="zero_rate_pct",
                   color="flow", markers=True,
                   labels={"zero_rate_pct": "zero rate (%)"}, height=340)
fig_zero.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_zero, use_container_width=True)

st.markdown(
    "How concentrated is that trade across partners? Both flows lean heavily on a "
    "few partners, but the export side is the more concentrated of the two on every measure."
)

st.markdown("## Partner concentration (HHI and top-share)")
st.dataframe(load_csv(T("tbl_ch3_concentration.csv")),
             use_container_width=True, hide_index=True)

st.markdown(
    "The partner mix is also markedly asymmetric across flows: most of Kosovo's "
    "partners are import-dominant or import-only, and only a small minority are "
    "export-dominant."
)

st.markdown("## Partner asymmetry")
asymmetry = load_csv(T("tbl_ch3_partner_asymmetry.csv"))
counts = (asymmetry["classification"].value_counts()
          .rename_axis("classification").reset_index(name="n_partners"))
fig_asym = px.bar(counts, x="n_partners", y="classification", orientation="h",
                  height=300)
fig_asym.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_asym, use_container_width=True)
asym_fig = F("fig_ch3_partner_asymmetry.png")
if os.path.exists(asym_fig):
    st.image(asym_fig,
             caption="Cumulative imports vs exports per partner (log-log, parity line).")

# ---------------------------------------------------------------------------
# Sectoral composition
# ---------------------------------------------------------------------------
st.markdown("## Sectoral composition")
flow = flowmod.flow_toggle()

sector = load_parquet(P("panel_sector.parquet"))
long = sector.melt(id_vars=["year", "hs_section"],
                   value_vars=["imports_eur_thousands", "exports_eur_thousands"],
                   var_name="flow_col", value_name="value")
long["flow"] = long["flow_col"].map({"imports_eur_thousands": "import",
                                      "exports_eur_thousands": "export"})
selected = long[long["flow"] == flow]

st.markdown(
    "What does Kosovo trade, and how has the composition shifted over time? The "
    "section-by-year matrix shows which HS sections dominate the selected flow "
    "and how their shares move."
)

st.markdown(f"## HS-section {flowmod.FLOW_LABEL[flow].lower()} over time")
pivot = selected.pivot(index="hs_section", columns="year", values="value").fillna(0)
heatmap = px.imshow(pivot.values, x=pivot.columns.astype(int), y=pivot.index,
                    aspect="auto", color_continuous_scale="Viridis",
                    labels=dict(x="year", y="HS section", color="EUR k"), height=620)
heatmap.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(heatmap, use_container_width=True)

st.markdown(
    "A handful of sections account for most of the flow; the leaders are stable "
    "across the panel even as their relative shares rotate."
)

st.markdown("## Top-5 HS sections by 2024 volume")
composition = load_csv(T("tbl_ch3_sector_composition.csv"))
comp_flow = composition[composition["flow"].astype(str).str.lower().str.startswith(flow)]
latest = comp_flow[comp_flow["year"] == comp_flow["year"].max()].sort_values("rank").head(5)
st.dataframe(latest[["rank", "hs_section", "value_eur_thousands", "share_of_flow_pct"]],
             use_container_width=True, hide_index=True)

st.markdown(
    "Aligning year-on-year share shifts with known external events surfaces where "
    "the composition moved most. These are descriptive alignments, not causal effects."
)

st.markdown("## Sector event response (largest pp share shifts)")
events = load_csv(T("tbl_ch3_sector_event_response_ppshift.csv"))
events_flow = events[events["flow"].astype(str).str.lower().str.startswith(flow)].copy()
events_flow = events_flow.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False)
st.dataframe(events_flow[["hs_section", "event_label", "delta_pp", "z_score"]],
             use_container_width=True, hide_index=True)
event_fig = F("fig_ch3_sector_event_response.png")
if os.path.exists(event_fig):
    st.image(event_fig, caption="Top-10 most-reactive (section, flow, event) triples.")

st.markdown("## Gap Institute (2019) — published sectoral diversion")
gap_path = cfg.DATA_RAW / "gap_institute_diversion.csv"
if gap_path.exists():
    st.dataframe(pd.read_csv(gap_path), use_container_width=True, hide_index=True)
    st.caption("Source: Gap Institute (2019). External evidence, not derived here.")
else:
    st.info("Gap Institute seed CSV is not present in this working tree.")

st.markdown("## Bilateral x HS2 charts")
eurostat_path = cfg.DATA_RAW / "eurostat_comext_hs2.csv"
if eurostat_path.exists():
    st.dataframe(pd.read_csv(eurostat_path).head(20), use_container_width=True)
else:
    st.info("Eurostat COMEXT HS2 was not acquired within scope; aggregate ASK "
            "sector data and Gap Institute evidence are used instead.")

# ---------------------------------------------------------------------------
# Serbia anatomy
# ---------------------------------------------------------------------------
st.markdown(
    "Against this backdrop, one relationship is the single most distinct feature "
    "of the panel. Imports from Serbia collapse around the 2018 tariff, and the "
    "monthly data place the onset and the recovery precisely on the policy dates."
)

st.markdown("## Serbia: annual and monthly")
serbia = load_csv(T("tbl_ch3_serbia_trajectory.csv"))
serbia = serbia[serbia["iso2"] == "XS"]
fig_serbia = go.Figure()
for flow_s, color in (("import", "#1f6feb"), ("export", "#d9480f")):
    series = (serbia[serbia["flow"].astype(str).str.lower().str.startswith(flow_s)]
              .sort_values("year"))
    fig_serbia.add_trace(go.Scatter(
        x=series["year"], y=series["value_eur_thousands"], mode="lines+markers",
        name=f"Serbia {flow_s}s", line=dict(width=3, color=color)))
fig_serbia.add_vrect(x0=2018.83, x1=2020.33, fillcolor="rgba(220,40,60,0.08)",
                     line=dict(color="rgba(220,40,60,0.5)", width=1, dash="dash"))
fig_serbia.update_layout(height=380,
                         yaxis=dict(title="EUR thousands", rangemode="tozero"),
                         hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_serbia, use_container_width=True)

st.markdown("**Monthly breakpoint (import shock onset and recovery):**")
st.dataframe(load_csv(T("tbl_ch3_serbia_monthly_breakpoint.csv")),
             use_container_width=True, hide_index=True)
monthly_fig = F("fig_ch3_serbia_monthly_breakpoint.png")
if os.path.exists(monthly_fig):
    st.image(monthly_fig,
             caption="Monthly Serbia imports; onset Nov 2018, recovery Apr 2020.")

# ---------------------------------------------------------------------------
# Asymmetry preview (DiD headline)
# ---------------------------------------------------------------------------
st.markdown(
    "The descriptive collapse above is formalised on the Causal page. The "
    "difference-in-differences estimates preview the asymmetry:"
)
comparison = load_json(M("import_vs_export_did_comparison.json"))
theme.kpi_row([
    ("β_DiD imports", f"{comparison['beta_import']:+.3f}",
     f"exp(β)-1 = {comparison['import_exp_minus_1_pct']:+.1f} %"),
    ("β_DiD exports", f"{comparison['beta_export']:+.3f}",
     f"exp(β)-1 = {comparison['export_exp_minus_1_pct']:+.1f} %"),
])
st.caption(
    "*The point estimate indicates* a deep contraction in imports from Serbia and "
    "essentially no matching move on exports. The estimate and its safeguards are "
    "on the Causal page."
)

# ---------------------------------------------------------------------------
# Export modellability
# ---------------------------------------------------------------------------
st.markdown(
    "What does this descriptive picture imply for modelling the export side? The "
    "higher zero rate and variance raise the error floor, but a modellable subset "
    "of partners clearly exists."
)

st.markdown("## Export modellability")
modellability = load_json(M("export_modellability.json"))
theme.kpi_row([
    ("Export zero rate", f"{modellability['export_zero_rate'] * 100:.1f} %"),
    ("Import zero rate", f"{modellability['import_zero_rate'] * 100:.1f} %"),
    ("Var ratio (log exp/imp)", f"{modellability['var_log_ratio_exp_to_imp']:.2f}x"),
    ("Partners >5 nonzero export yrs", f"{modellability['partners_with_gt5_nonzero_export_years']}"),
])

# ---------------------------------------------------------------------------
# Reconciliation diagnostics
# ---------------------------------------------------------------------------
with st.expander("Data reliability — reconciliation diagnostics"):
    st.markdown("**Partner vs HS-section reconciliation**")
    st.dataframe(load_csv(T("tbl_ch3_partner_vs_sections_reconciliation.csv")),
                 use_container_width=True, hide_index=True)
    st.markdown("**Monthly vs yearly reconciliation**")
    st.dataframe(load_csv(T("tbl_ch3_monthly_vs_yearly_reconciliation.csv")),
                 use_container_width=True, hide_index=True)
