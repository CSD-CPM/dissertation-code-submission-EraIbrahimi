"""Predictive analysis: XGBoost versus the PPML baseline and a persistence floor
for the selected flow, the SHAP ranking, the import-side ablation, and the
import-vs-export comparison. SHAP values are predictive contributions, not
effects."""
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
from lib.loaders import load_csv, load_json, T, M, F  # noqa: E402

theme.inject()
st.title("Predictive: XGBoost vs PPML + SHAP + ablation")
theme.intro_card(
    "What this page shows",
    "Out-of-sample accuracy of XGBoost against the PPML baseline and a "
    "persistence floor for the selected flow, what the booster leans on (SHAP), "
    "and the import-side ablation. SHAP values are predictive contributions, "
    "not effects.",
)

flow = flowmod.flow_toggle()
is_export = flow == "export"

st.markdown("Can a flexible learner forecast bilateral trade better than the econometric baseline? Expanding-window cross-validation pits XGBoost against PPML on identical rows for the selected flow.")

st.markdown(f"## CV head-to-head — {flowmod.FLOW_LABEL[flow]} (expanding-window folds)")
cv = load_csv(M("cv_xgb_vs_ppml_export.csv" if is_export else "cv_xgb_vs_ppml.csv"))
fold_rows = cv[pd.to_numeric(cv["test_year"], errors="coerce").notna()]
mean_row = {
    "last_train_year": "—", "test_year": "mean",
    "ppml_rmse_eur_thousands": fold_rows["ppml_rmse_eur_thousands"].mean(),
    "ppml_r2": fold_rows["ppml_r2"].mean(),
    "xgb_rmse_eur_thousands": fold_rows["xgb_rmse_eur_thousands"].mean(),
    "xgb_r2": fold_rows["xgb_r2"].mean(), "winner": "—",
}
st.dataframe(pd.concat([fold_rows, pd.DataFrame([mean_row])], ignore_index=True),
             use_container_width=True, hide_index=True)

st.markdown("The 2023-2024 holdout was untouched during tuning, so it is the cleaner test of out-of-sample accuracy.")

st.markdown("## Holdout 2023 + 2024")
holdout = load_json(M("holdout_export_2023_2024.json" if is_export else "holdout_2023_2024.json"))
rows = []
combined = holdout["xgboost"]["combined"]
rows.append({"model": "XGBoost", "RMSE": combined["rmse"], "R2": combined["r2"], "n": int(combined["n"])})
ppml_combined = holdout.get("ppml_predictive_combined", {})
if ppml_combined:
    rows.append({"model": "PPML", "RMSE": ppml_combined["rmse"], "R2": ppml_combined["r2"],
                 "n": int(ppml_combined["n"])})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("Both models still have to beat a far simpler rule: last year's trade carried forward.")

st.markdown("## Persistence baseline (lag-only floor)")
if is_export:
    persistence = load_json(M("persistence_baseline_export.json"))
    theme.kpi_row([
        ("CV pool R2", f"{persistence['cv']['r2']:.3f}"),
        ("CV pool RMSE", f"{persistence['cv']['rmse_eur_thousands']:,.0f}"),
        ("Holdout R2", f"{persistence['holdout_combined']['r2']:.3f}"),
        ("Holdout RMSE", f"{persistence['holdout_combined']['rmse_eur_thousands']:,.0f}"),
    ])
else:
    comparison = load_csv(T("tbl_ch4_import_vs_export_prediction.csv"))
    persistence = comparison[comparison["section"] == "persistence"].set_index("metric")["import_value"]
    theme.kpi_row([
        ("CV pool R2", f"{float(persistence['CV pool R2']):.3f}"),
        ("CV pool RMSE", f"{float(persistence['CV pool RMSE']):,.0f}"),
        ("Holdout R2", f"{float(persistence['Holdout combined R2']):.3f}"),
        ("Holdout RMSE", f"{float(persistence['Holdout combined RMSE']):,.0f}"),
    ])
st.caption("Lagged trade is load-bearing on both flows; the marginal gain of ML "
           "over this lag-only floor is limited (see the comparison below).")

st.markdown("Is the gap between the two models' errors statistically meaningful, or just noise?")

st.markdown("## Diebold-Mariano (HLN, pooled CV errors)")
dm = load_json(M("dm_test_export.json" if is_export else "dm_test.json"))
theme.kpi_row([
    ("n", f"{dm['n']}"),
    ("DM_HLN", f"{dm['dm_hln']:+.4f}"),
    ("p-value", f"{dm['p_value']:.4f}"),
])
st.markdown(f"**At alpha = 0.05:** {dm['interpretation']}")

st.markdown("What does the booster actually lean on? SHAP decomposes its predictions into per-feature contributions, which are predictive associations, not effects.")

st.markdown("## SHAP global ranking — predictive contributions")
shap = load_csv(T("tbl_ch4_shap_global_export.csv" if is_export else "tbl_ch4_shap_global.csv"))
shap = shap.sort_values("mean_abs_shap", ascending=True)
shap_fig = px.bar(shap, x="mean_abs_shap", y="feature", orientation="h", height=520,
                  labels={"mean_abs_shap": "mean |SHAP| (log1p(EUR k) units)"})
shap_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(shap_fig, use_container_width=True)
st.caption("SHAP values are predictive contributions in the booster's log1p "
           "output space, not effects. Lagged trade dominates on both flows.")

st.markdown("Because the two targets sit on very different scales, the fair cross-flow comparison uses normalised error.")

st.markdown("## Import-vs-export comparison (normalised metrics)")
comparison_table = load_csv(T("tbl_ch4_import_vs_export_prediction.csv")).fillna("")
st.dataframe(comparison_table, use_container_width=True, hide_index=True)
st.caption("Level RMSE is not comparable across flows (import levels are roughly "
           "7x export); the normalised rows (WAPE, RMSE/mean) are the fair comparison.")

st.markdown("Adding feature blocks one layer at a time shows where the predictive lift comes from.")

st.markdown("## 4-layer ablation — import-only (not refit on exports)")
ablation_cv = load_csv(T("tbl_ch4_ablation_cv.csv"))
ablation_holdout = load_csv(T("tbl_ch4_ablation_holdout.csv")).rename(
    columns={"rmse": "holdout_rmse", "r2": "holdout_r2"})
means = ablation_cv[ablation_cv["fold_idx"] == "mean"][["layer", "n_features", "rmse", "r2"]].rename(
    columns={"rmse": "cv_rmse", "r2": "cv_r2"})
merged = means.merge(ablation_holdout[["layer", "holdout_rmse", "holdout_r2"]], on="layer", how="left")
st.dataframe(merged, use_container_width=True, hide_index=True)
ablation_fig = go.Figure()
ablation_fig.add_trace(go.Bar(name="CV mean RMSE", x=merged["layer"], y=merged["cv_rmse"]))
ablation_fig.add_trace(go.Bar(name="Holdout RMSE", x=merged["layer"], y=merged["holdout_rmse"]))
ablation_fig.update_layout(barmode="group", height=400, yaxis=dict(title="RMSE (EUR k)"),
                           margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(ablation_fig, use_container_width=True)

serbia_shap = F("fig_ch4_serbia_shap_heatmap.png")
if os.path.exists(serbia_shap):
    st.markdown("**Serbia-only SHAP (import model):** ln_partner_gdp carries the "
                "2019 contribution; the serbia_tariff dummy is flat.")
    st.image(serbia_shap)
