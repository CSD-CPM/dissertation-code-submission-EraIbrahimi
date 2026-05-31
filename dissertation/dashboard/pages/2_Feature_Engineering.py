"""Feature engineering: how the merged raw panel becomes the twelve model inputs.
A descriptive reference between the EDA pages and the modelling pages — it defines
every feature the Predictive page's SHAP ranking and ablation refer to."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from lib import theme  # noqa: E402
from lib.loaders import load_parquet, P  # noqa: E402

from src import config as cfg  # noqa: E402

theme.inject()
st.title("Feature engineering")
theme.intro_card(
    "What this page shows",
    "How the merged raw panel — ASK bilateral trade, World Bank WDI macro, and "
    "CEPII gravity — becomes the twelve inputs the models use. It sits between the "
    "descriptive pages and the modelling pages and defines every feature the "
    "Predictive page's SHAP ranking and ablation refer to.",
)

FEATURE_INFO = {
    "ln_partner_gdp": ("Macro", "Log of the partner's GDP (current USD, World Bank WDI)."),
    "ln_kosovo_gdp": ("Macro", "Log of Kosovo's GDP (current USD, World Bank WDI)."),
    "ln_distance": ("Gravity", "Log capital-to-capital distance (CEPII GeoDist; Haversine fallback for Kosovo, which is absent from CEPII)."),
    "contiguity": ("Gravity", "1 if the partner shares a land border with Kosovo."),
    "common_language": ("Gravity", "1 if the partner shares a common official language."),
    "serbia_tariff": ("Policy", "1 for Serbia in the tariff-window years 2018-2020."),
    "saa_in_force": ("Policy", "1 from 2016 onward for EU partners (Stabilisation and Association Agreement); partner-conditional."),
    "covid": ("Policy / event", "1 in 2020."),
    "cefta_member": ("Policy", "1 if the partner is a CEFTA member (AL, MK, ME, BA, XS, MD)."),
    "lagged_imports_log1p": ("Lagged", "log1p of the partner's previous-year trade; NaN on the first panel year (no prior history)."),
    "year_trend": ("Time", "Calendar year minus 2010."),
    "partner_import_share_lag": ("Lagged", "The partner's share of total trade in the previous year."),
}

st.markdown("The engineered panel is balanced — every partner observed in every "
            "year — and feeds both the predictive and the causal models.")

st.markdown("## The modelling panel")
panel = load_parquet(P("panel_bilateral.parquet"))
theme.kpi_row([
    ("Partners", f"{panel['iso2'].nunique()}"),
    ("Years", f"{panel['year'].nunique()}  ({int(panel['year'].min())}-{int(panel['year'].max())})"),
    ("Rows", f"{len(panel):,}"),
    ("Engineered features", f"{len(cfg.FEATURE_ORDER)}"),
])
st.markdown("Raw inputs are merged from three sources — ASK bilateral trade, World "
            "Bank WDI macro indicators, and CEPII gravity (distance, contiguity, "
            "language) — then transformed into the features below. Missing external "
            "values are flagged, never silently imputed.")

st.markdown("## The twelve features")
features = pd.DataFrame(
    [{"feature": f, "family": FEATURE_INFO[f][0], "definition": FEATURE_INFO[f][1]}
     for f in cfg.FEATURE_ORDER]
)
st.dataframe(features, use_container_width=True, hide_index=True)
st.caption("The target is modelled as log1p(trade) and back-transformed via expm1 "
           "for level-scale scoring. On the export side the two lag features become "
           "lagged_exports_log1p and partner_export_share_lag; the other ten are "
           "flow-agnostic.")

st.markdown("## Ablation layers")
st.markdown("The Predictive page strips the feature set to cumulative blocks and "
            "adds them back one layer at a time, isolating where predictive lift "
            "comes from:")
ablation = pd.DataFrame(
    [{"layer": layer, "n_features": len(feats), "features": ", ".join(feats)}
     for layer, feats in cfg.ABLATION_LAYERS.items()]
)
st.dataframe(ablation, use_container_width=True, hide_index=True)
