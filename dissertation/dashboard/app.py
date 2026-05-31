"""Root landing page for the Kosovo external-trade dashboard.

Holds no flow-specific figures — project framing, the predictive/causal
distinction, dashboard-wide caveats, and a navigation guide. All data lives on
the numbered pages. No live training and no live API calls anywhere in the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from lib import theme  # noqa: E402

st.set_page_config(
    page_title="Kosovo external trade — analysis dashboard",
    layout="wide",
)
theme.inject()

st.title("Kosovo external trade — analysis dashboard")

theme.intro_card(
    "How to read this dashboard",
    "Work through four stages: the descriptive <b>Trade Landscape</b> (EDA), then "
    "<b>Feature engineering</b>, then <b>Prediction & modelling</b> (Causal, "
    "Predictive), then the <b>Simulator</b>. The Causal and Predictive pages — and "
    "the sectoral section of Trade Landscape — carry an Imports / Exports toggle. "
    "Use the sidebar to move through the sections.",
)

st.markdown(
    """
This dashboard accompanies a study of Kosovo's bilateral **imports and exports**
over 2010–2024 across a panel of 113 partners. The empirical case study is
**Serbia (XS)** and Kosovo's 100 % tariff on Serbian goods (Nov 2018 – Apr 2020).

**Three kinds of quantity, kept distinct (ŷ / β̂):**

- **Descriptive** facts — what the trade data shows (Trade Landscape).
- **Predictive** quantities (ŷ) — what a model forecasts and which inputs it leans
  on. SHAP values are *predictive contributions*, not effects.
- **Causal** parameter (β̂) — the difference-in-differences estimate of the
  tariff-attributable change, under identifying assumptions. "Effect" refers
  only to this; causal prose is hedged ("the point estimate indicates", "suggestive").

Scenario outputs on the Simulator page are **conditional predictions under
manipulated inputs** — model behaviour, not policy effects.
"""
)

st.sidebar.title("Sections")
st.sidebar.markdown(
    """
**EDA**
- **1 · Trade Landscape** — scale, structure, sectors, and the Serbia anatomy.

**Feature engineering**
- **2 · Feature Engineering** — the twelve model inputs and how they are built.

**Prediction & modelling**
- **3 · Causal** — PPML-DiD, safeguards, the import-vs-export asymmetry.
- **4 · Predictive** — XGBoost vs PPML, persistence, SHAP, ablation, Diebold–Mariano.

**Simulation**
- **5 · Simulator** — interactive what-if predictions + scenario cards.
"""
)
st.sidebar.caption(
    "All artefacts are pre-computed; the dashboard performs no live training and "
    "no live API calls."
)
