"""Light professional theme: CSS injection plus reusable card/KPI components."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
h1, h2, h3, h4 { font-family: Georgia, 'Times New Roman', serif; letter-spacing: .2px; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }
.intro-card {
  background: #f5f7fa; border: 1px solid #e3e8ef; border-left: 4px solid #1f6feb;
  border-radius: 8px; padding: 14px 18px; margin: 6px 0 18px 0;
}
.intro-card h4 { margin: 0 0 6px 0; font-size: 1.02rem; }
.intro-card p { margin: 0; color: #36506b; font-size: 0.93rem; line-height: 1.5; }
div[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e3e8ef; border-radius: 8px; padding: 10px 14px;
}
div[data-testid="stMetricValue"] {
  font-size: 1.5rem; line-height: 1.2; white-space: normal; overflow-wrap: anywhere;
}
div[data-testid="stMetricLabel"] { white-space: normal; }
</style>
"""


def inject() -> None:
    """Inject the dashboard stylesheet. Call once per page after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def intro_card(title: str, body: str) -> None:
    """Render a 'what this page shows / how to read it' card. body may contain HTML."""
    st.markdown(
        f'<div class="intro-card"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    """Render a row of KPI cards. Each item is (label, value) or (label, value, help)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else None
        col.metric(label, value, help=help_text)
