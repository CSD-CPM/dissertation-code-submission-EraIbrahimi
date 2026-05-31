"""Trade-flow toggle and the fixed import/export colour convention."""
from __future__ import annotations

import streamlit as st

FLOWS = ("import", "export")
FLOW_LABEL = {"import": "Imports", "export": "Exports"}
FLOW_COLOR = {"import": "#1f6feb", "export": "#d9480f"}


def flow_toggle(key: str = "flow") -> str:
    """Horizontal radio returning 'import' or 'export'."""
    return st.radio(
        "Trade flow",
        FLOWS,
        format_func=lambda f: FLOW_LABEL[f],
        horizontal=True,
        key=key,
    )
