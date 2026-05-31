"""Phase 0 EDA — read-only on Phase 1–5 modules.

Parses the May 2026 ASK source files for BOTH imports and exports,
produces hard + diagnostic reconciliations, descriptive tables, figures,
and a metric JSON for the dissertation's data chapter (Chapter 3).

This module is intentionally self-contained: helpers (`_to_csv_and_tex`,
`_matplotlib_setup`, `_save_fig`) are duplicated here rather than imported
from Phase 1–5 modules. A future cleanup phase will consolidate this surface.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASK_DIR = cfg.ROOT.parent / "data" / "ask"

PARTNER_FILE  = ASK_DIR / "ask_yearly_partner_trade_2010_2024.xlsx"
SECTIONS_FILE = ASK_DIR / "ask_yearly_hs_sections_trade_2010_2024.xlsx"
CHAPTER_FILE  = ASK_DIR / "ask_yearly_chapter_trade_2010_2024.xlsx"
BEC_FILE      = ASK_DIR / "ask_yearly_bec_trade_2010_2024.xlsx"
MONTHLY_IMP   = ASK_DIR / "ask_monthly_import_partner_2010M01_2026M03.xlsx"
MONTHLY_EXP   = ASK_DIR / "ask_monthly_export_partner_2010M01_2026M03.xlsx"

PARTNER_CODE_REGEX = re.compile(r"^([A-Z]{2}):(.*)$")  # allow empty name (ZZ residual)
CHAPTER_CODE_REGEX = re.compile(r"^(\d{2}):(.+)$")
SECTION_CODE_REGEX = re.compile(r"^(\d{2})\s+(.+)$")
BEC_CODE_REGEX     = re.compile(r"^(\d+):(.+)$")
MONTH_LABEL_REGEX  = re.compile(r"^(\d{4})M(\d{2})$")

# ZZ = unspecified-destination residual (not a partner; closes partner↔sections
# reconciliation). Keep in partner_long for reconciliation; analyzers strip it.
RESIDUAL_ISO2 = "ZZ"
# Other no-name aggregate codes in the file have zero values in May 2026:
# XX (world total), XY, XZ, UE (European Union), YU (former Yugoslavia).
# Parsed but harmless; analyzers strip them too.
AGGREGATE_ISO2 = {"ZZ", "XX", "XY", "XZ", "UE", "YU"}

DID_PARTNERS = ["XS", "AL", "MK", "ME"]
DID_CONTROLS = ["AL", "MK", "ME"]

EXPORT_FLOW_NAMES = {"export", "eksport"}


# ---------------------------------------------------------------------------
# Block A — Helpers (private, duplicated to keep eda.py self-contained)
# ---------------------------------------------------------------------------

def _latex_escape(s):
    if not isinstance(s, str):
        return s
    out = s
    for old, new in [
        ("\\", "\\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]:
        out = out.replace(old, new)
    return out


def _to_csv_and_tex(df: pd.DataFrame, name: str) -> None:
    cfg.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.TABLES / f"{name}.csv", index=False)
    safe = df.copy()
    for c in safe.select_dtypes(include="object").columns:
        safe[c] = safe[c].map(_latex_escape)
    safe.to_latex(cfg.TABLES / f"{name}.tex", index=False, escape=False)


def _matplotlib_setup() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


def _save_fig(fig, name: str) -> None:
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _hhi(shares) -> float:
    s = np.asarray(shares, dtype=float)
    return float((s ** 2).sum() * 10000.0)


def _strip_aggregates(partner_long: pd.DataFrame) -> pd.DataFrame:
    """Filter out non-partner aggregate rows (ZZ residual + zero-valued codes)
    for per-partner analyses. Reconciliation should NOT use this — ZZ closes
    the partner↔sections totals identity."""
    return partner_long[~partner_long["iso2"].isin(AGGREGATE_ISO2)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Block B — Parsers
# ---------------------------------------------------------------------------

def _parse_alternating_xy(
    path: Path | str,
    sheet: str,
    code_regex: re.Pattern,
    code_name: str,
    label_name: str,
    data_start_row: int = 4,
) -> pd.DataFrame:
    """Parse files where row 2 holds year labels (forward-fill across two cols
    per year) and row 3 holds Import/Export labels. Data starts at `data_start_row`,
    partner code in col 0. Used by yearly partner (tab02) and yearly chapter (tab03).

    Returns long-format: [code_name, label_name, year, flow, value_eur_thousands].
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    year_row = df.iloc[2].tolist()
    flow_row = df.iloc[3].tolist()

    col_years = [None] * df.shape[1]
    col_flows = [None] * df.shape[1]
    current_y = None
    for c in range(1, df.shape[1]):
        v = year_row[c]
        if isinstance(v, str) and v.strip().isdigit():
            current_y = int(v.strip())
        elif isinstance(v, (int, float)) and not pd.isna(v):
            yi = int(v)
            if 2000 <= yi <= 2030:
                current_y = yi
        col_years[c] = current_y

        fv = flow_row[c]
        if isinstance(fv, str):
            fvl = fv.strip().lower()
            if fvl == "import":
                col_flows[c] = "import"
            elif fvl in EXPORT_FLOW_NAMES:
                col_flows[c] = "export"

    rows = []
    for r in range(data_start_row, df.shape[0]):
        cell0 = df.iloc[r, 0]
        if not isinstance(cell0, str):
            continue
        m = code_regex.match(cell0.strip())
        if not m:
            continue
        code = m.group(1)
        name = m.group(2).strip() or "(unspecified)"
        for c in range(1, df.shape[1]):
            y = col_years[c]; f = col_flows[c]
            if y is None or f is None:
                continue
            v = pd.to_numeric(df.iloc[r, c], errors="coerce")
            if pd.isna(v):
                v = 0.0
            rows.append({
                code_name: code, label_name: name, "year": y,
                "flow": f, "value_eur_thousands": float(v),
            })

    out = pd.DataFrame(rows)
    out = out[out["year"].isin(cfg.YEARS)].reset_index(drop=True)
    return out


def parse_partner_full(path: Path | str | None = None) -> pd.DataFrame:
    """Yearly partner trade (tab02), long-format, both flows."""
    return _parse_alternating_xy(
        path or PARTNER_FILE, "tab02", PARTNER_CODE_REGEX, "iso2", "partner_name",
    )


def parse_chapter_full(path: Path | str | None = None) -> pd.DataFrame:
    """Yearly HS-chapter trade (tab03), long-format, both flows."""
    return _parse_alternating_xy(
        path or CHAPTER_FILE, "tab03", CHAPTER_CODE_REGEX, "hs_chapter", "chapter_name",
    )


def _parse_year_block(
    path: Path | str,
    sheet: str,
    label_regex: re.Pattern,
    label_col_name: str,
) -> pd.DataFrame:
    """Parse files with vertical year blocks (year in col 0, label in col 1,
    Import in col 2, Export in col 3). Used by yearly sections (tab04) and
    yearly BEC (tab01).

    Returns long-format: [year, label_col_name, flow, value_eur_thousands].
    """
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    current_year = None
    rows = []
    for r in range(df.shape[0]):
        y_cell = df.iloc[r, 0]
        yi = None
        if isinstance(y_cell, str) and y_cell.strip().isdigit():
            yi = int(y_cell.strip())
        elif isinstance(y_cell, (int, float)) and not pd.isna(y_cell):
            yi = int(y_cell)
        if yi is not None and 2000 <= yi <= 2030:
            current_year = yi
        if current_year is None:
            continue

        label_cell = df.iloc[r, 1]
        if not isinstance(label_cell, str):
            continue
        if not label_regex.match(label_cell.strip()):
            continue
        label = label_cell.strip()

        imp = pd.to_numeric(df.iloc[r, 2], errors="coerce")
        exp = pd.to_numeric(df.iloc[r, 3], errors="coerce")
        if pd.notna(imp):
            rows.append({"year": current_year, label_col_name: label,
                         "flow": "import", "value_eur_thousands": float(imp)})
        if pd.notna(exp):
            rows.append({"year": current_year, label_col_name: label,
                         "flow": "export", "value_eur_thousands": float(exp)})

    out = pd.DataFrame(rows)
    out = out[out["year"].isin(cfg.YEARS)].reset_index(drop=True)
    return out


def parse_sections_full(path: Path | str | None = None) -> pd.DataFrame:
    """Yearly HS-section trade (tab04), long-format, both flows."""
    return _parse_year_block(
        path or SECTIONS_FILE, "tab04", SECTION_CODE_REGEX, "hs_section",
    )


def parse_bec_full(path: Path | str | None = None) -> pd.DataFrame:
    """Yearly BEC trade (tab01), long-format, both flows."""
    return _parse_year_block(
        path or BEC_FILE, "tab01", BEC_CODE_REGEX, "bec_category",
    )


def parse_monthly_partner(
    path: Path | str | None = None,
    flow: str = "import",
) -> tuple[pd.DataFrame, list]:
    """Monthly partner trade. Returns (long-format df, anomalies list).

    DF columns: [iso2, partner_name, year_month, year, month, flow, value_eur_thousands].
    Defensive: columns whose row-2 label fails `^\\d{4}M\\d{2}$` (e.g. `"2203M06"`
    in the May-2026 monthly export file) are SKIPPED and listed in `anomalies`.
    """
    if path is None:
        path = MONTHLY_IMP if flow == "import" else MONTHLY_EXP
    path = Path(path)

    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, header=None)

    period_row = df.iloc[2].tolist()
    col_periods = [None] * df.shape[1]
    anomalies: list[dict] = []
    for c in range(1, df.shape[1]):
        v = period_row[c]
        if isinstance(v, str):
            vs = v.strip()
            m = MONTH_LABEL_REGEX.match(vs)
            if m:
                yr = int(m.group(1)); mo = int(m.group(2))
                # Defend against ASK typos like "2203M06" (should be "2023M06"):
                # the regex matches but the year is implausible.
                if 2000 <= yr <= 2030 and 1 <= mo <= 12:
                    col_periods[c] = vs
                    continue
                anomalies.append({
                    "file": path.name, "column_index": int(c),
                    "value": vs, "issue": f"year_month_out_of_range_skipped"
                                          f" (year={yr}, month={mo}; likely typo)",
                })
                continue
            anomalies.append({
                "file": path.name, "column_index": int(c),
                "value": vs, "issue": "year_month_label_invalid_skipped",
            })
        elif pd.isna(v):
            continue
        else:
            anomalies.append({
                "file": path.name, "column_index": int(c),
                "value": str(v), "issue": "year_month_label_non_string_skipped",
            })

    rows = []
    for r in range(3, df.shape[0]):
        cell0 = df.iloc[r, 0]
        if not isinstance(cell0, str):
            continue
        m = PARTNER_CODE_REGEX.match(cell0.strip())
        if not m:
            continue
        iso2 = m.group(1)
        name = m.group(2).strip() or "(unspecified)"
        for c in range(1, df.shape[1]):
            p = col_periods[c]
            if p is None:
                continue
            v = pd.to_numeric(df.iloc[r, c], errors="coerce")
            if pd.isna(v):
                v = 0.0
            ym = MONTH_LABEL_REGEX.match(p)
            year = int(ym.group(1)); month = int(ym.group(2))
            rows.append({
                "iso2": iso2, "partner_name": name,
                "year_month": p, "year": year, "month": month,
                "flow": flow, "value_eur_thousands": float(v),
            })

    return pd.DataFrame(rows), anomalies


# ---------------------------------------------------------------------------
# Block C — Reconcilers
# ---------------------------------------------------------------------------

def _classify_delta(d):
    if pd.isna(d):
        return "missing"
    a = abs(d)
    if a <= 0.05:
        return "pass"
    if a <= 1.0:
        return "documented(scope)"
    return "documented(corruption)"


def reconcile_partner_vs_sections(
    partner_long: pd.DataFrame,
    sections_long: pd.DataFrame,
) -> pd.DataFrame:
    """HARD: partner-level totals MUST equal section-level totals per (year, flow)
    within ±0.05%. Raises RuntimeError on failure."""
    p = partner_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("partner_total")
    s = sections_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("sections_total")
    df = pd.concat([p, s], axis=1).reset_index()
    df["abs_delta"] = df["partner_total"] - df["sections_total"]
    df["pct_delta"] = 100.0 * df["abs_delta"] / df["sections_total"]
    df["pass"] = df["pct_delta"].abs() <= 0.05

    _to_csv_and_tex(df, "tbl_ch3_partner_vs_sections_reconciliation")

    if not df["pass"].all():
        failing = df[~df["pass"]]
        raise RuntimeError(
            "HARD reconciliation FAILED: partner-vs-sections totals differ by "
            f">0.05% on {len(failing)} (year,flow) cells:\n{failing.to_string()}"
        )
    return df


def reconcile_partner_vs_chapter_and_bec(
    partner_long: pd.DataFrame,
    chapter_long: pd.DataFrame,
    bec_long: pd.DataFrame,
) -> pd.DataFrame:
    """DIAGNOSTIC: chapter/BEC may differ from partner totals legitimately
    (scope differences). Classify each cell."""
    p = partner_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("partner_total")
    c = chapter_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("chapter_total")
    b = bec_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("bec_total")
    df = pd.concat([p, c, b], axis=1).reset_index()
    df["chapter_pct_delta"] = 100.0 * (df["chapter_total"] - df["partner_total"]) / df["partner_total"]
    df["bec_pct_delta"]     = 100.0 * (df["bec_total"]     - df["partner_total"]) / df["partner_total"]
    df["chapter_status"] = df["chapter_pct_delta"].map(_classify_delta)
    df["bec_status"]     = df["bec_pct_delta"].map(_classify_delta)
    _to_csv_and_tex(df, "tbl_ch3_partner_vs_chapter_bec_diagnostic")
    return df


def reconcile_monthly_vs_yearly(
    monthly_imp: pd.DataFrame,
    monthly_exp: pd.DataFrame,
    partner_long: pd.DataFrame,
) -> pd.DataFrame:
    """DIAGNOSTIC: monthly (sum of 12) vs yearly totals per (year, flow).
    Partial years 2025/2026 excluded by cfg.YEARS filter."""
    mi = monthly_imp[monthly_imp["year"].isin(cfg.YEARS)]
    me = monthly_exp[monthly_exp["year"].isin(cfg.YEARS)]
    monthly = pd.concat([mi, me], ignore_index=True)
    m = monthly.groupby(["year","flow"])["value_eur_thousands"].sum().rename("monthly_total")
    y = partner_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("yearly_total")
    df = pd.concat([m, y], axis=1).reset_index()
    df["abs_delta"] = df["monthly_total"] - df["yearly_total"]
    df["pct_delta"] = 100.0 * df["abs_delta"] / df["yearly_total"]
    df["status"] = df["pct_delta"].map(_classify_delta)
    _to_csv_and_tex(df, "tbl_ch3_monthly_vs_yearly_reconciliation")
    return df


# ---------------------------------------------------------------------------
# Block D — Analyzers
# ---------------------------------------------------------------------------

def annual_totals_by_flow(partner_long: pd.DataFrame) -> pd.DataFrame:
    p = partner_long.groupby(["year","flow"])["value_eur_thousands"].sum().unstack("flow").reset_index()
    p = p.rename(columns={"import": "imports", "export": "exports"})
    p["deficit"] = p["imports"] - p["exports"]
    p["export_to_import_ratio"] = p["exports"] / p["imports"]
    _to_csv_and_tex(p, "tbl_ch3_annual_totals")

    _matplotlib_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p["year"], p["imports"]/1000, marker="o", linewidth=1.8, label="Imports")
    ax.plot(p["year"], p["exports"]/1000, marker="s", linewidth=1.8, label="Exports")
    ax.plot(p["year"], p["deficit"]/1000, marker="^", linestyle="--", linewidth=1.4,
            color="gray", label="Trade deficit")
    ax.axvspan(2018.84, 2020.33, alpha=0.12, color="red", label="Serbia tariff 2018-11/2020-04")
    ax.set_xlabel("Year")
    ax.set_ylabel("€ million")
    ax.set_title("Kosovo annual trade flows, 2010–2024")
    ax.legend(loc="upper left", framealpha=0.92)
    _save_fig(fig, "fig_ch3_imports_exports_totals")
    return p


def zero_count_by_flow(partner_long: pd.DataFrame) -> pd.DataFrame:
    g = partner_long.groupby(["year","flow"])["value_eur_thousands"]
    df = pd.DataFrame({
        "n_zero":  g.apply(lambda s: int((s == 0).sum())),
        "n_total": g.size(),
    }).reset_index()
    df["zero_rate_pct"] = 100.0 * df["n_zero"] / df["n_total"]
    _to_csv_and_tex(df, "tbl_ch3_zero_counts")
    return df


def top_partners_by_flow(partner_long: pd.DataFrame, n: int = 20, year: int = 2024) -> pd.DataFrame:
    partner_long = _strip_aggregates(partner_long)
    sub = partner_long[partner_long["year"] == year]
    pv = sub.pivot_table(index=["iso2","partner_name"], columns="flow",
                         values="value_eur_thousands", aggfunc="sum").fillna(0).reset_index()
    pv = pv.rename(columns={"import":"imports", "export":"exports"})
    pv["share_imports_pct"] = 100 * pv["imports"] / pv["imports"].sum()
    pv["share_exports_pct"] = 100 * pv["exports"] / pv["exports"].sum()

    top_imp = pv.sort_values("imports", ascending=False).head(n).copy()
    top_imp["rank_imports"] = range(1, len(top_imp)+1)
    top_exp = pv.sort_values("exports", ascending=False).head(n).copy()
    top_exp["rank_exports"] = range(1, len(top_exp)+1)

    combined = top_imp[["rank_imports","iso2","partner_name","imports","share_imports_pct"]].merge(
        top_exp[["rank_exports","iso2","partner_name","exports","share_exports_pct"]],
        on=["iso2","partner_name"], how="outer",
    ).sort_values(["rank_imports","rank_exports"], na_position="last").reset_index(drop=True)
    _to_csv_and_tex(combined, "tbl_ch3_top_partners_2024")
    return combined


def concentration_curves(partner_long: pd.DataFrame) -> pd.DataFrame:
    partner_long = _strip_aggregates(partner_long)
    rows = []
    for flow in ["import","export"]:
        for scope_label, sub in [
            ("2024", partner_long[(partner_long["flow"]==flow) & (partner_long["year"]==2024)]),
            ("cum2010_2024", partner_long[partner_long["flow"]==flow]),
        ]:
            g = sub.groupby("iso2")["value_eur_thousands"].sum().sort_values(ascending=False)
            tot = g.sum()
            shares = (g.values / tot) if tot > 0 else np.zeros(len(g))
            row = {"flow": flow, "scope": scope_label,
                   "hhi": _hhi(shares),
                   "n_partners_nonzero": int((g > 0).sum())}
            for K in [5, 10, 20, 50]:
                row[f"top_{K}_share_pct"] = 100 * shares[:K].sum()
            rows.append(row)
    df = pd.DataFrame(rows)
    _to_csv_and_tex(df, "tbl_ch3_concentration")

    _matplotlib_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    for flow, marker, color in [("import","o","C0"), ("export","s","C3")]:
        sub = partner_long[(partner_long["flow"]==flow) & (partner_long["year"]==2024)]
        g = sub.groupby("iso2")["value_eur_thousands"].sum().sort_values(ascending=False)
        tot = g.sum()
        cum = np.cumsum(g.values) / tot if tot > 0 else np.zeros(len(g))
        ax.plot(range(1, len(cum)+1), 100*cum, marker=marker, markersize=3.5,
                label=f"{flow.capitalize()}s (2024)", color=color)
    ax.axhline(80, linestyle=":", color="gray", alpha=0.5)
    ax.axhline(90, linestyle=":", color="gray", alpha=0.5)
    ax.set_xlabel("Partner rank (descending by value)")
    ax.set_ylabel("Cumulative share of trade (%)")
    ax.set_title("Partner concentration, 2024")
    ax.set_xscale("log")
    ax.legend()
    _save_fig(fig, "fig_ch3_concentration_curves")
    return df


def sector_composition_by_flow(sections_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in [2010, 2018, 2024]:
        for flow in ["import","export"]:
            sub = sections_long[(sections_long["year"]==year) & (sections_long["flow"]==flow)]
            tot = sub["value_eur_thousands"].sum()
            top = sub.sort_values("value_eur_thousands", ascending=False).head(5)
            for rank, (_, r) in enumerate(top.iterrows(), 1):
                rows.append({
                    "year": year, "flow": flow, "rank": rank,
                    "hs_section": r["hs_section"],
                    "value_eur_thousands": r["value_eur_thousands"],
                    "share_of_flow_pct": 100 * r["value_eur_thousands"] / tot if tot > 0 else 0.0,
                })
    df = pd.DataFrame(rows)
    _to_csv_and_tex(df, "tbl_ch3_sector_composition")
    return df


def sector_evolution_heatmap(sections_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in cfg.YEARS:
        for flow in ["import","export"]:
            sub = sections_long[(sections_long["year"]==year) & (sections_long["flow"]==flow)]
            tot = sub["value_eur_thousands"].sum()
            for _, r in sub.iterrows():
                rows.append({
                    "year": year, "flow": flow, "hs_section": r["hs_section"],
                    "share_of_flow_pct": 100 * r["value_eur_thousands"] / tot if tot > 0 else 0.0,
                })
    df = pd.DataFrame(rows)
    _to_csv_and_tex(df, "tbl_ch3_sector_evolution")

    _matplotlib_setup()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    for ax, flow in zip(axes, ["import","export"]):
        sub = df[df["flow"]==flow]
        pivot = sub.pivot(index="hs_section", columns="year", values="share_of_flow_pct").fillna(0)
        order = sorted(pivot.index.astype(str), key=lambda s: s[:2])
        pivot = pivot.loc[order]
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([s[:60] for s in pivot.index], fontsize=7)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8, rotation=45)
        ax.set_title(f"{flow.capitalize()}s — share of flow (%)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Sector composition over time, by flow")
    _save_fig(fig, "fig_ch3_sector_heatmap")
    return df


def serbia_trajectory_both_flows(partner_long: pd.DataFrame) -> pd.DataFrame:
    sub = partner_long[partner_long["iso2"].isin(DID_PARTNERS)].copy()
    df = sub.groupby(["iso2","year","flow"])["value_eur_thousands"].sum().reset_index()
    df = df.sort_values(["iso2","year","flow"]).reset_index(drop=True)
    _to_csv_and_tex(df, "tbl_ch3_serbia_trajectory")

    _matplotlib_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, flow in zip(axes, ["import","export"]):
        for iso in DID_PARTNERS:
            x = df[(df["iso2"]==iso) & (df["flow"]==flow)].sort_values("year")
            color = "C3" if iso == "XS" else None
            ax.plot(x["year"], x["value_eur_thousands"]/1000, marker="o", linewidth=1.6,
                    label=iso, color=color)
        ax.axvspan(2018.84, 2020.33, alpha=0.15, color="red")
        ax.set_xlabel("Year"); ax.set_ylabel("€ million")
        ax.set_title(f"{flow.capitalize()}s")
        ax.legend()
    fig.suptitle("Trade with Serbia (XS) and DiD controls (AL, MK, ME)")
    _save_fig(fig, "fig_ch3_serbia_imports_exports")
    return df


def serbia_event_study_monthly(
    monthly_imp: pd.DataFrame,
    monthly_exp: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly Serbia vs control mean (AL+MK+ME), Jan 2017 – Dec 2024.
    Two-panel figure: imports panel + exports panel separately."""
    monthly = pd.concat([monthly_imp, monthly_exp], ignore_index=True)
    monthly = monthly[monthly["iso2"].isin(DID_PARTNERS)].copy()
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    mask = (monthly["date"] >= "2017-01-01") & (monthly["date"] <= "2024-12-31")
    monthly = monthly[mask].copy()

    serbia = monthly[monthly["iso2"]=="XS"][
        ["year_month","year","month","flow","value_eur_thousands"]
    ].rename(columns={"value_eur_thousands":"serbia"})
    ctrl = (monthly[monthly["iso2"].isin(DID_CONTROLS)]
            .groupby(["year_month","year","month","flow"])["value_eur_thousands"]
            .mean().reset_index().rename(columns={"value_eur_thousands":"control_mean"}))
    df = serbia.merge(ctrl, on=["year_month","year","month","flow"], how="outer")
    df = df.sort_values(["flow","year","month"]).reset_index(drop=True)
    _to_csv_and_tex(df, "tbl_ch3_serbia_monthly_event_study")

    _matplotlib_setup()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, flow in zip(axes, ["import","export"]):
        sub = df[df["flow"]==flow].sort_values(["year","month"])
        dates = pd.to_datetime(
            sub["year"].astype(str) + "-" + sub["month"].astype(str).str.zfill(2) + "-01"
        )
        ax.plot(dates, sub["serbia"]/1000, label="Serbia (XS)", color="C3", linewidth=1.4)
        ax.plot(dates, sub["control_mean"]/1000, label="Control mean (AL+MK+ME)",
                color="C0", linewidth=1.4)
        ax.axvline(pd.Timestamp("2018-11-01"), linestyle="--", color="black", alpha=0.6,
                   label="Tariff onset 2018-11")
        ax.axvline(pd.Timestamp("2020-04-01"), linestyle="--", color="gray", alpha=0.6,
                   label="Tariff end 2020-04")
        ax.set_ylabel(f"€ million ({flow}s)")
        ax.set_title(f"{flow.capitalize()}s — Serbia vs control mean")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Serbia trade event study, monthly Jan-2017 to Dec-2024")
    _save_fig(fig, "fig_ch3_serbia_event_study")
    return df


def panel_construction_proposal(partner_long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Top-113 by imports vs by exports + intersection/union + 90%-coverage K."""
    partner_long = _strip_aggregates(partner_long)
    g_imp = partner_long[partner_long["flow"]=="import"].groupby("iso2")["value_eur_thousands"].sum().sort_values(ascending=False)
    g_exp = partner_long[partner_long["flow"]=="export"].groupby("iso2")["value_eur_thousands"].sum().sort_values(ascending=False)

    n = cfg.TOP_N_PARTNERS
    top_imp = set(g_imp.head(n).index)
    top_exp = set(g_exp.head(n).index)
    inter = top_imp & top_exp
    union = top_imp | top_exp

    def k_for_coverage(g: pd.Series, threshold: float = 0.90) -> int:
        tot = g.sum()
        if tot <= 0:
            return 0
        cum = np.cumsum(g.values) / tot
        return int((cum < threshold).sum() + 1)

    k_imp = k_for_coverage(g_imp)
    k_exp = k_for_coverage(g_exp)
    proposed = set(g_imp.head(k_imp).index) | set(g_exp.head(k_exp).index)

    # Coverage check: cumulative trade share of import-derived panel for exports
    exports_in_imports_panel = float(g_exp[g_exp.index.isin(top_imp)].sum() / g_exp.sum()) if g_exp.sum() > 0 else 0.0

    rows = [
        {"metric": f"top-{n} by imports",                "count": len(top_imp)},
        {"metric": f"top-{n} by exports",                "count": len(top_exp)},
        {"metric": "intersection (imports & exports)",   "count": len(inter)},
        {"metric": "union (imports | exports)",          "count": len(union)},
        {"metric": "in imports-top only",                "count": len(top_imp - top_exp)},
        {"metric": "in exports-top only",                "count": len(top_exp - top_imp)},
        {"metric": "K_imp for 90% coverage",             "count": k_imp},
        {"metric": "K_exp for 90% coverage",             "count": k_exp},
        {"metric": "proposed unified panel size",        "count": len(proposed)},
    ]
    df = pd.DataFrame(rows)
    _to_csv_and_tex(df, "tbl_ch3_panel_construction")

    summary = {
        "top_n": int(n),
        "top_imports_count": len(top_imp),
        "top_exports_count": len(top_exp),
        "intersection_count": len(inter),
        "union_count": len(union),
        "in_imports_only": sorted(top_imp - top_exp),
        "in_exports_only": sorted(top_exp - top_imp),
        "K_imp_90pct": k_imp,
        "K_exp_90pct": k_exp,
        "proposed_panel_size": len(proposed),
        "imports_panel_exports_coverage_pct": 100 * exports_in_imports_panel,
    }
    return df, summary


def export_modellability_assessment(partner_long: pd.DataFrame) -> dict:
    """Read Phase-1 partner set from panel_bilateral.parquet (read-only) and
    score export modellability."""
    phase1 = pd.read_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet")
    phase1_iso2 = set(phase1["iso2"].unique())

    sub = partner_long[partner_long["iso2"].isin(phase1_iso2)].copy()
    sub_imp = sub[sub["flow"]=="import"]
    sub_exp = sub[sub["flow"]=="export"]

    n_partners = len(phase1_iso2)
    n_years = sub_exp["year"].nunique()
    effective_n = int(len(sub_exp))

    export_zero_rate = float((sub_exp["value_eur_thousands"] == 0).mean())
    import_zero_rate = float((sub_imp["value_eur_thousands"] == 0).mean())

    var_log_exp = float(np.var(np.log1p(sub_exp["value_eur_thousands"].values)))
    var_log_imp = float(np.var(np.log1p(sub_imp["value_eur_thousands"].values)))

    nonzero_years = sub_exp[sub_exp["value_eur_thousands"]>0].groupby("iso2")["year"].nunique()
    modellable = int((nonzero_years > 5).sum())
    partners_with_any_export = int(sub_exp[sub_exp["value_eur_thousands"]>0]["iso2"].nunique())
    structural_zeros = n_partners - partners_with_any_export

    out = {
        "phase1_panel_iso2_count": n_partners,
        "phase1_panel_years": int(n_years),
        "effective_n_export_rows": effective_n,
        "export_zero_rate": export_zero_rate,
        "import_zero_rate": import_zero_rate,
        "var_log1p_exports": var_log_exp,
        "var_log1p_imports": var_log_imp,
        "var_log_ratio_exp_to_imp": var_log_exp / var_log_imp if var_log_imp > 0 else None,
        "partners_with_gt5_nonzero_export_years": modellable,
        "partners_with_any_export": partners_with_any_export,
        "partners_with_zero_export_years": structural_zeros,
    }
    cfg.METRICS.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICS / "export_modellability.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Block E — Adaptive analyses
# ---------------------------------------------------------------------------

def eda_partner_asymmetry(partner_long: pd.DataFrame) -> pd.DataFrame:
    """Adaptive #1: partner-level import vs export imbalance.

    For each partner: cumulative imports and exports 2010-2024, imbalance ratio
    in [-1, +1] where -1 = exports-only, +1 = imports-only. Classify partners
    into asymmetry buckets.
    """
    long = _strip_aggregates(partner_long)
    wide = long.groupby(["iso2", "partner_name", "flow"])["value_eur_thousands"].sum().unstack("flow").fillna(0).reset_index()
    wide = wide.rename(columns={"import":"cum_imports","export":"cum_exports"})
    wide["total"] = wide["cum_imports"] + wide["cum_exports"]
    wide = wide[wide["total"] > 0].copy()
    wide["imbalance_ratio"] = (wide["cum_imports"] - wide["cum_exports"]) / wide["total"]

    def classify(r):
        if r["imbalance_ratio"] >= 0.95:  return "imports_only"
        if r["imbalance_ratio"] >= 0.50:  return "imports_dominant"
        if r["imbalance_ratio"] >= -0.50: return "balanced"
        if r["imbalance_ratio"] >= -0.95: return "exports_dominant"
        return "exports_only"
    wide["classification"] = wide.apply(classify, axis=1)
    wide = wide.sort_values("total", ascending=False).reset_index(drop=True)
    _to_csv_and_tex(
        wide[["iso2","partner_name","cum_imports","cum_exports","total",
              "imbalance_ratio","classification"]],
        "tbl_ch3_partner_asymmetry",
    )

    # Figure: scatter log10(cum_imports+1) vs log10(cum_exports+1)
    _matplotlib_setup()
    fig, ax = plt.subplots(figsize=(8, 8))
    colour_map = {"imports_only":"C0", "imports_dominant":"C9",
                  "balanced":"gray", "exports_dominant":"C1", "exports_only":"C3"}
    for cls, group in wide.groupby("classification"):
        ax.scatter(np.log10(group["cum_imports"]+1), np.log10(group["cum_exports"]+1),
                   s=18, alpha=0.7, label=cls, color=colour_map.get(cls, "k"))
    # Diagonal
    mx = max(np.log10(wide["cum_imports"]+1).max(), np.log10(wide["cum_exports"]+1).max())
    ax.plot([0, mx], [0, mx], "k--", alpha=0.3, linewidth=0.8, label="parity")
    # Annotate big DiD partners
    for iso in ["XS", "AL", "MK", "ME", "DE", "TR", "CN"]:
        row = wide[wide["iso2"]==iso]
        if not row.empty:
            ax.annotate(iso, (np.log10(row["cum_imports"].iloc[0]+1),
                              np.log10(row["cum_exports"].iloc[0]+1)),
                        fontsize=8, color="black", weight="bold")
    ax.set_xlabel("log10(cumulative imports + 1) (EUR thousands)")
    ax.set_ylabel("log10(cumulative exports + 1) (EUR thousands)")
    ax.set_title("Partner import vs export asymmetry, cumulative 2010–2024")
    ax.legend(loc="upper left", fontsize=8)
    _save_fig(fig, "fig_ch3_partner_asymmetry")
    return wide


def eda_monthly_breakpoint(monthly_imp: pd.DataFrame) -> pd.DataFrame:
    """Adaptive #2: precise month-of-onset / month-of-recovery for Serbia
    bilateral imports around the 2018-11 tariff onset and 2020-04 removal.

    Identifies the months in candidate windows with the largest month-on-month
    deviations relative to the 2017-2021 baseline volatility. Descriptive only.
    """
    sub = monthly_imp[(monthly_imp["iso2"]=="XS") &
                      (monthly_imp["year"].between(2017, 2021))].copy()
    sub["date"] = pd.to_datetime(
        sub["year"].astype(str) + "-" + sub["month"].astype(str).str.zfill(2) + "-01"
    )
    sub = sub.sort_values("date").reset_index(drop=True)
    sub["mom_pct"] = 100 * sub["value_eur_thousands"].pct_change()

    # Z-score baseline: non-event months only (exclude 2018-10..2019-02 and 2020-01..2020-07)
    onset_win = (sub["date"] >= "2018-08-01") & (sub["date"] <= "2019-02-28")
    recovery_win = (sub["date"] >= "2020-01-01") & (sub["date"] <= "2020-07-31")
    event_mask = onset_win | recovery_win
    baseline = sub[~event_mask]["mom_pct"]
    baseline_mu = float(baseline.mean())
    baseline_sigma = float(baseline.std())
    sub["mom_z"] = (sub["mom_pct"] - baseline_mu) / baseline_sigma

    onset_candidates = sub[onset_win].nsmallest(3, "mom_pct")[["date","value_eur_thousands","mom_pct","mom_z"]]
    onset_candidates["window"] = "onset"
    recovery_candidates = sub[recovery_win].nlargest(3, "mom_pct")[["date","value_eur_thousands","mom_pct","mom_z"]]
    recovery_candidates["window"] = "recovery"
    flagged = pd.concat([onset_candidates, recovery_candidates]).reset_index(drop=True)
    flagged["date"] = flagged["date"].astype(str)
    _to_csv_and_tex(flagged, "tbl_ch3_serbia_monthly_breakpoint")

    # Figure
    _matplotlib_setup()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sub["date"], sub["value_eur_thousands"]/1000, color="C3", linewidth=1.4,
            label="Serbia (XS) monthly imports")
    # Reference policy dates
    ax.axvline(pd.Timestamp("2018-11-01"), color="black", linestyle="--", alpha=0.6,
               label="Reference: tariff onset 2018-11")
    ax.axvline(pd.Timestamp("2020-04-01"), color="gray", linestyle="--", alpha=0.6,
               label="Reference: tariff end 2020-04")
    # Detected breakpoints
    for _, r in flagged.iterrows():
        d = pd.Timestamp(r["date"])
        v = sub[sub["date"]==d]["value_eur_thousands"].iloc[0]/1000
        color = "C0" if r["window"]=="onset" else "C2"
        ax.scatter(d, v, s=80, marker="v", color=color, edgecolors="black", zorder=5)
    ax.set_xlabel("Month")
    ax.set_ylabel("€ million (imports)")
    ax.set_title("Serbia monthly imports — month-of-largest-deviation in event windows")
    ax.legend(fontsize=9, loc="upper left")
    _save_fig(fig, "fig_ch3_serbia_monthly_breakpoint")
    return flagged


# Events for adaptive #3. The "2020" event combines the tariff removal (Apr)
# and the COVID first-wave shock (Mar-May), which are inseparable in
# annual data; both are noted in the event label.
SECTOR_EVENTS = {
    2016: "SAA into force",
    2018: "Serbia tariff onset (Nov)",
    2020: "Tariff removal Apr + COVID shock",
    2022: "Energy / Ukraine shock",
}


def eda_sector_event_response(sections_long: pd.DataFrame) -> pd.DataFrame:
    """Adaptive #3: per-section share-of-flow change in event years, expressed
    as z-score relative to that section's non-event-year YoY share-change
    volatility.

    PURELY DESCRIPTIVE. Does NOT establish causality. Surfaces (section, flow,
    event) triples whose share movement is temporally aligned with the event.
    """
    # Compute per-year share-of-flow per section
    sections_long = sections_long.copy()
    total_per_year_flow = sections_long.groupby(["year","flow"])["value_eur_thousands"].sum().rename("year_total")
    sub = sections_long.merge(total_per_year_flow.reset_index(), on=["year","flow"])
    sub["share_pct"] = 100 * sub["value_eur_thousands"] / sub["year_total"]

    # Pivot to (section, flow) × year
    pivot = sub.pivot_table(index=["hs_section","flow"], columns="year",
                            values="share_pct", aggfunc="first")
    # YoY delta
    yoy = pivot.diff(axis=1)
    event_years = list(SECTOR_EVENTS.keys())

    rows = []
    for (section, flow), s in yoy.iterrows():
        non_event = s.drop(event_years, errors="ignore").dropna()
        if len(non_event) < 5:
            continue
        sigma = float(non_event.std())
        if sigma <= 0 or pd.isna(sigma):
            continue
        for ev in event_years:
            if ev not in s.index or pd.isna(s[ev]):
                continue
            pre_share = float(pivot.loc[(section, flow), ev-1]) if (ev-1) in pivot.columns else None
            post_share = float(pivot.loc[(section, flow), ev])
            delta_pp = float(s[ev])
            z = delta_pp / sigma
            rows.append({
                "hs_section": section, "flow": flow, "event_year": ev,
                "event_label": SECTOR_EVENTS[ev],
                "pre_share_pct": pre_share, "post_share_pct": post_share,
                "delta_pp": delta_pp,
                "baseline_sigma_pp": sigma, "z_score": z,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("z_score", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    _to_csv_and_tex(df, "tbl_ch3_sector_event_response")

    # Complementary view: substantive percentage-point shifts (|Δpp| >= 2),
    # regardless of z-score. Captures large absolute structural movements
    # that high-baseline-volatility sections mask in the z-score view.
    pp = df[df["delta_pp"].abs() >= 2.0].copy()
    pp = pp.sort_values("delta_pp", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    _to_csv_and_tex(pp, "tbl_ch3_sector_event_response_ppshift")

    # Figure: top-10 most-reactive (section, event, flow) triples
    top = df.head(10).copy().iloc[::-1]
    top["label"] = top.apply(
        lambda r: f"{r['hs_section'][:38]}  [{r['flow'][:3]}]  {r['event_year']}",
        axis=1,
    )
    _matplotlib_setup()
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["C0" if f == "import" else "C3" for f in top["flow"]]
    ax.barh(range(len(top)), top["delta_pp"], color=colors, alpha=0.75)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ share of flow (percentage points), event year vs prior year")
    ax.set_title("Top-10 sector-flow movements temporally aligned with policy/macro events")
    # Legend by colour
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="C0", label="Imports"),
                       Patch(color="C3", label="Exports")], loc="lower right")
    _save_fig(fig, "fig_ch3_sector_event_response")
    return df


def eda_chapter_anomaly_check(
    chapter_long: pd.DataFrame,
    sections_long: pd.DataFrame,
    partner_long: pd.DataFrame,
) -> dict:
    """Bonus side-check: is the 2018 chapter-file anomaly an isolated ASK error
    confined to a few chapters? Returns a small dict for §10."""
    # Annual totals from each source
    ch_tot = chapter_long.groupby(["year","flow"])["value_eur_thousands"].sum().unstack("flow")
    se_tot = sections_long.groupby(["year","flow"])["value_eur_thousands"].sum().unstack("flow")
    pa_tot = _strip_aggregates(partner_long).groupby(["year","flow"])["value_eur_thousands"].sum().unstack("flow")
    # Add ZZ back for partner totals (matches sections)
    pa_full = partner_long.groupby(["year","flow"])["value_eur_thousands"].sum().unstack("flow")

    cmp = pd.DataFrame({
        "partner_total_imports": pa_full["import"],
        "sections_total_imports": se_tot["import"],
        "chapter_total_imports": ch_tot["import"],
    })
    cmp["chapter_vs_partner_ratio"] = cmp["chapter_total_imports"] / cmp["partner_total_imports"]

    # Per-chapter 2018 outliers: which chapters have z > 3 vs their own non-2018 mean
    ch = chapter_long[chapter_long["flow"]=="import"]
    pivot = ch.pivot_table(index="hs_chapter", columns="year",
                           values="value_eur_thousands", aggfunc="sum").fillna(0)
    other_years = [y for y in pivot.columns if y != 2018]
    mu = pivot[other_years].mean(axis=1)
    sd = pivot[other_years].std(axis=1)
    z2018 = (pivot[2018] - mu) / sd.replace(0, np.nan)
    extreme = z2018.dropna().sort_values(ascending=False).head(8)

    return {
        "chapter_vs_partner_ratio_by_year": cmp["chapter_vs_partner_ratio"].round(4).to_dict(),
        "anomaly_confined_to_2018": bool(
            (cmp["chapter_vs_partner_ratio"].drop(2018, errors="ignore") - 1).abs().max() < 0.05
        ),
        "top_8_chapters_with_2018_z_score_above_3": {
            str(idx)[:60]: float(z) for idx, z in extreme.items() if z > 3
        },
    }
