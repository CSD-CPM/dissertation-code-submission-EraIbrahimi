"""The 12 features of the implementation plan, §3.

Each feature has one authoritative operationalisation. Call `engineer_all(panel)`
on a merged panel to add every feature column. Missing external data (WDI,
CEPII) produces NaNs — features.py will NOT impute silently; downstream code
decides.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def add_policy_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add serbia_tariff, saa_in_force, covid, cefta_member.

    serbia_tariff = 1 iff iso2=='XS' AND year in {2018,2019,2020}  (Plan §6)
    saa_in_force  = 1 iff year >= 2016 AND iso2 in EU_ISO2        (§11.6 default)
    covid         = 1 iff year == 2020
    cefta_member  = 1 iff iso2 in CEFTA_MEMBERS_ISO2
    """
    df = df.copy()
    df["serbia_tariff"] = (
        (df["iso2"] == "XS") & (df["year"].isin(cfg.SERBIA_TARIFF_FEATURE_YEARS))
    ).astype(int)
    df["saa_in_force"] = (
        (df["year"] >= cfg.SAA_IN_FORCE_FROM) & (df["iso2"].isin(cfg.EU_ISO2))
    ).astype(int)
    df["covid"] = (df["year"] == cfg.COVID_YEAR).astype(int)
    df["cefta_member"] = df["iso2"].isin(cfg.CEFTA_MEMBERS_ISO2).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_trend"] = df["year"] - min(cfg.YEARS)
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """lagged_imports_log1p(t-1) and partner_import_share_lag(t-1).

    Lagged share = Imports_{i,t-1} / sum_i Imports_{i,t-1}. Computed from the
    panel itself. 2010 rows have NaN lags — downstream decides whether to drop.
    """
    df = df.sort_values(["iso2", "year"]).copy()
    df["lagged_imports_log1p"] = (
        np.log1p(df.groupby("iso2")["imports_eur_thousands"].shift(1).fillna(0))
    )
    # Mark true NaN for 2010 rows so consumers can distinguish from log1p(0)
    df.loc[df["year"] == min(cfg.YEARS), "lagged_imports_log1p"] = np.nan

    # Partner share of lagged imports
    df["prev_imports"] = df.groupby("iso2")["imports_eur_thousands"].shift(1)
    year_totals = df.groupby("year")["prev_imports"].transform("sum")
    df["partner_import_share_lag"] = df["prev_imports"] / year_totals
    df.loc[df["year"] == min(cfg.YEARS), "partner_import_share_lag"] = np.nan
    df = df.drop(columns=["prev_imports"])
    return df


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """ln_distance, contiguity, common_language from CEPII-resolved dyads.

    If the resolve step was not run (network blocked), columns are all NaN.
    """
    df = df.copy()
    if "dist_km" in df.columns:
        df["ln_distance"] = np.log(df["dist_km"].where(df["dist_km"] > 0))
    else:
        df["ln_distance"] = np.nan
    for col in ("contiguity", "common_language"):
        if col not in df.columns:
            df[col] = np.nan
    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """ln_partner_gdp and ln_kosovo_gdp from WDI-joined columns."""
    df = df.copy()
    if "gdp_usd_current" in df.columns:
        df["ln_partner_gdp"] = np.log(df["gdp_usd_current"].where(df["gdp_usd_current"] > 0))
    else:
        df["ln_partner_gdp"] = np.nan
    if "gdp_usd_current_kosovo" in df.columns:
        df["ln_kosovo_gdp"] = np.log(
            df["gdp_usd_current_kosovo"].where(df["gdp_usd_current_kosovo"] > 0))
    else:
        df["ln_kosovo_gdp"] = np.nan
    return df


def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_policy_dummies(df)
    df = add_time_features(df)
    df = add_lagged_features(df)
    df = add_structural_features(df)
    df = add_macro_features(df)
    return df


def feature_coverage_report(df: pd.DataFrame) -> dict:
    """Return per-feature non-null coverage pct over the panel."""
    return {
        f: round(df[f].notna().mean() * 100, 2)
        for f in cfg.FEATURE_ORDER if f in df.columns
    }
