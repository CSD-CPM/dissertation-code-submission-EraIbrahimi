"""Phase 1 data pipeline.

Responsibilities (per implementation plan §2):
1. Parse ASK tab02.xlsx into a long bilateral panel.
2. Parse ASK tab04.xlsx into a long sectoral panel.
3. Apply the top-113 partner filter (yielding the plan's n = 1,695).
4. Acquire WDI, CEPII, Gap Institute (network-dependent; Eurostat optional).
5. Merge into a clean, feature-ready bilateral panel.
6. Report sample size, zero count, Serbia trajectory, join rates, anomalies.

This module is idempotent: re-running rebuilds parquet from raw xlsx + cached
external downloads. Network calls only happen if caches are missing.
"""
from __future__ import annotations

import re
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config as cfg


# =============================================================================
# 1. ASK tab02 — bilateral imports
# =============================================================================

def parse_tab02(path: Path | str = None) -> pd.DataFrame:
    """Return a long dataframe: iso2, partner_name, year, imports_eur_thousands.

    The ZZ 'total' partner aggregate is dropped; all other partner codes kept.
    """
    path = Path(path or (cfg.ROOT.parent / "data" / "ask" / "ask_yearly_partner_trade_2010_2024.xlsx"))
    df = pd.read_excel(path, sheet_name="tab02", header=None)

    # Row 2 carries year labels (as strings like '2024'); row 3 carries
    # Import/Export labels; data starts at row 4, col 0 holds 'ISO2:NAME'.
    year_row = df.iloc[2].tolist()
    kind_row = df.iloc[3].tolist()

    # Forward-fill year label across the pair of Import/Export columns.
    last_year = None
    col_meta = []  # list of (col_idx, year, 'Import'|'Export')
    for c in range(1, df.shape[1]):
        val = year_row[c]
        if isinstance(val, str) and val.strip().isdigit():
            last_year = int(val.strip())
        elif isinstance(val, (int, float)) and pd.notna(val):
            last_year = int(val)
        col_meta.append((c, last_year, kind_row[c]))

    partner_mask = df.iloc[:, 0].astype(str).str.match(r"^[A-Z]{2}:")
    pdf = df.loc[partner_mask].copy()
    pdf["iso2"] = pdf.iloc[:, 0].str.split(":", n=1).str[0]
    pdf["partner_name"] = pdf.iloc[:, 0].str.split(":", n=1).str[1].str.strip()
    pdf = pdf[pdf["iso2"] != "ZZ"].reset_index(drop=True)

    rows = []
    for _, pr in pdf.iterrows():
        for c, yr, kind in col_meta:
            if kind != "Import" or yr is None:
                continue
            v = pd.to_numeric(pr.iloc[c], errors="coerce")
            rows.append({
                "iso2": pr["iso2"],
                "partner_name": pr["partner_name"],
                "year": int(yr),
                "imports_eur_thousands": float(v) if pd.notna(v) else 0.0,
            })
    long = pd.DataFrame(rows)
    long = long[long["year"].isin(cfg.YEARS)].reset_index(drop=True)
    return long


DROP_ISO2_CODES = {"XX"}  # ASK 'unknown / not specified' aggregate


def apply_top_n_filter(panel: pd.DataFrame, n: int = cfg.TOP_N_PARTNERS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter to top-n *identifiable* partners by cumulative imports 2010–2024.

    Drops non-country aggregates (ASK's 'XX') before ranking so the top-n
    selection yields n real countries. This is documented in
    EUROSTAT_DECISION.md § partner-filter anomaly.

    Returns
    -------
    filtered_panel : pd.DataFrame
        The top-n partners only (n × 15 rows, balanced).
    activity : pd.DataFrame
        Per-partner cumulative imports (for auditing which partners were kept).
    """
    eligible = panel[~panel["iso2"].isin(DROP_ISO2_CODES)]
    activity = (eligible.groupby(["iso2", "partner_name"])["imports_eur_thousands"]
                        .agg(total="sum", n_nonzero_years=lambda s: int((s > 0).sum()))
                        .reset_index()
                        .sort_values("total", ascending=False))
    kept = set(activity.head(n)["iso2"])
    filtered = panel[panel["iso2"].isin(kept)].copy()
    return filtered, activity


# =============================================================================
# 2. ASK tab04 — sectoral imports (21 HS sections × 15 years)
# =============================================================================

def parse_tab04(path: Path | str = None) -> pd.DataFrame:
    path = Path(path or (cfg.ROOT.parent / "data" / "ask" / "ask_yearly_hs_sections_trade_2010_2024.xlsx"))
    df = pd.read_excel(path, sheet_name="tab04", header=None)

    current_year = None
    rows = []
    for _, r in df.iterrows():
        y = r.iloc[0]
        sec = r.iloc[1]
        if isinstance(y, str) and y.strip().isdigit():
            yi = int(y.strip())
            if 2000 < yi < 2030:
                current_year = yi
        elif isinstance(y, (int, float)) and pd.notna(y) and 2000 < int(y) < 2030:
            current_year = int(y)
        if current_year is not None and isinstance(sec, str) and re.match(r"^\d{2}\s", sec):
            imp = pd.to_numeric(r.iloc[2], errors="coerce")
            exp = pd.to_numeric(r.iloc[3], errors="coerce")
            rows.append({
                "year": current_year,
                "hs_section": sec.strip(),
                "imports_eur_thousands": float(imp) if pd.notna(imp) else 0.0,
                "exports_eur_thousands": float(exp) if pd.notna(exp) else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["year", "hs_section"]).reset_index(drop=True)


# =============================================================================
# 3. Partner master (iso2 -> iso3 -> wdi -> cepii) used for every join
# =============================================================================

ASK_ISO2_TO_ISO3_OVERRIDES = {
    # ASK uses non-ISO codes for some entities; map them here.
    "XS": "SRB",   # Serbia (post-2006) in ASK is coded XS
    "XK": "XKX",   # Kosovo (self)
    "CS": "SCG",   # Serbia and Montenegro (historical, all-zero in post-2010 data)
    "YU": "YUG",   # Yugoslavia (historical)
    "AN": "ANT",   # Netherlands Antilles (historical)
    "XM": None,    # unknown aggregate
    "XZ": None,    # unknown aggregate
    "ZZ": None,    # total
    "GB": "GBR",   # United Kingdom
    "TL": "TLS",   # Timor-Leste
    "PS": "PSE",   # Palestine
    "UK": "GBR",
    "EL": "GRC",   # Greece (EU alt code)
    "LI": "LIE",   # Liechtenstein (active partner; missing from ref CSV)
    "TW": "TWN",   # Taiwan, Province of China
    "XX": None,    # ASK 'unknown / not specified' aggregate; documented anomaly
}

def build_partner_master(tab02_long: pd.DataFrame, iso_ref_csv: Path | str = None) -> pd.DataFrame:
    """Build iso2 -> iso3 mapping with ASK overrides.

    Requires the provided country_codes_reference.csv (columns: country_code,
    country_name, country_iso2, country_iso3) OR falls back to pycountry.
    """
    ref = pd.read_csv(iso_ref_csv or (cfg.DATA_RAW / "country_codes_reference.csv"))
    iso_map = dict(zip(ref["country_iso2"], ref["country_iso3"]))
    # apply overrides
    iso_map.update({k: v for k, v in ASK_ISO2_TO_ISO3_OVERRIDES.items() if v is not None})

    partners = tab02_long[["iso2", "partner_name"]].drop_duplicates().sort_values("iso2").reset_index(drop=True)
    partners["iso3"] = partners["iso2"].map(iso_map)
    unmapped = partners[partners["iso3"].isna()]["iso2"].tolist()
    if unmapped:
        print(f"[partner_master] WARNING: {len(unmapped)} iso2 codes unmapped: {unmapped}")
    # stable integer id for clustering
    partners["partner_id"] = range(len(partners))
    return partners


# =============================================================================
# 4. External data — WDI, CEPII, Gap Institute
# =============================================================================

def fetch_wdi(partner_iso3_list, years=range(2009, 2025), cache: Path | str = None) -> pd.DataFrame:
    """Download WDI indicators (Plan §7.1 Task A). Cached to CSV.

    Uses wbdata (primary). If the cache exists, return it unchanged.
    """
    cache = Path(cache or (cfg.DATA_RAW / "wdi_macro.csv"))
    if cache.exists():
        print(f"[wdi] using cache: {cache}")
        return pd.read_csv(cache)

    try:
        import wbdata
    except ImportError as e:
        raise RuntimeError("wbdata not installed. Run: pip install wbdata") from e

    import datetime as dt
    date_range = (dt.datetime(min(years), 1, 1), dt.datetime(max(years), 12, 31))
    iso3_list = list(set(partner_iso3_list) | {"XKX"})  # always include Kosovo itself

    print(f"[wdi] fetching {len(iso3_list)} countries × {len(cfg.WDI_INDICATORS)} indicators ...")
    raw = wbdata.get_dataframe(
        cfg.WDI_INDICATORS,
        country=iso3_list,
        date=date_range,
    ).reset_index()
    # wbdata returns MultiIndex (country, date); normalise
    raw = raw.rename(columns={"country": "country_name", "date": "year"})
    # Year is returned as datetime; extract the integer year
    if pd.api.types.is_datetime64_any_dtype(raw["year"]):
        raw["year"] = raw["year"].dt.year
    else:
        raw["year"] = pd.to_numeric(raw["year"], errors="coerce").astype("Int64")
    raw.to_csv(cache, index=False)
    print(f"[wdi] cached to {cache}: {len(raw)} rows")
    return raw


def fetch_cepii(cache: Path | str = None) -> pd.DataFrame:
    """Download CEPII GeoDist bilateral distances (Plan §7.1 Task B).

    Filters to Kosovo dyads. If Kosovo is absent from the CEPII file, falls
    back to a Haversine computation from capital coordinates.
    """
    cache = Path(cache or (cfg.DATA_RAW / "cepii_geodist.csv"))
    if cache.exists():
        print(f"[cepii] using cache: {cache}")
        return pd.read_csv(cache)

    import requests, io, zipfile
    url = "http://www.cepii.fr/distance/dist_cepii.zip"
    print(f"[cepii] downloading {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = None
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = z.namelist()
        # Prefer CSV → Stata → Excel (in that order).
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        dta_names = [n for n in names if n.lower().endswith(".dta")]
        xls_names = [n for n in names if n.lower().endswith((".xls", ".xlsx"))]
        if csv_names:
            with z.open(csv_names[0]) as fp:
                df = pd.read_csv(fp)
        elif dta_names:
            with z.open(dta_names[0]) as fp:
                df = pd.read_stata(io.BytesIO(fp.read()))
        elif xls_names:
            with z.open(xls_names[0]) as fp:
                df = pd.read_excel(io.BytesIO(fp.read()))
        else:
            raise RuntimeError(f"No CSV/Stata/Excel found inside CEPII zip; got {names}")
    df.to_csv(cache, index=False)
    print(f"[cepii] cached to {cache}: {len(df)} rows")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def resolve_kosovo_distances(partners: pd.DataFrame, cepii: pd.DataFrame) -> pd.DataFrame:
    """Return iso2 -> {dist_km, contiguity, common_language}.

    Tries CEPII first (Kosovo as XKO or XKX). Falls back to a Haversine
    computation using capital coordinates and hand-curated
    contiguity/language dummies for the five Kosovo land neighbours.
    """
    # Kosovo's land neighbours and shared language partners (hand-curated)
    KOSOVO_NEIGHBOURS = {"AL", "MK", "ME", "XS", "BA"}  # Bosnia shares no border; kept for safety
    KOSOVO_NEIGHBOURS_STRICT = {"AL", "MK", "ME", "XS"}  # real land borders
    KOSOVO_ALBANIAN_PARTNERS = {"AL", "MK", "ME"}  # Albanian-speaking minorities
    KOSOVO_SERBIAN_PARTNERS = {"XS", "ME", "BA"}   # Serbian-speaking partners

    out_rows = []
    # CEPII candidate codes for Kosovo
    kos_ceps = [c for c in ["XKO", "XKX", "KSV", "RKS"] if ("iso_o" in cepii.columns and c in set(cepii["iso_o"]))]

    if kos_ceps:
        # CEPII has Kosovo — use it
        ko = cepii[cepii["iso_o"].isin(kos_ceps)]
        for _, pr in partners.iterrows():
            match = ko[ko["iso_d"] == pr["iso3"]]
            if len(match):
                m = match.iloc[0]
                out_rows.append({
                    "iso2": pr["iso2"],
                    "dist_km": float(m.get("dist", np.nan)),
                    "contiguity": int(m.get("contig", 0) or 0),
                    "common_language": int((m.get("comlang_off", 0) or 0) or (m.get("comlang_ethno", 0) or 0)),
                    "distance_source": "cepii",
                })
            else:
                out_rows.append({"iso2": pr["iso2"], "dist_km": np.nan,
                                 "contiguity": 0, "common_language": 0,
                                 "distance_source": "missing"})
    else:
        # CEPII lacks Kosovo -> Haversine fallback using a capital-coordinates table
        coords = _load_capital_coords()
        pristina = coords.get("XKS") or coords.get("XKX") or (42.67, 21.17)
        for _, pr in partners.iterrows():
            iso3 = pr["iso3"]
            cap = coords.get(iso3)
            if cap is None:
                out_rows.append({"iso2": pr["iso2"], "dist_km": np.nan,
                                 "contiguity": 0, "common_language": 0,
                                 "distance_source": "missing"})
                continue
            d = haversine_km(pristina[0], pristina[1], cap[0], cap[1])
            out_rows.append({
                "iso2": pr["iso2"],
                "dist_km": d,
                "contiguity": int(pr["iso2"] in KOSOVO_NEIGHBOURS_STRICT),
                "common_language": int(pr["iso2"] in (KOSOVO_ALBANIAN_PARTNERS | KOSOVO_SERBIAN_PARTNERS)),
                "distance_source": "haversine_fallback",
            })
    return pd.DataFrame(out_rows)


def _load_capital_coords() -> dict:
    """Hand-curated iso3 -> (lat, lon) for all countries we care about.

    Ship a compact table in data/raw/capital_coords.csv; this loader is the
    single lookup for Haversine fallback. The CSV must be present for the
    fallback branch; if it is not, resolve_kosovo_distances will emit missing
    rows and the join report will flag them.
    """
    p = cfg.DATA_RAW / "capital_coords.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {r["iso3"]: (float(r["lat"]), float(r["lon"])) for _, r in df.iterrows()}


def curate_gap_institute_csv(out_path: Path | str = None) -> pd.DataFrame:
    """Seed the Gap Institute (2019) sector diversion CSV with the values
    cited in the dissertation plan (Plan §6 Pillar 2 supplementary enrichment).

    Era should reconcile these numbers against the actual PDF once downloaded
    locally; this seed file exists so downstream sector analysis has a concrete
    artefact to consume even before the PDF is in hand.
    """
    out_path = Path(out_path or (cfg.DATA_RAW / "gap_institute_diversion.csv"))
    data = [
        {"hs_section": "10 Cereals",           "pre_tariff_serbia_share_pct": 83.0,
         "post_tariff_substitutes": "North Macedonia; Bulgaria; Turkey"},
        {"hs_section": "04 Prepared foodstufs beverageas and tobacco",
         "pre_tariff_serbia_share_pct": 62.0,
         "post_tariff_substitutes": "North Macedonia; Albania; Bulgaria; Croatia"},
        {"hs_section": "22 Beverages (subset of section 04)",
         "pre_tariff_serbia_share_pct": 55.0,
         "post_tariff_substitutes": "North Macedonia; Turkey"},
    ]
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=False)
    return df


# =============================================================================
# 5. Merge everything into the feature-ready panel
# =============================================================================

def build_merged_panel(
    tab02_panel: pd.DataFrame,
    partners: pd.DataFrame,
    wdi: pd.DataFrame | None,
    cepii_resolved: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join tab02 ← partner_master ← WDI ← CEPII-resolved.

    WDI and CEPII may be None (network-blocked); missing feature columns are
    written as NaN and flagged in the report. The caller decides whether to
    proceed with feature engineering.
    """
    p = tab02_panel.merge(partners[["iso2", "iso3", "partner_id"]], on="iso2", how="left")

    if wdi is not None and len(wdi):
        # wdi expected columns: country_name OR country, year, + rename-mapped indicators
        wdi_cols = list(cfg.WDI_INDICATORS.values())
        # Attach partner GDP etc.
        # Standardise: wbdata returns the country name; we need iso3 to merge.
        # Prefer a user-supplied 'country_iso3' column if present.
        if "country_iso3" in wdi.columns:
            wdi_key = "country_iso3"
        else:
            # attempt to derive iso3 from country name via pycountry, with
            # explicit overrides for entities pycountry does not resolve
            # (Kosovo most importantly — wbdata returns it as "Kosovo" but
            # pycountry has no record, so it would silently drop and the
            # gdp_usd_current_kosovo column would end up 100 % NaN).
            WDI_NAME_OVERRIDES = {
                "Kosovo": "XKX",
                "Korea, Rep.": "KOR",
                "Korea, Dem. People's Rep.": "PRK",
                "Egypt, Arab Rep.": "EGY",
                "Iran, Islamic Rep.": "IRN",
                "Russian Federation": "RUS",
                "Slovak Republic": "SVK",
                "Czechia": "CZE",
                "Turkiye": "TUR",
                "Türkiye": "TUR",
                "Hong Kong SAR, China": "HKG",
                "Macao SAR, China": "MAC",
                "Taiwan, China": "TWN",
                "Viet Nam": "VNM",
                "Lao PDR": "LAO",
                "Syrian Arab Republic": "SYR",
                "Venezuela, RB": "VEN",
                "Yemen, Rep.": "YEM",
                "Bahamas, The": "BHS",
                "Gambia, The": "GMB",
                "Congo, Dem. Rep.": "COD",
                "Congo, Rep.": "COG",
                "Cote d'Ivoire": "CIV",
                "Cabo Verde": "CPV",
                "Eswatini": "SWZ",
                "Micronesia, Fed. Sts.": "FSM",
                "St. Kitts and Nevis": "KNA",
                "St. Lucia": "LCA",
                "St. Vincent and the Grenadines": "VCT",
                "West Bank and Gaza": "PSE",
                "Brunei Darussalam": "BRN",
                "Moldova": "MDA",
            }
            try:
                import pycountry
                def to_iso3(n):
                    if n in WDI_NAME_OVERRIDES:
                        return WDI_NAME_OVERRIDES[n]
                    try:
                        m = pycountry.countries.lookup(n)
                        return m.alpha_3
                    except LookupError:
                        return None
                wdi = wdi.copy()
                src_col = "country_name" if "country_name" in wdi else "country"
                wdi["country_iso3"] = wdi[src_col].map(to_iso3)
                wdi_key = "country_iso3"
            except ImportError:
                wdi_key = None

        if wdi_key is not None:
            partner_wdi = wdi[[wdi_key, "year"] + wdi_cols].rename(columns={wdi_key: "iso3"})
            p = p.merge(partner_wdi, on=["iso3", "year"], how="left", suffixes=("", "_partner"))
            # Kosovo macro: merge by year only
            ko = wdi[wdi[wdi_key] == "XKX"][["year"] + wdi_cols].rename(
                columns={c: c + "_kosovo" for c in wdi_cols})
            p = p.merge(ko, on="year", how="left")

    if cepii_resolved is not None and len(cepii_resolved):
        p = p.merge(cepii_resolved, on="iso2", how="left")

    return p


# =============================================================================
# 6. Phase 1 report printer
# =============================================================================

def phase1_report(
    panel: pd.DataFrame,
    partners: pd.DataFrame,
    wdi: pd.DataFrame | None,
    cepii_resolved: pd.DataFrame | None,
    eurostat_status: str,
) -> dict:
    """Compute and print every Phase 1 diagnostic. Also save to JSON."""
    report = {}

    # Sample size + balance
    report["n"] = int(len(panel))
    report["n_partners"] = int(panel["iso2"].nunique())
    report["n_years"] = int(panel["year"].nunique())
    report["year_min"] = int(panel["year"].min())
    report["year_max"] = int(panel["year"].max())
    report["balanced"] = bool(
        report["n_partners"] * report["n_years"] == report["n"]
    )

    # Zero count
    report["zero_rows"] = int((panel["imports_eur_thousands"] == 0).sum())
    report["zero_rows_pct"] = round(report["zero_rows"] / report["n"] * 100, 2)

    # Serbia trajectory
    xs = panel[panel["iso2"] == "XS"].sort_values("year")
    report["serbia_trajectory"] = {int(r.year): float(r.imports_eur_thousands)
                                   for r in xs.itertuples()}
    if 2017 in report["serbia_trajectory"] and 2019 in report["serbia_trajectory"]:
        v17 = report["serbia_trajectory"][2017]
        v19 = report["serbia_trajectory"][2019]
        report["serbia_collapse_2017_to_2019_pct"] = round((1 - v19 / v17) * 100, 2) if v17 else None

    # WDI join
    if wdi is not None and "gdp_usd_current" in panel.columns:
        j = panel["gdp_usd_current"].notna().mean()
        report["wdi_join_success_pct"] = round(j * 100, 2)
    else:
        report["wdi_join_success_pct"] = None

    # CEPII join
    if cepii_resolved is not None and "dist_km" in panel.columns:
        j = panel["dist_km"].notna().mean()
        report["cepii_join_success_pct"] = round(j * 100, 2)
        sources = panel["distance_source"].value_counts().to_dict() if "distance_source" in panel else {}
        report["cepii_distance_sources"] = sources
    else:
        report["cepii_join_success_pct"] = None

    # Eurostat
    report["eurostat_status"] = eurostat_status

    # Anomalies
    anomalies = []
    if report["n"] != cfg.EXPECTED_N:
        anomalies.append(f"sample n={report['n']} != expected {cfg.EXPECTED_N}")
    if not report["balanced"]:
        anomalies.append("panel is not balanced (partners × years mismatch)")
    if partners["iso3"].isna().any():
        unmapped = partners[partners["iso3"].isna()]["iso2"].tolist()
        anomalies.append(f"unmapped iso2 in partner master: {unmapped}")
    report["anomalies"] = anomalies

    # Print
    print("\n" + "=" * 70)
    print("PHASE 1 REPORT")
    print("=" * 70)
    print(f"sample n .................... {report['n']}  (expected {cfg.EXPECTED_N})")
    print(f"partners × years ............ {report['n_partners']} × {report['n_years']}  balanced={report['balanced']}")
    print(f"zero rows ................... {report['zero_rows']}  ({report['zero_rows_pct']}%)")
    print(f"serbia 2017→2019 collapse ... {report.get('serbia_collapse_2017_to_2019_pct')}%")
    print("serbia trajectory (key years):")
    for y in (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024):
        v = report["serbia_trajectory"].get(y)
        if v is not None:
            print(f"  {y}: {v:>12,.0f} EUR thousands")
    print(f"WDI join success % .......... {report['wdi_join_success_pct']}")
    print(f"CEPII join success % ........ {report['cepii_join_success_pct']}")
    print(f"Eurostat decision ........... {report['eurostat_status']}")
    print(f"Anomalies ................... {report['anomalies'] or 'none'}")
    print("=" * 70 + "\n")

    # Save JSON
    out = cfg.METRICS / "metric_ch3_phase1.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report
