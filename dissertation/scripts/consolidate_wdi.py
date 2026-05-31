"""Consolidate the five World Bank bulk indicator ZIPs into wdi_macro.csv.

Use this ONLY if the World Bank API is unreachable from your machine
(§ B.2 of DOWNLOAD_CHECKLIST.md). The preferred path is § B.1 — let
`data_pipeline.fetch_wdi` pull directly via `wbdata`.

Expected input: `data/raw/wdi_bulk/` contains 5 ZIPs downloaded from
data.worldbank.org (any filename, each ZIP must contain one CSV with the
standard WDI wide layout: country_code, year columns).

Output: `data/raw/wdi_macro.csv` with columns:
    country_name, year, gdp_usd_current, gdp_growth_pct,
    population, fdi_pct_gdp, inflation_pct

Run from the `dissertation/` directory:
    python -m scripts.consolidate_wdi
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from src import config as cfg

BULK_DIR = cfg.DATA_RAW / "wdi_bulk"
OUT_CSV = cfg.DATA_RAW / "wdi_macro.csv"

# Map indicator code → column name used throughout the pipeline.
INDICATORS = cfg.WDI_INDICATORS


def _read_bulk_csv(zip_path: Path) -> pd.DataFrame:
    """Return the long-format dataframe from one World Bank bulk ZIP.

    WB bulk files are wide (year columns) and often have 4 metadata rows
    at the top. We locate the CSV whose name starts with 'API_'.
    """
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist()
                      if n.lower().endswith(".csv") and "metadata" not in n.lower()]
        if not candidates:
            raise RuntimeError(f"No data CSV found in {zip_path.name}")
        name = candidates[0]
        with z.open(name) as fp:
            raw = fp.read().decode("utf-8", errors="replace")
    # Skip the 4 metadata rows at the top.
    df = pd.read_csv(io.StringIO(raw), skiprows=4)
    # Find indicator code to retain only that slice (bulk file is one indicator
    # but double-check).
    if "Indicator Code" in df.columns:
        codes = df["Indicator Code"].dropna().unique()
        if len(codes) != 1:
            raise RuntimeError(f"{zip_path.name}: expected one indicator, got {codes}")
        code = codes[0]
    else:
        raise RuntimeError(f"{zip_path.name}: missing 'Indicator Code' column")
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="year", value_name="value",
    )
    long["year"] = long["year"].astype(int)
    long = long.rename(columns={"Country Name": "country_name",
                                 "Country Code": "country_iso3"})
    col = INDICATORS.get(code, code)
    long = long.rename(columns={"value": col})
    long = long[["country_name", "country_iso3", "year", col]]
    return long


def main() -> int:
    if not BULK_DIR.exists():
        raise SystemExit(
            f"{BULK_DIR} does not exist. Create it and place the 5 World Bank "
            f"indicator ZIPs inside, then re-run.")

    zips = sorted(BULK_DIR.glob("*.zip"))
    if len(zips) < 5:
        print(f"[wdi] WARNING: expected 5 ZIPs, found {len(zips)}. Will consolidate what is present.")

    frames = []
    for zp in zips:
        print(f"[wdi] reading {zp.name} ...")
        frames.append(_read_bulk_csv(zp))

    # Merge all indicator frames on (country, year).
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["country_name", "country_iso3", "year"], how="outer")

    # Restrict to 2009–2024 (the pipeline's panel window).
    out = out[out["year"].between(2009, 2024)].copy()
    out = out.sort_values(["country_iso3", "year"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[wdi] wrote {OUT_CSV}: {len(out)} rows, {out['country_iso3'].nunique()} countries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
