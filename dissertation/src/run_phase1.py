"""Phase 1 orchestrator.

Runs the data pipeline end-to-end and prints the Phase 1 report.

Usage (from repo root):
    python -m src.run_phase1

Network-dependent steps (WDI, CEPII, Gap Institute) will execute if their
caches are missing. If the host cannot reach those URLs, the steps raise and
the report marks them as FAILED / BLOCKED. Era can supply cache files
manually by placing them in data/raw/ with the filenames documented in
data_pipeline.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from . import config as cfg
from . import data_pipeline as dp
from . import features as fx
from . import ppml


def main():
    print("\n[Phase 1] step 1/8 — parse tab02")
    tab02 = dp.parse_tab02()
    panel_all_partners, activity = dp.apply_top_n_filter(tab02, n=cfg.TOP_N_PARTNERS)
    top_panel = panel_all_partners
    print(f"  tab02 raw: {len(tab02)} rows, {tab02['iso2'].nunique()} partners")
    print(f"  top-{cfg.TOP_N_PARTNERS} filtered: {len(top_panel)} rows "
          f"({top_panel['iso2'].nunique()} partners × {top_panel['year'].nunique()} years)")

    print("\n[Phase 1] step 2/8 — parse tab04 (sector panel)")
    sector = dp.parse_tab04()
    print(f"  tab04: {len(sector)} rows "
          f"({sector['hs_section'].nunique()} sections × {sector['year'].nunique()} years)")
    sector.to_parquet(cfg.DATA_PROCESSED / "panel_sector.parquet", index=False)

    print("\n[Phase 1] step 3/8 — partner master")
    partners = dp.build_partner_master(top_panel)
    partners.to_csv(cfg.DATA_RAW / "partner_master.csv", index=False)
    # Note: iso3 / partner_id are attached to the panel inside
    # dp.build_merged_panel — do not pre-merge here or pandas will create
    # iso3_x / iso3_y and break the downstream WDI join on iso3.

    print("\n[Phase 1] step 4/8 — WDI macro")
    try:
        wdi = dp.fetch_wdi(partners["iso3"].dropna().tolist())
    except Exception as e:
        print(f"  WDI FAILED: {e}")
        wdi = None

    print("\n[Phase 1] step 5/8 — CEPII GeoDist")
    try:
        cepii_raw = dp.fetch_cepii()
        cepii_resolved = dp.resolve_kosovo_distances(partners, cepii_raw)
    except Exception as e:
        print(f"  CEPII FAILED: {e}")
        cepii_resolved = None

    print("\n[Phase 1] step 6/8 — Eurostat COMEXT HS2 (optional)")
    eurostat_decision_file = cfg.DATA_RAW / "EUROSTAT_DECISION.md"
    if eurostat_decision_file.exists():
        eurostat_status = eurostat_decision_file.read_text().strip().splitlines()[0]
    else:
        eurostat_status = "not yet attempted"

    print("\n[Phase 1] step 7/8 — Gap Institute curated CSV")
    try:
        dp.curate_gap_institute_csv()
        print(f"  wrote data/raw/gap_institute_diversion.csv "
              f"(reconcile with PDF once downloaded)")
    except Exception as e:
        print(f"  Gap Institute CSV seeding failed: {e}")

    print("\n[Phase 1] step 8/8 — merge panel + engineer features")
    merged = dp.build_merged_panel(top_panel, partners, wdi, cepii_resolved)
    merged = fx.engineer_all(merged)
    merged.to_parquet(cfg.DATA_PROCESSED / "panel_bilateral.parquet", index=False)
    coverage = fx.feature_coverage_report(merged)
    print(f"  merged panel: {len(merged)} rows, {merged.shape[1]} columns")
    print(f"  feature coverage (% non-null):")
    for f, pct in coverage.items():
        print(f"    {f:32s} {pct:6.2f}%")

    # --- Report
    report = dp.phase1_report(
        panel=merged,
        partners=partners,
        wdi=wdi,
        cepii_resolved=cepii_resolved,
        eurostat_status=eurostat_status,
    )

    # --- Fit both PPMLs (Phase 1 non-negotiables, if features available)
    print("\n[Phase 1] fit PPML-Predictive")
    try:
        ppml_pred = ppml.fit_ppml_predictive(merged)
        print(f"  {ppml_pred.spec_name}: n={ppml_pred.n}, partner FE={ppml_pred.n_partner_fe}, "
              f"year FE={ppml_pred.n_year_fe}, zeros={ppml_pred.zero_rows}, clustered={ppml_pred.clustered}")
    except Exception as e:
        print(f"  PPML-Predictive FAILED: {e}")

    print("\n[Phase 1] fit PPML-DiD (primary: treatment={2019})")
    try:
        ppml_did = ppml.fit_ppml_did(merged, treatment_years=cfg.DID_TREATMENT_YEARS_PRIMARY)
        print(f"  {ppml_did.spec_name}: n={ppml_did.n}, partner FE={ppml_did.n_partner_fe}, "
              f"year FE={ppml_did.n_year_fe}, zeros={ppml_did.zero_rows}")
        # print β_DiD
        res = ppml_did.model_result
        import numpy as np
        try:
            names = list(res.model.exog_names)
            idx = names.index("serbia_x_post")
            b = float(np.asarray(res.params)[idx])
            se = float(np.asarray(res.bse)[idx])
            print(f"  β_DiD (serbia_x_post) = {b:.4f}  (SE={se:.4f})")
        except Exception as e:
            print(f"  (could not extract β_DiD: {e})")
    except Exception as e:
        print(f"  PPML-DiD FAILED: {e}")

    print("\n[Phase 1] complete. Awaiting sign-off before Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
