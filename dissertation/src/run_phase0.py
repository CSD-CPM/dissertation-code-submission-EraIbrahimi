"""Phase 0 — Trade-flow EDA orchestrator.

Produces the fixed-core descriptive deliverables (tables, figures, metric JSON)
for the dissertation's data chapter. Read-only on Phase 1–5 modules.

Run from `dissertation/`:
    python -m src.run_phase0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import eda


def main() -> int:
    print("="*70)
    print("PHASE 0 — Trade-flow EDA (fixed core)")
    print("="*70)

    print("\n[Phase 0] step 1/10 — parse ASK files (4 yearly + 2 monthly)")
    partner = eda.parse_partner_full()
    print(f"  yearly partner: {len(partner):,} rows, "
          f"{partner['iso2'].nunique()} unique iso2, years "
          f"{partner['year'].min()}–{partner['year'].max()}")

    sections = eda.parse_sections_full()
    print(f"  yearly sections: {len(sections):,} rows, "
          f"{sections['hs_section'].nunique()} sections")

    chapter = eda.parse_chapter_full()
    print(f"  yearly chapter: {len(chapter):,} rows, "
          f"{chapter['hs_chapter'].nunique()} chapters")

    bec = eda.parse_bec_full()
    print(f"  yearly BEC: {len(bec):,} rows, "
          f"{bec['bec_category'].nunique()} categories")

    monthly_imp, anom_imp = eda.parse_monthly_partner(flow="import")
    print(f"  monthly imports: {len(monthly_imp):,} rows, "
          f"anomalies: {len(anom_imp)}")

    monthly_exp, anom_exp = eda.parse_monthly_partner(flow="export")
    print(f"  monthly exports: {len(monthly_exp):,} rows, "
          f"anomalies: {len(anom_exp)}")

    all_anomalies = anom_imp + anom_exp
    if all_anomalies:
        print("  ANOMALIES (will be surfaced in report §10):")
        for a in all_anomalies[:5]:
            print(f"    {a}")
        if len(all_anomalies) > 5:
            print(f"    ... and {len(all_anomalies)-5} more")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 2/10 — HARD reconciliation (partner vs sections)")
    try:
        rec_hard = eda.reconcile_partner_vs_sections(partner, sections)
        max_dev = rec_hard["pct_delta"].abs().max()
        print(f"  HARD recon PASS — max |pct_delta| = {max_dev:.5f}% (threshold 0.05%)")
    except RuntimeError as exc:
        print(f"  HARD recon FAILED — stopping.\n{exc}")
        return 1

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 3/10 — diagnostic reconciliations (chapter, BEC, monthly↔yearly)")
    rec_chapbec = eda.reconcile_partner_vs_chapter_and_bec(partner, chapter, bec)
    rec_monthly = eda.reconcile_monthly_vs_yearly(monthly_imp, monthly_exp, partner)
    n_chap_pass = (rec_chapbec["chapter_status"] == "pass").sum()
    n_bec_pass  = (rec_chapbec["bec_status"] == "pass").sum()
    n_mo_pass   = (rec_monthly["status"] == "pass").sum()
    print(f"  chapter status: {n_chap_pass}/{len(rec_chapbec)} pass")
    print(f"  BEC status:     {n_bec_pass}/{len(rec_chapbec)} pass")
    print(f"  monthly status: {n_mo_pass}/{len(rec_monthly)} pass")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 4/10 — annual totals + zero counts")
    annual = eda.annual_totals_by_flow(partner)
    zeros  = eda.zero_count_by_flow(partner)
    last = annual.iloc[-1]
    print(f"  2024: imports={last['imports']:,.0f}k  exports={last['exports']:,.0f}k  "
          f"deficit={last['deficit']:,.0f}k  ratio={last['export_to_import_ratio']:.3f}")
    zr_2024 = zeros[zeros["year"]==2024]
    for _, r in zr_2024.iterrows():
        print(f"  2024 {r['flow']:6s} zero rate: {r['zero_rate_pct']:.2f}% ({r['n_zero']}/{r['n_total']})")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 5/10 — top partners + concentration")
    top20 = eda.top_partners_by_flow(partner, n=20, year=2024)
    print(f"  top-20 partners 2024: {len(top20)} rows (union of import/export top-20)")
    conc = eda.concentration_curves(partner)
    for _, r in conc.iterrows():
        print(f"  {r['flow']:6s} {r['scope']:14s} top5={r['top_5_share_pct']:.1f}% "
              f"top20={r['top_20_share_pct']:.1f}% HHI={r['hhi']:.1f}")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 6/10 — sector composition + evolution heatmap")
    sec_comp = eda.sector_composition_by_flow(sections)
    sec_evo  = eda.sector_evolution_heatmap(sections)
    print(f"  sector composition: {len(sec_comp)} rows")
    print(f"  sector evolution: {len(sec_evo)} rows")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 7/10 — Serbia trajectory + monthly event study")
    serbia = eda.serbia_trajectory_both_flows(partner)
    print(f"  Serbia trajectory: {len(serbia)} rows (expected 120 = 4 × 2 × 15)")
    event = eda.serbia_event_study_monthly(monthly_imp, monthly_exp)
    print(f"  monthly event study: {len(event)} rows")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 8/10 — panel-construction proposal")
    panel_df, panel_summary = eda.panel_construction_proposal(partner)
    print(f"  imports-top vs exports-top: intersection={panel_summary['intersection_count']}, "
          f"union={panel_summary['union_count']}, "
          f"imports-only={len(panel_summary['in_imports_only'])}, "
          f"exports-only={len(panel_summary['in_exports_only'])}")
    print(f"  K_imp(90%)={panel_summary['K_imp_90pct']}, "
          f"K_exp(90%)={panel_summary['K_exp_90pct']}, "
          f"proposed unified panel size={panel_summary['proposed_panel_size']}")
    print(f"  imports-derived panel covers {panel_summary['imports_panel_exports_coverage_pct']:.1f}% of cumulative exports")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 9/10 — export modellability assessment")
    mod = eda.export_modellability_assessment(partner)
    print(f"  Phase-1 panel: {mod['phase1_panel_iso2_count']} partners, "
          f"{mod['effective_n_export_rows']} export rows")
    print(f"  export zero rate: {mod['export_zero_rate']*100:.2f}%  "
          f"(vs import zero rate: {mod['import_zero_rate']*100:.2f}%)")
    print(f"  var(log1p exports): {mod['var_log1p_exports']:.3f}  "
          f"var(log1p imports): {mod['var_log1p_imports']:.3f}  "
          f"ratio: {mod['var_log_ratio_exp_to_imp']:.3f}")
    print(f"  partners with >5 non-zero export years: {mod['partners_with_gt5_nonzero_export_years']}/"
          f"{mod['phase1_panel_iso2_count']}")
    print(f"  structural zero partners (0 non-zero export years): {mod['partners_with_zero_export_years']}")

    # ------------------------------------------------------------------
    print("\n[Phase 0] step 10/10 — summary block")
    print("="*70)
    print("FIXED-CORE HEADLINE NUMBERS")
    print("="*70)
    print(f"\nAnnual totals (€ thousands):")
    for _, r in annual.iterrows():
        print(f"  {int(r['year'])}: imports={r['imports']:>12,.0f}  "
              f"exports={r['exports']:>10,.0f}  deficit={r['deficit']:>12,.0f}  "
              f"x/m={r['export_to_import_ratio']:.3f}")

    print(f"\nZero rates (2024): "
          f"imports={zr_2024[zr_2024['flow']=='import']['zero_rate_pct'].iloc[0]:.2f}%  "
          f"exports={zr_2024[zr_2024['flow']=='export']['zero_rate_pct'].iloc[0]:.2f}%")

    print(f"\nReconciliation summary:")
    print(f"  HARD (partner↔sections): PASS, max |Δ| = {rec_hard['pct_delta'].abs().max():.5f}%")
    print(f"  Chapter: {n_chap_pass}/{len(rec_chapbec)} cells within 0.05%; max |Δ| = "
          f"{rec_chapbec['chapter_pct_delta'].abs().max():.3f}%")
    print(f"  BEC:     {n_bec_pass}/{len(rec_chapbec)} cells within 0.05%; max |Δ| = "
          f"{rec_chapbec['bec_pct_delta'].abs().max():.3f}%")
    print(f"  Monthly↔yearly: {n_mo_pass}/{len(rec_monthly)} cells within 0.05%; max |Δ| = "
          f"{rec_monthly['pct_delta'].abs().max():.3f}%")

    if all_anomalies:
        print(f"\nParser anomalies (n={len(all_anomalies)}): see report §10. First entries:")
        for a in all_anomalies[:3]:
            print(f"  {a}")

    # Save anomalies + summary for the report writer
    state = {
        "anomalies": all_anomalies,
        "panel_summary": panel_summary,
        "export_modellability": mod,
        "reconciliation": {
            "hard_max_abs_pct": float(rec_hard["pct_delta"].abs().max()),
            "chapter_max_abs_pct": float(rec_chapbec["chapter_pct_delta"].abs().max()),
            "bec_max_abs_pct":     float(rec_chapbec["bec_pct_delta"].abs().max()),
            "monthly_max_abs_pct": float(rec_monthly["pct_delta"].abs().max()),
        },
    }
    state_path = Path("/tmp/phase0_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\n[Phase 0] state snapshot → {state_path}")

    print("\n[Phase 0] fixed core complete; running adaptive layer.")

    # ------------------------------------------------------------------
    print("\n[Phase 0] adaptive 1/3 — partner asymmetry (imports vs exports)")
    asy = eda.eda_partner_asymmetry(partner)
    bal = (asy["classification"] == "balanced").sum()
    imp_only = (asy["classification"] == "imports_only").sum()
    exp_only = (asy["classification"] == "exports_only").sum()
    imp_dom = (asy["classification"] == "imports_dominant").sum()
    exp_dom = (asy["classification"] == "exports_dominant").sum()
    print(f"  partners: imports_only={imp_only}, imports_dominant={imp_dom}, "
          f"balanced={bal}, exports_dominant={exp_dom}, exports_only={exp_only}")

    print("\n[Phase 0] adaptive 2/3 — Serbia monthly breakpoint precision")
    bp = eda.eda_monthly_breakpoint(monthly_imp)
    print(f"  onset/recovery candidates flagged: {len(bp)}")
    for _, r in bp.iterrows():
        print(f"  {r['window']:8s}  {r['date']}: value={r['value_eur_thousands']/1000:>7.1f}M  "
              f"MoM={r['mom_pct']:+.1f}%  z={r['mom_z']:+.2f}")

    print("\n[Phase 0] adaptive 3/3 — sector event response (Option A)")
    ser = eda.eda_sector_event_response(sections)
    above_2 = ser[ser["z_score"].abs() >= 2.0]
    print(f"  (section, flow, event) triples with |z| >= 2: {len(above_2)}/{len(ser)}")
    print(f"  top-5 most reactive movements:")
    for _, r in ser.head(5).iterrows():
        print(f"    [{r['flow'][:3]}] {r['hs_section'][:48]:48s} {r['event_year']} "
              f"Δ={r['delta_pp']:+.2f}pp z={r['z_score']:+.2f}")

    print("\n[Phase 0] bonus — chapter file 2018 anomaly check")
    anom = eda.eda_chapter_anomaly_check(chapter, sections, partner)
    print(f"  anomaly confined to 2018? {anom['anomaly_confined_to_2018']}")
    print(f"  chapter/partner ratio by year (should be ~1.0 except 2018):")
    for y, ratio in sorted(anom["chapter_vs_partner_ratio_by_year"].items()):
        flag = "  <-- ANOMALY" if abs(ratio - 1) > 0.05 else ""
        print(f"    {y}: {ratio:.4f}{flag}")
    print(f"  top chapters with 2018 z > 3:")
    for ch, z in list(anom["top_8_chapters_with_2018_z_score_above_3"].items())[:5]:
        print(f"    z={z:5.1f}  {ch}")

    state["adaptive_partner_asymmetry"] = {
        "classification_counts": asy["classification"].value_counts().to_dict(),
        "top10_by_total": asy.head(10)[["iso2","partner_name","cum_imports","cum_exports","imbalance_ratio","classification"]].to_dict(orient="records"),
    }
    state["adaptive_monthly_breakpoint"] = bp.to_dict(orient="records")
    state["adaptive_sector_event"] = {
        "n_above_z2": int(len(above_2)),
        "top10": ser.head(10).to_dict(orient="records"),
    }
    state["bonus_chapter_anomaly"] = anom

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"\n[Phase 0] adaptive layer complete; full state at {state_path}")
    print("[Phase 0] ready for PHASE0_EDA_REPORT.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
