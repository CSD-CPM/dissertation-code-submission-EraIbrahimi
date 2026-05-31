"""Phase 1.5 orchestrator - export-side spillover DiD.

Mirrors the structure of run_phase4.py. Single causal estimate, no ML,
no scenarios. Writes 1 figure (PNG+PDF), 1 table (CSV+TEX), 3 metric
JSONs, 1 raw-draws npy. Does not modify any Phase 1-5 module.

Run from dissertation/:
    python -m src.run_phase1_5
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import config as cfg
from . import did_export


def main() -> int:
    print("[phase1.5] Loading 4-partner DiD panel with both flows ...")
    panel = did_export.load_did_export_panel()
    assert len(panel) == 60, f"panel rows: {len(panel)} != 60"
    assert set(panel.iso2.unique()) == {"XS", "AL", "MK", "ME"}
    assert panel.exports_eur_thousands.notna().all()
    print(f"[phase1.5] Panel OK: n={len(panel)}, partners={sorted(panel.iso2.unique())}")
    print(f"[phase1.5] Total exports across panel: "
          f"{panel.exports_eur_thousands.sum():,.0f} EUR thousand")
    print(f"[phase1.5] Total imports across panel: "
          f"{panel.imports_eur_thousands.sum():,.0f} EUR thousand")

    # ------------------------------------------------------------------
    print("\n[phase1.5] Smoke test (40-draw bootstrap) ...")
    smoke = did_export.smoke_test_bootstrap_export(panel, n_boot=40, seed=42)
    print(f"[phase1.5]   beta_main         = {smoke['beta_main']:.4f}")
    print(f"[phase1.5]   bootstrap_mean    = {smoke['bootstrap_mean']:.4f}")
    print(f"[phase1.5]   n_failed/n_total  = {smoke['n_failed']}/40 "
          f"({smoke['fail_rate']*100:.1f}%)")
    print(f"[phase1.5]   same_sign         = {smoke['same_sign']}")
    print(f"[phase1.5]   in_magnitude_band = {smoke['in_magnitude_band']}")
    print(f"[phase1.5]   pass_flag         = {smoke['pass_flag']}")
    if not smoke["pass_flag"]:
        raise RuntimeError(
            f"Smoke test failed: {smoke}. "
            "Stopping before the canonical N=1000 run. "
            "Investigate before retrying."
        )

    # ------------------------------------------------------------------
    print("\n[phase1.5] Main fit + N=1000 bootstrap ...")
    result = did_export.main_export_did_with_bootstrap(panel, n_boot=1000, seed=42)
    print(f"[phase1.5]   beta_export        = {result['beta_export']:.4f}")
    print(f"[phase1.5]   bootstrap_mean     = {result['bootstrap_mean']:.4f}")
    print(f"[phase1.5]   bootstrap_ci_95    = "
          f"[{result['bootstrap_ci_low']:.4f}, {result['bootstrap_ci_high']:.4f}]")
    print(f"[phase1.5]   exp(beta_export)-1 = {result['exp_beta_minus_1_pct']:.2f}%")
    print(f"[phase1.5]   exp(CI)-1          = "
          f"[{result['exp_ci_low_pct']:.2f}%, {result['exp_ci_high_pct']:.2f}%]")
    print(f"[phase1.5]   n_completed/n_boot = "
          f"{result['n_boot_completed']}/{result['n_boot']} "
          f"(failed: {result['n_boot_failed']})")
    print(f"[phase1.5]   wall_time          = {result['wall_time_seconds']:.1f}s")

    if result["n_boot_completed"] < 950:
        raise RuntimeError(
            f"Bootstrap completion rate too low: "
            f"{result['n_boot_completed']}/1000 (< 950). Stopping."
        )
    for k in ("bootstrap_ci_low", "bootstrap_ci_high"):
        if not np.isfinite(result[k]):
            raise RuntimeError(f"CI endpoint {k} is not finite: {result[k]}")

    draws = result.pop("draws")
    np.save(cfg.METRICS / "bootstrap_draws_export.npy", draws)
    with open(cfg.METRICS / "bootstrap_ci_export.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[phase1.5] Saved bootstrap_ci_export.json + bootstrap_draws_export.npy")

    # ------------------------------------------------------------------
    print("\n[phase1.5] Building import-vs-export comparison ...")
    with open(cfg.METRICS / "bootstrap_ci.json") as f:
        import_ci = json.load(f)
    comparison = did_export.import_vs_export_comparison(import_ci, result)
    with open(cfg.METRICS / "import_vs_export_did_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[phase1.5]   beta_import           = {comparison['beta_import']:.4f}")
    print(f"[phase1.5]   beta_export           = {comparison['beta_export']:.4f}")
    print(f"[phase1.5]   ratio                 = {comparison['ratio']:.4f}")
    print(f"[phase1.5]   delta_exp_minus_1_pct = "
          f"{comparison['delta_exp_minus_1_pct']:+.2f}pp")
    print(f"[phase1.5]   cis_overlap           = {comparison['cis_overlap']}  "
          f"(descriptive heuristic only)")

    # ------------------------------------------------------------------
    print("\n[phase1.5] Rendering asymmetry figure and summary table ...")
    did_export.build_asymmetry_figure(panel, comparison)
    did_export.build_summary_table(import_ci, result)
    print("[phase1.5] Wrote fig_ch5_import_vs_export_asymmetry.{png,pdf}")
    print("[phase1.5] Wrote tbl_ch5_export_spillover_summary.{csv,tex}")

    print("\n[phase1.5] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
