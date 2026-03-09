"""
Backtest the Snoqualmie Pass forecast models (leave-one-year-out).

Run after building the merged dataset and patching:
  python fetch_data.py
  python build_merged_dataset.py
  python forecast.py   # optional: builds features; or we do it here

Then:
  python backtest.py

Tries several configurations (full ensemble, Ridge-only, recent years) and reports
RMSE, skill vs climatology, correlation, and bias. Use results to decide if the
model is acceptable or what to tweak.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Run from project root
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
os.chdir(BASE)

from forecast import (
    load_base,
    patch_fresh_telecons,
    patch_fresh_snotel,
    patch_historical_snowfall,
    apply_sno_pass_first,
    patch_ndbc_buoy,
    patch_synoptic_monthly,
    patch_slp_nepac,
    patch_hgt500_gradient,
    patch_nino12_tni,
    build_features,
    run_backtest,
    WINTER_MONTHS,
)


def load_full_df():
    """Load and patch dataframe (same as forecast.main up to build_features)."""
    df = load_base()
    df = patch_fresh_telecons(df)
    df = patch_fresh_snotel(df)
    df = patch_historical_snowfall(df)
    df = apply_sno_pass_first(df)
    df = patch_ndbc_buoy(df)
    df = patch_synoptic_monthly(df)
    df = patch_slp_nepac(df)
    df = patch_hgt500_gradient(df)
    df = patch_nino12_tni(df)
    df = build_features(df)
    return df


def main():
    print("Loading data (same pipeline as forecast.py) ...")
    df = load_full_df()
    print(f"   {len(df)} rows, winter months with WTEQ: {df[df['month'].isin(WINTER_MONTHS)]['WTEQ'].notna().sum()}\n")

    configs = [
        ("Full ensemble (all models)", None),
        ("Ridge only", ["Ridge"]),
        ("Ridge + RF + GBR", ["Ridge", "RF", "GBR"]),
    ]

    all_metrics = {}
    bt_wteq_full = None
    for label, model_names in configs:
        print(f"--- {label} ---")
        bt_wteq = run_backtest(df, "WTEQ", model_names=model_names, verbose=True)
        if label == "Full ensemble (all models)":
            bt_wteq_full = bt_wteq
        all_metrics[label] = {"WTEQ": {k: v for k, v in bt_wteq.items() if k != "results"}}
        print()

    # Summary table
    print("=" * 70)
    print("BACKTEST SUMMARY (leave-one-year-out) — SWE only")
    print("=" * 70)
    print(f"{'Config':<30} {'RMSE':>8} {'RMSE_clim':>10} {'Skill':>8} {'Corr':>7} {'Bias':>8}")
    print("-" * 70)
    for label, metrics in all_metrics.items():
        m = metrics["WTEQ"]
        rmse = m.get("rmse", np.nan)
        rmse_c = m.get("rmse_clim", np.nan)
        skill = m.get("skill", np.nan)
        corr = m.get("correlation", np.nan)
        bias = m.get("bias", np.nan)
        skill_str = f"{skill:>6.1%}" if not np.isnan(skill) else "  N/A"
        print(f"{label:<30} {rmse:>8.2f} {rmse_c:>10.2f} {skill_str:>8} {corr:>7.3f} {bias:>8.2f}")
    print()

    # Save detailed results for full ensemble
    out_dir = os.path.join(BASE, "data")
    os.makedirs(out_dir, exist_ok=True)

    for target, bt in [("WTEQ", bt_wteq_full)]:
        if bt is None:
            continue
        res = bt.get("results", [])
        if res:
            recs = [{"year": r["year"], "month": r["month"], "actual": r["actual"], "pred": r["pred"], "clim_pred": r["clim_pred"]} for r in res]
            pd.DataFrame(recs).to_csv(os.path.join(out_dir, f"backtest_{target}.csv"), index=False)
            print(f"   Saved data/backtest_{target}.csv ({len(recs)} points)")

    metrics_save = {}
    for label, metrics in all_metrics.items():
        metrics_save[label] = {
            "WTEQ": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in metrics["WTEQ"].items()}
        }
    with open(os.path.join(out_dir, "backtest_metrics.json"), "w") as f:
        json.dump(metrics_save, f, indent=2)
    print("   Saved data/backtest_metrics.json")

    best = all_metrics["Full ensemble (all models)"]
    wteq_skill = best["WTEQ"].get("skill", np.nan)
    print()
    if not np.isnan(wteq_skill) and wteq_skill > 0:
        print("WTEQ: Model beats climatology (skill > 0).")
    else:
        print("WTEQ: Model does not beat climatology. Try Ridge-only or core telecons (see docs/BACKTEST.md).")
    print("\nDone. Pipeline is SWE-only; snowfall target removed.")


if __name__ == "__main__":
    main()
