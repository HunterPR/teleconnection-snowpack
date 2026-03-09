"""
tune_backtest.py
================
Grid search over model configs, feature subsets, climatology blend, and hyperparameters
to find the best-performing setup for Snoqualmie Pass SWE backtest.

Usage:
  python tune_backtest.py           # Full grid (may take 5–15 min)
  python tune_backtest.py --quick   # Smaller subset for faster iteration
  python tune_backtest.py --top 20  # Show top 20 configs
"""

import os
import argparse
import itertools
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)


def load_data():
    """Load and prepare data (same as forecast.py main path)."""
    from forecast import (
        load_base, patch_fresh_telecons, patch_fresh_snotel, patch_historical_snowfall,
        apply_sno_pass_first, patch_ndbc_buoy, patch_synoptic_monthly, patch_slp_nepac,
        patch_hgt500_gradient, patch_nino12_tni, build_features, run_backtest,
        WINTER_MONTHS, CORE_TELE,
    )

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


def build_grid(quick: bool = False):
    """
    Build grid of (label, model_names, tele_subset, clim_blend, pipeline_overrides).

    quick: if True, use a smaller subset for faster iteration.
    """
    # Teleconnection subsets (cast wide net)
    CORE_4 = ["ao", "roni", "pdo", "pna"]
    CORE_6 = ["ao", "roni", "pdo", "pna", "np", "epo"]
    CORE_8 = ["ao", "roni", "pdo", "pna", "np", "epo", "z500_nepac_anom"]
    CORE_10 = ["ao", "roni", "pdo", "pna", "np", "epo", "z500_nepac_anom", "amo", "nao"]

    tele_configs = [
        ("core4", CORE_4, "ao,roni,pdo,pna"),
        ("core6", CORE_6, "core4 + np,epo"),
        ("core8", CORE_8, "core6 + z500_nepac_anom"),
        ("core10", CORE_10, "core8 + amo,nao"),
        ("all", None, "all CORE_TELE"),
    ]

    # Model combos (cast wide net: linear, tree, and blends)
    model_configs = [
        ("Ridge", ["Ridge"]),
        ("BayesRidge", ["BayesRidge"]),
        ("ElasticNet", ["ElasticNet"]),
        ("Ridge+BayesRidge", ["Ridge", "BayesRidge"]),
        ("Ridge+RF", ["Ridge", "RF"]),
        ("Ridge+GBR", ["Ridge", "GBR"]),
        ("Ridge+KNN", ["Ridge", "KNN"]),
        ("Ridge+RF+GBR", ["Ridge", "RF", "GBR"]),
        ("Ridge+RF+GBR+XGB", ["Ridge", "RF", "GBR", "XGBoost"]),
        ("ensemble", None),  # full ensemble
    ]

    # Climatology blend (shrink toward mean)
    clim_values = [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    if quick:
        clim_values = [0.0, 0.25, 0.5]

    # Ridge alpha (when Ridge is used)
    ridge_alphas = [10, 50, 100, 200, 500]
    if quick:
        ridge_alphas = [50, 100, 200]

    configs = []
    for (tname, tcols, tdesc), (mname, mlist) in itertools.product(tele_configs, model_configs):
        for clim in clim_values:
            # For Ridge-only, sweep alpha
            if mlist and "Ridge" in mlist and mname == "Ridge":
                for alpha in ridge_alphas:
                    overrides = {"Ridge": {"m__alpha": alpha}}
                    label = f"{mname}(alpha={alpha}) | {tname} | clim={clim:.0%}"
                    configs.append((label, mlist, tcols, clim, overrides, tname))
            else:
                overrides = None
                label = f"{mname} | {tname} | clim={clim:.0%}"
                configs.append((label, mlist, tcols, clim, overrides, tname))

    if quick:
        # Fewer combos: only core4/core6, Ridge + Ridge+RF+GBR, subset clim
        configs = [
            (lb, mn, tc, cl, ov, tname)
            for lb, mn, tc, cl, ov, tname in configs
            if tc in (CORE_4, CORE_6, None) and (mn == ["Ridge"] or mn == ["Ridge", "RF", "GBR"])
            and cl in [0.0, 0.25, 0.5]
        ]
        # Add Ridge alpha sweep for Ridge-only (with tname)
        extra = []
        for alpha in ridge_alphas:
            for tname, tcols in [("core4", CORE_4), ("core6", CORE_6)]:
                for clim in [0.0, 0.5]:
                    label = f"Ridge(alpha={alpha}) | {tname} | clim={clim:.0%}"
                    extra.append((label, ["Ridge"], tcols, clim, {"Ridge": {"m__alpha": alpha}}, tname))
        configs = list(configs) + extra
        configs = configs[:50]

    return configs


def main():
    ap = argparse.ArgumentParser(description="Tune backtest: grid search over configs and params")
    ap.add_argument("--quick", action="store_true", help="Smaller grid for fast iteration")
    ap.add_argument("--top", type=int, default=15, help="Number of top configs to print")
    ap.add_argument("--limit", type=int, default=None, help="Max configs to run (for testing)")
    ap.add_argument("--snow", action="store_true", help="Also run snowfall grid; saves tune_backtest_results_snow.csv")
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("  Snoqualmie Pass Backtest Tuning — Grid Search")
    print("=" * 70)

    df = load_data()
    configs = build_grid(quick=args.quick)
    if args.limit:
        configs = configs[: args.limit]
    print(f"\nRunning {len(configs)} configs ...\n")

    from forecast import run_backtest

    results = []
    for i, (label, model_names, tele_subset, clim_blend, pipeline_overrides, tele_key) in enumerate(configs):
        try:
            bt = run_backtest(
                df, "WTEQ",
                model_names=model_names,
                tele_subset=tele_subset,
                clim_blend=clim_blend if clim_blend > 0 else None,  # 0 = no blend
                pipeline_overrides=pipeline_overrides,
                verbose=False,
            )
            skill = bt.get("skill")
            if skill is None or (isinstance(skill, float) and np.isnan(skill)):
                skill = -999
            results.append({
                "label": label,
                "model_names": "|".join(model_names) if model_names else "ensemble",
                "tele_subset": tele_key,
                "clim_blend": clim_blend,
                "ridge_alpha": (pipeline_overrides or {}).get("Ridge", {}).get("m__alpha"),
                "rmse": bt["rmse"],
                "rmse_clim": bt["rmse_clim"],
                "skill": skill,
                "corr": bt["correlation"],
                "bias": bt["bias"],
                "n": bt["n_points"],
            })
            status = "✓" if skill > -0.5 else " "  # ✓ = not terrible (skill > -50%), space = worse
            print(f"  [{i+1:3}/{len(configs)}] {status} skill={skill:.1%}  {label[:60]}")
        except Exception as e:
            results.append({
                "label": label,
                "model_names": "|".join(model_names) if model_names else "ensemble",
                "tele_subset": tele_key,
                "clim_blend": clim_blend,
                "ridge_alpha": (pipeline_overrides or {}).get("Ridge", {}).get("m__alpha"),
                "rmse": float("nan"), "rmse_clim": float("nan"),
                "skill": -999, "corr": float("nan"), "bias": float("nan"), "n": 0,
            })
            print(f"  [{i+1:3}/{len(configs)}] ✗ FAILED: {label[:50]} ... {e}")

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values("skill", ascending=False).reset_index(drop=True)

    out_path = os.path.join(DATA, "tune_backtest_results.csv")
    rdf.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    if getattr(args, "snow", False) and "snow_inches" in df.columns and df["snow_inches"].notna().any():
        print("\n[Snowfall grid] Running same configs for snow_inches ...\n")
        snow_results = []
        snow_configs = configs[: min(24, len(configs))]
        for i, (label, model_names, tele_subset, clim_blend, pipeline_overrides, tele_key) in enumerate(snow_configs):
            try:
                bt = run_backtest(df, "snow_inches", model_names=model_names, tele_subset=tele_subset,
                                  clim_blend=clim_blend if clim_blend > 0 else None,
                                  pipeline_overrides=pipeline_overrides, verbose=False)
                skill = bt.get("skill") if bt.get("skill") is not None and not (isinstance(bt.get("skill"), float) and np.isnan(bt.get("skill"))) else -999
                snow_results.append({"label": label, "model_names": "|".join(model_names) if model_names else "ensemble",
                    "tele_subset": tele_key, "clim_blend": clim_blend, "ridge_alpha": (pipeline_overrides or {}).get("Ridge", {}).get("m__alpha"),
                    "rmse": bt["rmse"], "rmse_clim": bt["rmse_clim"], "skill": skill, "corr": bt["correlation"], "bias": bt["bias"], "n": bt["n_points"]})
                print("  [%d/%d] skill=%s  %s" % (i + 1, len(snow_configs), "%.1f%%" % (skill * 100) if skill > -999 else "n/a", label[:50]))
            except Exception as e:
                snow_results.append({"label": label, "model_names": "|".join(model_names) if model_names else "ensemble", "tele_subset": tele_key, "clim_blend": clim_blend, "ridge_alpha": None,
                    "rmse": np.nan, "rmse_clim": np.nan, "skill": -999, "corr": np.nan, "bias": np.nan, "n": 0})
                print("  [%d/%d] FAILED: %s" % (i + 1, len(snow_configs), str(e)[:40]))
        sdf = pd.DataFrame(snow_results).sort_values("skill", ascending=False).reset_index(drop=True)
        snow_path = os.path.join(DATA, "tune_backtest_results_snow.csv")
        sdf.to_csv(snow_path, index=False)
        print("\nSaved %s" % snow_path)

    print("\n" + "=" * 70)
    print(f"  Top {args.top} configs by skill (1 - RMSE/RMSE_clim)")
    print("=" * 70)
    top = rdf.head(args.top)
    for i, row in top.iterrows():
        print(f"  {row['skill']:+.1%}  RMSE={row['rmse']:.2f}  corr={row['corr']:.3f}  |  {row['label']}")
    print("\nNote: positive skill = model beats climatology.")
    print("\nLegend: ✓ = skill > -50% (config not terrible); blank = worse; ✗ = run failed.")


if __name__ == "__main__":
    main()
