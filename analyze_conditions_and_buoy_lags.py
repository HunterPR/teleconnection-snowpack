"""
Analyze operational snow conditions and buoy lag correlations.

Outputs:
  - data/processed/analysis_buoy_lag_correlations.csv
  - data/processed/analysis_operational_metrics_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "data" / "processed" / "snoqualmie_model_daily.csv"
OUT_DIR = ROOT / "data" / "processed"


def calc_buoy_lag_correlations(df: pd.DataFrame) -> pd.DataFrame:
    buoy_base = [
        "buoy_max_wspd_mean",
        "buoy_max_gst_mean",
        "buoy_max_wvht_mean",
        "buoy_min_pres_mean",
    ]
    lag_days = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30]
    targets = [
        "target_snowfall_24h_in",
        "target_precip_24h_in",
        "target_freezing_hours_24h",
        "target_snowfall_likely_hours_24h",
    ]
    targets = [t for t in targets if t in df.columns]

    rows: List[dict] = []
    for target in targets:
        for col in buoy_base:
            if col not in df.columns:
                continue
            for lag in lag_days:
                series = pd.to_numeric(df[col], errors="coerce").shift(lag)
                sample = pd.concat([pd.to_numeric(df[target], errors="coerce"), series], axis=1).dropna()
                if len(sample) < 300:
                    continue
                corr = sample.iloc[:, 0].corr(sample.iloc[:, 1])
                rows.append(
                    {
                        "target": target,
                        "feature": col,
                        "lag_days": lag,
                        "corr": corr,
                        "abs_corr": abs(corr),
                        "sample_size": len(sample),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["target", "abs_corr"], ascending=[True, False]).reset_index(drop=True)


def calc_operational_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "target_snowmaking_good_hours_24h",
        "target_snowmaking_marginal_hours_24h",
        "target_snowfall_possible_hours_24h",
        "target_snowfall_likely_hours_24h",
        "target_snowfall_24h_in",
        "target_precip_24h_in",
    ]
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        return pd.DataFrame()

    desc = df[metrics].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T.reset_index()
    desc = desc.rename(columns={"index": "metric"})
    return desc


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {MODEL_PATH}")
    df = pd.read_csv(MODEL_PATH, low_memory=False)

    corr_df = calc_buoy_lag_correlations(df)
    ops_df = calc_operational_summary(df)

    corr_path = OUT_DIR / "analysis_buoy_lag_correlations.csv"
    ops_path = OUT_DIR / "analysis_operational_metrics_summary.csv"
    corr_df.to_csv(corr_path, index=False)
    ops_df.to_csv(ops_path, index=False)

    print(f"Saved {corr_path.relative_to(ROOT)} ({len(corr_df)} rows)")
    print(f"Saved {ops_path.relative_to(ROOT)} ({len(ops_df)} rows)")


if __name__ == "__main__":
    main()
