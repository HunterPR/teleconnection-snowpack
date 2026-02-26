"""
Create presentation-ready visuals and summary tables for Snoqualmie forecasting.

Outputs:
  - plots/presentation/*.png
  - data/processed/slide_metrics.csv
  - SLIDE_CONCEPTS.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "data" / "processed" / "snoqualmie_model_daily.csv"
PLOTS_DIR = ROOT / "plots" / "presentation"
METRICS_PATH = ROOT / "data" / "processed" / "slide_metrics.csv"
SLIDE_MD_PATH = ROOT / "SLIDE_CONCEPTS.md"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(MODEL_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    for c in [
        "target_snowfall_24h_in",
        "target_precip_24h_in",
        "target_freezing_hours_24h",
        "met_tavg",
        "met_freezing_line_gap_ft",
        "met_prcp",
        "ndbc_min_pres_mean",
        "wp",
        "np",
        "oni_anomaly",
        "pdo",
        "pna",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["water_year"] = np.where(out["month"] >= 10, out["year"] + 1, out["year"])
    return out


def plot_target_overview(df: pd.DataFrame) -> None:
    yearly = (
        df.groupby("water_year", as_index=False)
        .agg(
            snowfall_total=("target_snowfall_24h_in", "sum"),
            precip_total=("target_precip_24h_in", "sum"),
            freezing_hours_total=("target_freezing_hours_24h", "sum"),
        )
        .sort_values("water_year")
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(yearly["water_year"], yearly["snowfall_total"], color="steelblue", lw=1.8)
    axes[0].set_ylabel("Snowfall (in)")
    axes[0].set_title("Snoqualmie Water-Year Target Totals")
    axes[0].grid(alpha=0.3)

    axes[1].plot(yearly["water_year"], yearly["precip_total"], color="teal", lw=1.8)
    axes[1].set_ylabel("Precip (in)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(yearly["water_year"], yearly["freezing_hours_total"], color="darkorange", lw=1.8)
    axes[2].set_ylabel("Freezing hrs")
    axes[2].set_xlabel("Water Year")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_target_overview_water_year.png", dpi=170)
    plt.close(fig)


def plot_monthly_climatology(df: pd.DataFrame) -> None:
    cool = df[df["month"].isin([10, 11, 12, 1, 2, 3, 4])].copy()
    grouped = (
        cool.groupby("month", as_index=False)
        .agg(
            snowfall_mean=("target_snowfall_24h_in", "mean"),
            snowfall_q75=("target_snowfall_24h_in", lambda s: s.quantile(0.75)),
            precip_mean=("target_precip_24h_in", "mean"),
            freezing_mean=("target_freezing_hours_24h", "mean"),
        )
        .sort_values("month")
    )
    month_names = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    x_labels = [month_names[m] for m in grouped["month"]]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(grouped))
    ax1.plot(x, grouped["snowfall_mean"], marker="o", color="steelblue", label="Mean snowfall (in/day)")
    ax1.plot(x, grouped["snowfall_q75"], marker="o", ls="--", color="royalblue", label="75th pct snowfall")
    ax1.plot(x, grouped["precip_mean"], marker="o", color="teal", label="Mean precip (in/day)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Snow / Precip")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, grouped["freezing_mean"], marker="s", color="darkorange", label="Mean freezing hours/day")
    ax2.set_ylabel("Freezing Hours")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax1.set_title("Cool-Season Monthly Climatology")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_monthly_climatology.png", dpi=170)
    plt.close(fig)


def plot_feature_correlation_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_snowfall_24h_in",
        "target_precip_24h_in",
        "target_freezing_hours_24h",
        "met_tavg",
        "met_prcp",
        "met_freezing_line_gap_ft",
        "ndbc_min_pres_mean",
        "ndbc_max_wspd_mean",
        "ndbc_max_wvht_mean",
        "oni_anomaly",
        "pdo",
        "pna",
        "np",
        "wp",
    ]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=70, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Daily Feature Correlation Matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_feature_correlation_heatmap.png", dpi=170)
    plt.close(fig)
    return corr


def plot_heavy_snow_event_comparison(df: pd.DataFrame) -> pd.DataFrame:
    thr = df["target_snowfall_24h_in"].quantile(0.95)
    df = df.copy()
    df["heavy_snow"] = df["target_snowfall_24h_in"] >= thr

    compare_cols = [
        "met_freezing_line_gap_ft",
        "met_tavg",
        "met_prcp",
        "ndbc_min_pres_mean",
        "ndbc_max_wspd_mean",
        "wp",
        "np",
        "oni_anomaly",
    ]
    compare_cols = [c for c in compare_cols if c in df.columns]

    rows: List[Dict[str, float]] = []
    for c in compare_cols:
        event_mean = pd.to_numeric(df.loc[df["heavy_snow"], c], errors="coerce").mean()
        nonevent_mean = pd.to_numeric(df.loc[~df["heavy_snow"], c], errors="coerce").mean()
        rows.append(
            {
                "feature": c,
                "event_mean": float(event_mean),
                "non_event_mean": float(nonevent_mean),
                "difference": float(event_mean - nonevent_mean),
            }
        )
    comp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(comp))
    w = 0.37
    ax.bar(x - w / 2, comp["event_mean"], width=w, color="tomato", label="Heavy snow days")
    ax.bar(x + w / 2, comp["non_event_mean"], width=w, color="steelblue", label="Non-event days")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["feature"], rotation=35, ha="right")
    ax.set_title("Heavy Snow vs Non-Event Predictor Averages")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_heavy_snow_event_comparison.png", dpi=170)
    plt.close(fig)
    return comp


def plot_lag_scan(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(["year", "month"], as_index=False)
        .agg(target_snowfall_24h_in=("target_snowfall_24h_in", "sum"))
        .sort_values(["year", "month"])
    )

    tele = ["ao", "nao", "pna", "pdo", "oni_anomaly", "np", "wp", "qbo", "pmm"]
    tele = [c for c in tele if c in df.columns]
    tele_monthly = (
        df.groupby(["year", "month"], as_index=False)[tele]
        .mean()
        .sort_values(["year", "month"])
    )
    merged = monthly.merge(tele_monthly, on=["year", "month"], how="left")

    lags = [0, 1, 2, 3]
    lag_rows = []
    for c in tele:
        for lag in lags:
            r = merged["target_snowfall_24h_in"].corr(merged[c].shift(lag))
            lag_rows.append({"feature": c, "lag": lag, "corr": float(r) if pd.notna(r) else np.nan})
    lag_df = pd.DataFrame(lag_rows)

    features = sorted(lag_df["feature"].unique())
    mat = np.full((len(features), len(lags)), np.nan)
    for i, f in enumerate(features):
        for j, lag in enumerate(lags):
            v = lag_df[(lag_df["feature"] == f) & (lag_df["lag"] == lag)]["corr"]
            if not v.empty:
                mat[i, j] = v.iloc[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks(np.arange(len(lags)))
    ax.set_xticklabels([f"lag{l}" for l in lags])
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    ax.set_title("Teleconnection Lag Scan vs Monthly Snowfall")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_teleconnection_lag_scan.png", dpi=170)
    plt.close(fig)
    return lag_df


def save_slide_metrics(corr: pd.DataFrame, event_comp: pd.DataFrame, lag_df: pd.DataFrame) -> None:
    rows = []
    for target in ["target_snowfall_24h_in", "target_precip_24h_in", "target_freezing_hours_24h"]:
        if target not in corr.columns:
            continue
        vals = corr[target].drop(labels=[target], errors="ignore").dropna()
        if vals.empty:
            continue
        top = vals.reindex(vals.abs().sort_values(ascending=False).index).head(5)
        for f, v in top.items():
            rows.append({"section": f"top_corr_{target}", "metric": f, "value": float(v)})

    for _, r in event_comp.iterrows():
        rows.append({"section": "heavy_snow_diff", "metric": r["feature"], "value": float(r["difference"])})

    for feature in lag_df["feature"].unique():
        sub = lag_df[lag_df["feature"] == feature].copy()
        sub = sub.dropna(subset=["corr"])
        if sub.empty:
            continue
        best = sub.iloc[sub["corr"].abs().argmax()]
        rows.append(
            {
                "section": "best_lag_snowfall",
                "metric": feature,
                "value": float(best["corr"]),
                "extra": f"lag{int(best['lag'])}",
            }
        )

    pd.DataFrame(rows).to_csv(METRICS_PATH, index=False)


def write_slide_concepts() -> None:
    text = """# Snoqualmie Forecast Slide Concepts

## 1) Why This Matters
- Snoqualmie Pass operations are highly sensitive to snowfall phase and freezing level.
- Goal: improve storm-period decisions with earlier probabilistic signal.

## 2) Data Coverage
- Daily targets from Snoqualmie station history (2003+).
- Teleconnections + nearby SNOTEL + streamflow + historical marine predictors.
- New marine context from historical NDBC backfill improves long-period coverage.

## 3) Bottom-Line Findings
- Freezing-line/temperature metrics are strongest direct snowfall-phase signals.
- Marine pressure/wind adds useful storm-intensity context.
- Teleconnections are weaker daily predictors but useful regime indicators at monthly scale.

## 4) Proposed Forecast Products
- Event probability: P(snowfall_24h >= 3 inches).
- Event probability: P(freezing_line_gap_ft < 0).
- Daily expected snowfall/precip range with confidence bands.

## 5) Next Data Acquisitions
- WSDOT RWIS nearest-pass stations (road temp + precip type).
- Reanalysis 500mb height / SLP gradient indices.
- Higher-resolution freezing-level truth source for calibration.

## 6) Operational Decision Framing
- Use event probabilities with threshold triggers (plow staffing, chain messaging).
- Communicate forecast in tiers: low / moderate / high impact windows.
"""
    SLIDE_MD_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    df = add_time_features(load_data())

    plot_target_overview(df)
    plot_monthly_climatology(df)
    corr = plot_feature_correlation_heatmap(df)
    event_comp = plot_heavy_snow_event_comparison(df)
    lag_df = plot_lag_scan(df)

    save_slide_metrics(corr, event_comp, lag_df)
    write_slide_concepts()
    print(f"Saved presentation plots to {PLOTS_DIR}")
    print(f"Saved metrics table to {METRICS_PATH}")
    print(f"Saved slide concepts to {SLIDE_MD_PATH}")


if __name__ == "__main__":
    main()
