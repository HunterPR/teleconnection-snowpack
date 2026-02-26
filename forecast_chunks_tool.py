"""
Multi-horizon chunk forecast tool for Snoqualmie:
  - 14-day daily forecast
  - 30-day (monthly-style) chunk summary
  Targets: snowfall, precip, freezing-hours

Also writes:
  - JSON summary output
  - calibration plot for snowfall event probability
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "data" / "processed" / "snoqualmie_model_daily.csv"
SYNOPTIC_FC_PATH = ROOT / "data" / "synoptic_forecast_daily_features.csv"

OUT_DAILY = ROOT / "data" / "processed" / "forecast_14day_daily.csv"
OUT_JSON = ROOT / "data" / "processed" / "forecast_chunk_summary.json"
OUT_CAL_PLOT = ROOT / "plots" / "presentation" / "07_snow_event_calibration.png"

SNOW_EVENT_THR = 3.0
PRECIP_EVENT_THR = 0.5
FREEZE_EVENT_THR_HRS = 12.0


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    return df.sort_values("date").reset_index(drop=True)


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snow_lag1"] = out["target_snowfall_24h_in"].shift(1)
    out["snow_lag2"] = out["target_snowfall_24h_in"].shift(2)
    out["snow_lag3"] = out["target_snowfall_24h_in"].shift(3)
    out["precip_lag1"] = out["target_precip_24h_in"].shift(1)
    out["freeze_lag1"] = out["target_freezing_hours_24h"].shift(1)
    return out


def select_features(df: pd.DataFrame) -> List[str]:
    cols = [
        "met_tavg",
        "met_tmin",
        "met_tmax",
        "met_prcp",
        "met_wspd",
        "met_pres",
        "met_freezing_line_gap_ft",
        "ndbc_max_wspd_mean",
        "ndbc_max_wvht_mean",
        "ndbc_min_pres_mean",
        "ndbc_station_count",
        "rwis_temp_f_mean",
        "rwis_precip_in_sum",
        "rwis_pressure_mean",
        "rwis_wind_mph_mean",
        "syn_hgt500_gradient_offshore_minus_cascade",
        "syn_slp_gradient_offshore_minus_cascade",
        "syn_thickness_proxy_500_850",
        "syn_freezing_line_gap_ft",
        "ao",
        "nao",
        "pna",
        "pdo",
        "oni_anomaly",
        "np",
        "wp",
        "qbo",
        "pmm",
        "snow_lag1",
        "snow_lag2",
        "snow_lag3",
        "precip_lag1",
        "freeze_lag1",
    ]
    cols = [c for c in cols if c in df.columns]
    cols += ["month", "dayofyear", "is_cool_season"]
    return cols


def prep_training(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
    w = df.copy()
    for c in [target_col] + feature_cols:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    w["target_next"] = w[target_col].shift(-1)
    w["month"] = w["date"].dt.month
    w["dayofyear"] = w["date"].dt.dayofyear
    w["is_cool_season"] = w["month"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    cols = ["date", "target_next"] + feature_cols
    out = w[cols].dropna(subset=["target_next"]).reset_index(drop=True)
    return out


def fit_regressor(train: pd.DataFrame, feature_cols: List[str]):
    reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    reg.fit(train[feature_cols], train["target_next"])
    return reg


def fit_classifier(train: pd.DataFrame, feature_cols: List[str], event_thr: float):
    y = (train["target_next"] >= event_thr).astype(int)
    clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    clf.fit(train[feature_cols], y)
    return clf


def apply_synoptic_to_future(last_hist_row: pd.Series, syn_fc: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    future = syn_fc.copy()
    future = future.sort_values("date").head(max(horizon, 30)).copy()
    if future.empty:
        raise RuntimeError("No synoptic forecast rows found. Run fetch_synoptic_features.py first.")

    rows = []
    for _, r in future.iterrows():
        row = last_hist_row.copy()
        row["date"] = pd.to_datetime(r["date"])
        # Map synoptic forecast to met-style predictors where possible.
        if "cascade_t2m_mean" in r:
            row["met_tavg"] = float(r["cascade_t2m_mean"])
        if "cascade_t2m_min" in r:
            row["met_tmin"] = float(r["cascade_t2m_min"])
        if "cascade_t2m_max" in r:
            row["met_tmax"] = float(r["cascade_t2m_max"])
        if "cascade_precip_sum" in r:
            row["met_prcp"] = float(r["cascade_precip_sum"]) / 25.4
        if "cascade_surface_pressure_mean" in r:
            row["met_pres"] = float(r["cascade_surface_pressure_mean"])
        if "cascade_wind10m_mean" in r:
            row["met_wspd"] = float(r["cascade_wind10m_mean"])
        if "syn_freezing_line_gap_ft" in r:
            row["met_freezing_line_gap_ft"] = float(r["syn_freezing_line_gap_ft"])
            row["syn_freezing_line_gap_ft"] = float(r["syn_freezing_line_gap_ft"])
        for c in [
            "syn_hgt500_gradient_offshore_minus_cascade",
            "syn_slp_gradient_offshore_minus_cascade",
            "syn_thickness_proxy_500_850",
        ]:
            if c in r:
                row[c] = float(r[c])
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def calibrate_plot(train: pd.DataFrame, feature_cols: List[str], event_thr: float) -> Dict[str, float]:
    n_holdout = min(365, max(120, len(train) // 5))
    split = len(train) - n_holdout
    tr = train.iloc[:split]
    te = train.iloc[split:]
    if len(tr) < 200 or len(te) < 80:
        return {"brier": float("nan"), "mae": float("nan")}

    reg = fit_regressor(tr, feature_cols)
    clf = fit_classifier(tr, feature_cols, event_thr)

    y_true = te["target_next"].values
    y_pred = reg.predict(te[feature_cols])
    p = clf.predict_proba(te[feature_cols])[:, 1]
    y_event = (y_true >= event_thr).astype(int)

    # Reliability bins
    bins = np.linspace(0, 1, 11)
    inds = np.digitize(p, bins) - 1
    x, y = [], []
    for i in range(10):
        mask = inds == i
        if mask.sum() < 15:
            continue
        x.append(p[mask].mean())
        y.append(y_event[mask].mean())

    OUT_CAL_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    if x:
        ax.plot(x, y, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Snow Event Calibration (>= 3 in/day)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_CAL_PLOT, dpi=160)
    plt.close(fig)

    return {
        "brier": float(brier_score_loss(y_event, p)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def main() -> None:
    df = load_df(MODEL_PATH)
    df = add_lags(df)
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_cool_season"] = df["month"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)

    syn_fc = load_df(SYNOPTIC_FC_PATH)
    feature_cols = select_features(df)

    # Train target-specific models.
    targets = [
        ("target_snowfall_24h_in", SNOW_EVENT_THR),
        ("target_precip_24h_in", PRECIP_EVENT_THR),
        ("target_freezing_hours_24h", FREEZE_EVENT_THR_HRS),
    ]
    models = {}
    for target, thr in targets:
        tr = prep_training(df, target, feature_cols)
        reg = fit_regressor(tr, feature_cols)
        clf = fit_classifier(tr, feature_cols, thr)
        models[target] = {"reg": reg, "clf": clf, "thr": thr, "train": tr}

    # Build future forcing rows from synoptic forecast + last known context.
    last_hist = df.iloc[-1].copy()
    future = apply_synoptic_to_future(last_hist, syn_fc, horizon=14)

    # Iterative prediction so lags can update with predicted outcomes.
    daily_rows = []
    snow_lags = [
        float(last_hist.get("target_snowfall_24h_in", 0.0) or 0.0),
        float(df.iloc[-2].get("target_snowfall_24h_in", 0.0) if len(df) > 1 else 0.0),
        float(df.iloc[-3].get("target_snowfall_24h_in", 0.0) if len(df) > 2 else 0.0),
    ]
    precip_lag1 = float(last_hist.get("target_precip_24h_in", 0.0) or 0.0)
    freeze_lag1 = float(last_hist.get("target_freezing_hours_24h", 0.0) or 0.0)

    for i in range(min(14, len(future))):
        row = future.iloc[[i]].copy()
        row["snow_lag1"] = snow_lags[0]
        row["snow_lag2"] = snow_lags[1]
        row["snow_lag3"] = snow_lags[2]
        row["precip_lag1"] = precip_lag1
        row["freeze_lag1"] = freeze_lag1
        row["month"] = pd.to_datetime(row["date"]).dt.month.values[0]
        row["dayofyear"] = pd.to_datetime(row["date"]).dt.dayofyear.values[0]
        row["is_cool_season"] = int(row["month"].iloc[0] in [10, 11, 12, 1, 2, 3, 4])

        x = row[feature_cols]
        pred_snow = float(models["target_snowfall_24h_in"]["reg"].predict(x)[0])
        pred_precip = float(models["target_precip_24h_in"]["reg"].predict(x)[0])
        pred_freeze = float(models["target_freezing_hours_24h"]["reg"].predict(x)[0])

        p_snow = float(models["target_snowfall_24h_in"]["clf"].predict_proba(x)[0, 1])
        p_precip = float(models["target_precip_24h_in"]["clf"].predict_proba(x)[0, 1])
        p_freeze = float(models["target_freezing_hours_24h"]["clf"].predict_proba(x)[0, 1])

        daily_rows.append(
            {
                "date": pd.to_datetime(row["date"].iloc[0]).date().isoformat(),
                "pred_snowfall_in": max(0.0, pred_snow),
                "pred_precip_in": max(0.0, pred_precip),
                "pred_freezing_hours": max(0.0, min(24.0, pred_freeze)),
                "p_snow_ge_3in": p_snow,
                "p_precip_ge_0p5in": p_precip,
                "p_freeze_ge_12h": p_freeze,
            }
        )

        # Update lags for next day.
        snow_lags = [max(0.0, pred_snow), snow_lags[0], snow_lags[1]]
        precip_lag1 = max(0.0, pred_precip)
        freeze_lag1 = max(0.0, min(24.0, pred_freeze))

    daily_fc = pd.DataFrame(daily_rows)
    OUT_DAILY.parent.mkdir(parents=True, exist_ok=True)
    daily_fc.to_csv(OUT_DAILY, index=False)

    # 30-day chunk estimate: 14-day model output + climatology for remaining days.
    hist = df.copy()
    hist["month"] = hist["date"].dt.month
    clim = hist.groupby("month", as_index=False).agg(
        clim_snow=("target_snowfall_24h_in", "mean"),
        clim_precip=("target_precip_24h_in", "mean"),
        clim_freeze=("target_freezing_hours_24h", "mean"),
    )
    clim_map = {int(r["month"]): r for _, r in clim.iterrows()}

    monthly_total_snow = daily_fc["pred_snowfall_in"].sum()
    monthly_total_precip = daily_fc["pred_precip_in"].sum()
    monthly_total_freeze = daily_fc["pred_freezing_hours"].sum()

    if len(daily_fc) > 0:
        last_date = pd.to_datetime(daily_fc["date"].iloc[-1])
    else:
        last_date = pd.to_datetime(df["date"].iloc[-1])

    needed = 30 - len(daily_fc)
    cur = last_date
    for _ in range(max(0, needed)):
        cur = cur + pd.Timedelta(days=1)
        m = int(cur.month)
        c = clim_map.get(m)
        if c is not None:
            monthly_total_snow += float(c["clim_snow"])
            monthly_total_precip += float(c["clim_precip"])
            monthly_total_freeze += float(c["clim_freeze"])

    # Calibration diagnostics for snow event model.
    cal = calibrate_plot(models["target_snowfall_24h_in"]["train"], feature_cols, SNOW_EVENT_THR)

    summary = {
        "forecast_generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "horizon_days": int(len(daily_fc)),
        "two_week": {
            "snowfall_total_in": float(daily_fc["pred_snowfall_in"].sum()),
            "precip_total_in": float(daily_fc["pred_precip_in"].sum()),
            "freezing_hours_total": float(daily_fc["pred_freezing_hours"].sum()),
            "snow_event_days_expected": float((daily_fc["p_snow_ge_3in"] > 0.5).sum()),
            "mean_daily_p_snow_ge_3in": float(daily_fc["p_snow_ge_3in"].mean()),
        },
        "thirty_day_chunk_estimate": {
            "snowfall_total_in": float(monthly_total_snow),
            "precip_total_in": float(monthly_total_precip),
            "freezing_hours_total": float(monthly_total_freeze),
            "method": "14-day model + monthly climatology extension",
        },
        "model_backtest": {
            "snow_event_brier": cal["brier"],
            "snow_nextday_mae_in": cal["mae"],
        },
        "files": {
            "daily_forecast_csv": str(OUT_DAILY.relative_to(ROOT)),
            "calibration_plot": str(OUT_CAL_PLOT.relative_to(ROOT)),
        },
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Forecast Chunk Tool ===")
    print(f"Saved {OUT_DAILY}")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {OUT_CAL_PLOT}")
    print("\nTwo-week forecast:")
    print(f"  snowfall_total_in: {summary['two_week']['snowfall_total_in']:.2f}")
    print(f"  precip_total_in:   {summary['two_week']['precip_total_in']:.2f}")
    print(f"  freezing_hours:    {summary['two_week']['freezing_hours_total']:.1f}")
    print("\n30-day chunk estimate:")
    print(f"  snowfall_total_in: {summary['thirty_day_chunk_estimate']['snowfall_total_in']:.2f}")
    print(f"  precip_total_in:   {summary['thirty_day_chunk_estimate']['precip_total_in']:.2f}")
    print(f"  freezing_hours:    {summary['thirty_day_chunk_estimate']['freezing_hours_total']:.1f}")


if __name__ == "__main__":
    main()
