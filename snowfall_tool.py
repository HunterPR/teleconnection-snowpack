"""
Snowfall estimation tool for Snoqualmie Pass.

Purpose:
  Given current and recent conditions, estimate next-day snowfall totals.

Method:
  - Train on historical daily data from data/processed/snoqualmie_model_daily.csv
  - Use a one-day-ahead target (predict snowfall on D+1 from features on D)
  - Combine:
      1) Gradient boosting regression (point estimate)
      2) Nearest-neighbor analogs (distribution / uncertainty)
  - Return:
      - Expected snowfall (inches)
      - Probability snowfall exceeds threshold (default >= 3 inches)
      - Analog range (p25 / p50 / p75)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
MODEL_TABLE = ROOT / "data" / "processed" / "snoqualmie_model_daily.csv"

TARGET = "target_snowfall_24h_in"
EVENT_THRESHOLD_IN = 3.0


@dataclass
class ToolOutput:
    target_date: pd.Timestamp
    expected_snow_in: float
    prob_event: float
    analog_p25: float
    analog_p50: float
    analog_p75: float
    analog_count: int


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing model table: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_recent_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snow_lag1"] = out[TARGET].shift(1)
    out["snow_lag2"] = out[TARGET].shift(2)
    out["snow_lag3"] = out[TARGET].shift(3)
    out["precip_lag1"] = out["target_precip_24h_in"].shift(1) if "target_precip_24h_in" in out.columns else np.nan
    out["freeze_lag1"] = out["target_freezing_hours_24h"].shift(1) if "target_freezing_hours_24h" in out.columns else np.nan
    return out


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        # Recent station meteorology
        "met_tavg",
        "met_tmin",
        "met_tmax",
        "met_prcp",
        "met_wspd",
        "met_pres",
        "met_freezing_line_gap_ft",
        # Marine context
        "ndbc_max_wspd_mean",
        "ndbc_max_wvht_mean",
        "ndbc_min_pres_mean",
        "ndbc_station_count",
        # Teleconnection regime context
        "ao",
        "nao",
        "pna",
        "pdo",
        "oni_anomaly",
        "np",
        "wp",
        "qbo",
        "pmm",
        # Nearby SNOTEL context
        "stampede_wteq",
        "olallie_wteq",
        "tinkham_wteq",
        # Recent observed lags
        "snow_lag1",
        "snow_lag2",
        "snow_lag3",
        "precip_lag1",
        "freeze_lag1",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        raise ValueError("No usable feature columns found in model table.")
    return cols


def prepare_training_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work["target_nextday_snow_in"] = work[TARGET].shift(-1)
    work["month"] = work["date"].dt.month
    work["dayofyear"] = work["date"].dt.dayofyear
    work["is_cool_season"] = work["month"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    final_features = feature_cols + ["month", "dayofyear", "is_cool_season"]
    work = work[["date", "target_nextday_snow_in"] + final_features].copy()
    work = work.dropna(subset=["target_nextday_snow_in"]).reset_index(drop=True)
    return work


def train_models(train: pd.DataFrame, features: List[str]):
    X = train[features].copy()
    y = pd.to_numeric(train["target_nextday_snow_in"], errors="coerce")
    y_event = (y >= EVENT_THRESHOLD_IN).astype(int)

    reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    )
    reg.fit(X, y)
    clf.fit(X, y_event)
    return reg, clf


def backtest_quick(train: pd.DataFrame, features: List[str]) -> tuple[float, float]:
    # Hold out the most recent 365 observations for a quick realism check.
    n_holdout = min(365, max(0, len(train) // 5))
    if n_holdout < 100:
        return float("nan"), float("nan")
    split = len(train) - n_holdout
    tr = train.iloc[:split]
    te = train.iloc[split:]

    reg, clf = train_models(tr, features)
    pred = reg.predict(te[features])
    prob = clf.predict_proba(te[features])[:, 1]

    mae = mean_absolute_error(te["target_nextday_snow_in"], pred)
    event_true = (te["target_nextday_snow_in"] >= EVENT_THRESHOLD_IN).astype(int)
    if event_true.nunique() < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(event_true, prob)
    return float(mae), float(auc)


def estimate_with_analogs(train: pd.DataFrame, features: List[str], latest_row: pd.DataFrame, k: int = 40):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = train[features]
    X_latest = latest_row[features]
    X_train_imp = imputer.fit_transform(X_train)
    X_latest_imp = imputer.transform(X_latest)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_latest_scaled = scaler.transform(X_latest_imp)

    nn = NearestNeighbors(n_neighbors=min(k, len(train)), metric="euclidean")
    nn.fit(X_train_scaled)
    _, idx = nn.kneighbors(X_latest_scaled)
    analog_targets = train.iloc[idx[0]]["target_nextday_snow_in"].astype(float)
    return analog_targets


def predict_next_day(df: pd.DataFrame, threshold_in: float) -> ToolOutput:
    work = add_recent_lags(df)
    feature_cols = choose_feature_columns(work)
    train = prepare_training_frame(work, feature_cols)
    features = [c for c in train.columns if c not in {"date", "target_nextday_snow_in"}]

    reg, clf = train_models(train, features)
    latest = work.iloc[[-1]].copy()
    latest["month"] = latest["date"].dt.month
    latest["dayofyear"] = latest["date"].dt.dayofyear
    latest["is_cool_season"] = latest["month"].isin([10, 11, 12, 1, 2, 3, 4]).astype(int)
    latest = latest[features]

    reg_pred = float(reg.predict(latest)[0])
    prob = float(clf.predict_proba(latest)[0, 1])

    analog_targets = estimate_with_analogs(train, features, latest, k=40)
    q25 = float(analog_targets.quantile(0.25))
    q50 = float(analog_targets.quantile(0.50))
    q75 = float(analog_targets.quantile(0.75))
    analog_mean = float(analog_targets.mean())

    # Blend model + analog mean for stability.
    expected = 0.65 * reg_pred + 0.35 * analog_mean
    if expected < 0:
        expected = 0.0

    target_date = pd.Timestamp(df.iloc[-1]["date"]) + pd.Timedelta(days=1)
    return ToolOutput(
        target_date=target_date,
        expected_snow_in=float(expected),
        prob_event=float(prob if threshold_in == EVENT_THRESHOLD_IN else np.nan),
        analog_p25=q25,
        analog_p50=q50,
        analog_p75=q75,
        analog_count=int(len(analog_targets)),
    )


def print_data_gaps(df: pd.DataFrame) -> None:
    key_cols = [
        "ndbc_max_wvht_mean",
        "ndbc_min_pres_mean",
        "met_freezing_line_gap_ft",
        "stampede_wteq",
        "olallie_wteq",
        "tinkham_wteq",
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols:
        return
    print("\nCoverage check (fraction non-null):")
    for c in key_cols:
        frac = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
        print(f"  {c:28s} {frac:.3f}")


def main() -> None:
    # Quiet joblib physical-core warning on some Windows setups.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

    parser = argparse.ArgumentParser(description="Estimate next-day Snoqualmie snowfall.")
    parser.add_argument("--table", type=str, default=str(MODEL_TABLE), help="Path to model table CSV.")
    parser.add_argument("--event-threshold", type=float, default=EVENT_THRESHOLD_IN, help="Event threshold in inches.")
    args = parser.parse_args()

    table_path = Path(args.table)
    df = load_data(table_path)
    if len(df) < 500:
        raise RuntimeError("Insufficient history in model table for stable tool output.")

    out = predict_next_day(df, threshold_in=float(args.event_threshold))
    work = add_recent_lags(df)
    feature_cols = choose_feature_columns(work)
    train = prepare_training_frame(work, feature_cols)
    features = [c for c in train.columns if c not in {"date", "target_nextday_snow_in"}]
    mae, auc = backtest_quick(train, features)

    print("\n=== Snoqualmie Next-Day Snowfall Tool ===")
    print(f"Latest observed date: {pd.Timestamp(df.iloc[-1]['date']).date()}")
    print(f"Forecast target date: {out.target_date.date()}")
    print(f"\nExpected snowfall (24h): {out.expected_snow_in:.2f} in")
    print(f"Analog distribution (k={out.analog_count}): p25={out.analog_p25:.2f}, p50={out.analog_p50:.2f}, p75={out.analog_p75:.2f}")
    if np.isfinite(out.prob_event):
        print(f"P(snowfall >= {args.event_threshold:.1f} in): {out.prob_event:.2%}")
    if np.isfinite(mae):
        print(f"\nQuick backtest: MAE={mae:.2f} in/day")
    if np.isfinite(auc):
        print(f"Quick backtest event AUC (>= {EVENT_THRESHOLD_IN:.1f} in): {auc:.3f}")

    print_data_gaps(df)

    print("\nRecommended next data additions:")
    print("  - WSDOT RWIS station history near Snoqualmie (road temp, precip type).")
    print("  - Reanalysis 500mb height / SLP gradients over NE Pacific.")
    print("  - Better freezing-level truth (soundings/reanalysis) for calibration.")


if __name__ == "__main__":
    main()
