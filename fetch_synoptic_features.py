"""
Fetch historical and near-term synoptic features for Snoqualmie forecasting.

Outputs:
  - data/synoptic_daily_features.csv
  - data/synoptic_forecast_daily_features.csv
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import time

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNOQUALMIE_FT = 3022.0

# Two points: offshore and inland to capture pressure/height gradient regime.
POINTS = [
    {"name": "offshore", "lat": 48.0, "lon": -128.0},
    {"name": "cascade", "lat": 47.42, "lon": -121.41},
]


def fetch_archive_hourly(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    start_y = int(start_date[:4])
    end_y = int(end_date[:4])
    parts: List[pd.DataFrame] = []

    for y in range(start_y, end_y + 1):
        s = f"{y}-01-01" if y > start_y else start_date
        e = f"{y}-12-31" if y < end_y else end_date
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": s,
            "end_date": e,
            "hourly": (
                "geopotential_height_500hPa,geopotential_height_850hPa,"
                "freezing_level_height,surface_pressure,temperature_2m,wind_speed_10m,precipitation"
            ),
            "timezone": "UTC",
        }

        payload = None
        for attempt in range(3):
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code != 200:
                break
            try:
                payload = r.json().get("hourly", {})
            except Exception:
                payload = None
            if payload is not None:
                break
            time.sleep(1)

        if not payload or "time" not in payload:
            continue
        df = pd.DataFrame(payload)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df[df["time"].notna()].copy()
        if not df.empty:
            parts.append(df)
        time.sleep(0.5)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def fetch_forecast_hourly(lat: float, lon: float, days: int = 16) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "geopotential_height_500hPa,geopotential_height_850hPa,"
            "freezing_level_height,surface_pressure,temperature_2m,wind_speed_10m,precipitation"
        ),
        "timezone": "UTC",
        "forecast_days": days,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    payload = r.json().get("hourly", {})
    if "time" not in payload:
        return pd.DataFrame()
    df = pd.DataFrame(payload)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df[df["time"].notna()].copy()
    return df


def aggregate_daily(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["time"]).dt.floor("D")
    for c in work.columns:
        if c not in {"time", "date"}:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    daily = work.groupby("date", as_index=False).agg(
        hgt500_mean=("geopotential_height_500hPa", "mean"),
        hgt850_mean=("geopotential_height_850hPa", "mean"),
        freezing_level_m_mean=("freezing_level_height", "mean"),
        surface_pressure_mean=("surface_pressure", "mean"),
        t2m_mean=("temperature_2m", "mean"),
        t2m_min=("temperature_2m", "min"),
        t2m_max=("temperature_2m", "max"),
        wind10m_mean=("wind_speed_10m", "mean"),
        precip_sum=("precipitation", "sum"),
    )
    daily = daily.rename(columns={c: f"{prefix}_{c}" for c in daily.columns if c != "date"})
    return daily


def build_synoptic_features(point_daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for _, df in point_daily.items():
        merged = df if merged is None else merged.merge(df, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["date"])

    # Derived synoptic regime features.
    if {"offshore_hgt500_mean", "cascade_hgt500_mean"}.issubset(merged.columns):
        merged["syn_hgt500_gradient_offshore_minus_cascade"] = (
            merged["offshore_hgt500_mean"] - merged["cascade_hgt500_mean"]
        )
    if {"offshore_surface_pressure_mean", "cascade_surface_pressure_mean"}.issubset(merged.columns):
        merged["syn_slp_gradient_offshore_minus_cascade"] = (
            merged["offshore_surface_pressure_mean"] - merged["cascade_surface_pressure_mean"]
        )
    if "cascade_hgt500_mean" in merged.columns and "cascade_hgt850_mean" in merged.columns:
        merged["syn_thickness_proxy_500_850"] = (
            merged["cascade_hgt500_mean"] - merged["cascade_hgt850_mean"]
        )
    if "cascade_freezing_level_m_mean" in merged.columns:
        merged["syn_freezing_level_ft"] = merged["cascade_freezing_level_m_mean"] * 3.28084
        merged["syn_freezing_line_gap_ft"] = merged["syn_freezing_level_ft"] - SNOQUALMIE_FT
    if "cascade_t2m_mean" in merged.columns:
        merged["syn_tavg_c"] = merged["cascade_t2m_mean"]
    if "cascade_precip_sum" in merged.columns:
        merged["syn_precip_mm"] = merged["cascade_precip_sum"]

    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def main() -> None:
    print("\n=== Fetch Synoptic Features ===\n")
    start = "2003-01-01"
    end = date.today().isoformat()

    hist_parts: Dict[str, pd.DataFrame] = {}
    fc_parts: Dict[str, pd.DataFrame] = {}

    for p in POINTS:
        print(f"[{p['name']}] fetching archive + forecast...")
        hist_hourly = fetch_archive_hourly(p["lat"], p["lon"], start, end)
        fc_hourly = fetch_forecast_hourly(p["lat"], p["lon"], days=16)
        hist_parts[p["name"]] = aggregate_daily(hist_hourly, p["name"])
        fc_parts[p["name"]] = aggregate_daily(fc_hourly, p["name"])

    hist = build_synoptic_features(hist_parts)
    fc = build_synoptic_features(fc_parts)

    hist.to_csv(DATA_DIR / "synoptic_daily_features.csv", index=False)
    fc.to_csv(DATA_DIR / "synoptic_forecast_daily_features.csv", index=False)
    print(f"Saved data/synoptic_daily_features.csv ({len(hist)} rows)")
    print(f"Saved data/synoptic_forecast_daily_features.csv ({len(fc)} rows)")


if __name__ == "__main__":
    main()
