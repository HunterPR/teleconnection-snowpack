"""
Fetch additional external predictors for Snoqualmie forecasting.

Outputs:
  - data/met_daily_features.csv
  - data/met_station_catalog.csv
  - data/ndbc_multi_daily_features.csv

Notes:
  - Open-Meteo archive data is used as a practical RWIS-style proxy near Snoqualmie Pass.
  - NDBC buoy data here uses real-time feeds (recent window), intended mainly
    for current-condition forecasting context.
"""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path
import time
from typing import List

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNOQUALMIE_LAT = 47.424
SNOQUALMIE_LON = -121.413
SNOQUALMIE_ELEV_FT = 3022.0
LAPSE_RATE_C_PER_KM = 6.5
KM_TO_FT = 3280.84

NDBC_STATIONS = ["46005", "46041", "46087", "46088", "46050"]
OPEN_METEO_LOCATIONS = [
    {"name": "snoqualmie_pass", "lat": 47.424, "lon": -121.413, "elev_ft": 3022.0},
    {"name": "north_bend", "lat": 47.495, "lon": -121.786, "elev_ft": 436.0},
    {"name": "seattle", "lat": 47.606, "lon": -122.332, "elev_ft": 175.0},
    {"name": "yakima", "lat": 46.602, "lon": -120.505, "elev_ft": 1099.0},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def estimate_freezing_level_ft(temp_c: pd.Series, base_elev_ft: float) -> pd.Series:
    # Standard-atmosphere approximation from station temp to 0C altitude.
    delta_ft = (temp_c / LAPSE_RATE_C_PER_KM) * KM_TO_FT
    return base_elev_ft + delta_ft


def fetch_nearby_openmeteo_daily() -> None:
    print("[Open-Meteo] Fetching nearby daily weather features ...")
    start_date = "2000-01-01"
    end_date = date.today().isoformat()

    frames: List[pd.DataFrame] = []
    for loc in OPEN_METEO_LOCATIONS:
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": (
                "temperature_2m_mean,temperature_2m_min,temperature_2m_max,"
                "precipitation_sum,wind_speed_10m_mean,surface_pressure_mean"
            ),
            "timezone": "UTC",
        }
        r = None
        for attempt in range(3):
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90, headers=HEADERS)
            if r.status_code != 429:
                break
            time.sleep(2 * (attempt + 1))
        if r is None or r.status_code >= 400:
            continue
        payload = r.json().get("daily", {})
        if not payload or "time" not in payload:
            continue
        df = pd.DataFrame(payload).rename(
            columns={
                "time": "date",
                "temperature_2m_mean": "tavg",
                "temperature_2m_min": "tmin",
                "temperature_2m_max": "tmax",
                "precipitation_sum": "prcp",
                "wind_speed_10m_mean": "wspd",
                "surface_pressure_mean": "pres",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["tavg", "tmin", "tmax", "prcp", "wspd", "pres"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["freezing_level_est_ft"] = estimate_freezing_level_ft(df["tavg"], float(loc["elev_ft"]))
        df["freezing_line_gap_ft"] = df["freezing_level_est_ft"] - SNOQUALMIE_ELEV_FT
        df["station_id"] = loc["name"]
        frames.append(df)
        time.sleep(1)

    if not frames:
        raise RuntimeError("No Open-Meteo records fetched.")

    all_station_daily = pd.concat(frames, ignore_index=True)
    agg_specs = {
        "tavg": "mean",
        "tmin": "mean",
        "tmax": "mean",
        "prcp": "sum",
        "wspd": "mean",
        "pres": "mean",
        "freezing_level_est_ft": "mean",
        "freezing_line_gap_ft": "mean",
    }
    met_daily = all_station_daily.groupby("date", as_index=False).agg(agg_specs)
    met_daily = met_daily.rename(columns={c: f"met_{c}" for c in met_daily.columns if c != "date"})
    met_daily["date"] = pd.to_datetime(met_daily["date"], errors="coerce")
    met_daily = met_daily.sort_values("date").reset_index(drop=True)

    catalog = pd.DataFrame(OPEN_METEO_LOCATIONS).rename(
        columns={"name": "station_id", "lat": "latitude", "lon": "longitude"}
    )
    catalog.to_csv(DATA_DIR / "met_station_catalog.csv", index=False)
    met_daily.to_csv(DATA_DIR / "met_daily_features.csv", index=False)
    print(f"  Saved met_daily_features.csv ({len(met_daily)} rows)")
    print(f"  Saved met_station_catalog.csv ({len(catalog)} stations)")


def fetch_ndbc_recent_multi_station() -> None:
    print("[NDBC] Fetching multi-station recent buoy features ...")
    frames: List[pd.DataFrame] = []

    for station in NDBC_STATIONS:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"
        try:
            r = requests.get(url, timeout=60, headers=HEADERS)
            r.raise_for_status()
        except Exception:
            continue

        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue

        # First line is header, second line is units.
        header = lines[0].split()
        data_lines = lines[2:]
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            sep=r"\s+",
            names=header,
            on_bad_lines="skip",
        )
        for required in ["#YY", "MM", "DD"]:
            if required not in df.columns:
                continue
        if "#YY" not in df.columns or "MM" not in df.columns or "DD" not in df.columns:
            continue

        year = pd.to_numeric(df["#YY"], errors="coerce")
        month = pd.to_numeric(df["MM"], errors="coerce")
        day = pd.to_numeric(df["DD"], errors="coerce")
        date = pd.to_datetime({"year": year, "month": month, "day": day}, errors="coerce")
        df["date"] = date
        for col in ["WSPD", "GST", "WVHT", "PRES"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["station"] = station
        frames.append(df)

    if not frames:
        print("  No NDBC real-time station feeds were available.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["date"].notna()].copy()
    agg = merged.groupby(["date", "station"], as_index=False).agg(
        max_wspd=("WSPD", "max"),
        max_gst=("GST", "max"),
        max_wvht=("WVHT", "max"),
        min_pres=("PRES", "min"),
    )
    # Regional daily summary across stations.
    daily = agg.groupby("date", as_index=False).agg(
        ndbc_max_wspd_mean=("max_wspd", "mean"),
        ndbc_max_gst_mean=("max_gst", "mean"),
        ndbc_max_wvht_mean=("max_wvht", "mean"),
        ndbc_min_pres_mean=("min_pres", "mean"),
        ndbc_station_count=("station", "nunique"),
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily.to_csv(DATA_DIR / "ndbc_multi_daily_features.csv", index=False)
    print(f"  Saved ndbc_multi_daily_features.csv ({len(daily)} rows)")


def main() -> None:
    print("\n=== Fetch External Predictors (Snoqualmie) ===\n")
    errors = []
    try:
        fetch_nearby_openmeteo_daily()
    except Exception as exc:
        errors.append(("Open-Meteo", exc))
        print(f"  Open-Meteo error: {exc}")

    try:
        fetch_ndbc_recent_multi_station()
    except Exception as exc:
        errors.append(("NDBC", exc))
        print(f"  NDBC error: {exc}")

    print("\nDone.")
    if errors:
        print("Some sources failed:")
        for src, exc in errors:
            print(f"  - {src}: {exc}")


if __name__ == "__main__":
    main()
