"""
Build a Snoqualmie Pass weather data pipeline.

This script fetches:
1) Historical weather-station data near Snoqualmie Pass
   - WSDOT RWIS stations (DOT API, requires WSDOT_ACCESS_CODE)
   - Open-Meteo archive at nearby points (long historical daily coverage)
2) Multi-model forecast data
   - ECMWF, GFS, HRRR (Open-Meteo forecast API)

Outputs are written to: data/pipeline/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "data" / "pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNOQUALMIE_LAT = 47.424
SNOQUALMIE_LON = -121.413
WSDOT_BASE = "https://wsdot.wa.gov/Traffic/api"

DEFAULT_OPEN_METEO_POINTS = [
    {"station_id": "snoqualmie_pass", "lat": 47.424, "lon": -121.413, "elev_ft": 3022.0},
    {"station_id": "north_bend", "lat": 47.495, "lon": -121.786, "elev_ft": 436.0},
    {"station_id": "cle_elum", "lat": 47.195, "lon": -120.939, "elev_ft": 1909.0},
    {"station_id": "stampede_pass", "lat": 47.278, "lon": -121.337, "elev_ft": 3965.0},
    {"station_id": "yakima", "lat": 46.602, "lon": -120.505, "elev_ft": 1099.0},
]
DEFAULT_FORECAST_MODELS = ["ecmwf_ifs025", "gfs_seamless", "hrrr_conus"]


@dataclass
class PipelineStats:
    wsdot_station_count: int = 0
    wsdot_raw_rows: int = 0
    wsdot_daily_rows: int = 0
    openmeteo_daily_rows: int = 0
    forecast_hourly_rows: int = 0
    forecast_daily_rows: int = 0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math

    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def parse_wsdot_time(x: object) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    s = str(x)
    m = re.search(r"/Date\(([-]?\d+)([-+]\d{4})?\)/", s)
    if m:
        ms = int(m.group(1))
        return pd.to_datetime(ms, unit="ms", utc=True)
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts


def _get_json(url: str, params: Dict[str, object], timeout: int = 90) -> object:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_wsdot_stations(access_code: str) -> pd.DataFrame:
    url = f"{WSDOT_BASE}/WeatherStations/WeatherStationsREST.svc/GetCurrentStationsAsJson"
    data = _get_json(url, params={"AccessCode": access_code}, timeout=60)
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for c in ["Latitude", "Longitude", "StationCode", "StationName"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["StationCode"] = pd.to_numeric(df["StationCode"], errors="coerce").astype("Int64")
    df = df[df["Latitude"].notna() & df["Longitude"].notna() & df["StationCode"].notna()].copy()
    if df.empty:
        return df
    df["distance_km"] = df.apply(
        lambda r_: haversine_km(SNOQUALMIE_LAT, SNOQUALMIE_LON, float(r_["Latitude"]), float(r_["Longitude"])),
        axis=1,
    )
    return df.sort_values("distance_km").reset_index(drop=True)


def fetch_wsdot_station_window(
    access_code: str, station_id: int, start_dt: datetime, end_dt: datetime
) -> List[Dict]:
    url = f"{WSDOT_BASE}/WeatherInformation/WeatherInformationREST.svc/SearchWeatherInformationAsJson"
    params = {
        "AccessCode": access_code,
        "StationID": station_id,
        "SearchStartTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "SearchEndTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        data = _get_json(url, params=params, timeout=90)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def fetch_wsdot_history(
    access_code: str, start_date: date, end_date: date, station_limit: int, chunk_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = get_wsdot_stations(access_code)
    if stations.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    nearby = stations.head(station_limit).copy()
    raw_rows: List[Dict] = []
    end_dt = datetime.combine(end_date, datetime.min.time())

    for _, row in nearby.iterrows():
        station_id = int(row["StationCode"])
        station_name = str(row["StationName"])
        current = datetime.combine(start_date, datetime.min.time())
        while current < end_dt:
            nxt = min(current + timedelta(days=chunk_days), end_dt)
            recs = fetch_wsdot_station_window(access_code, station_id, current, nxt)
            for rec in recs:
                rec["station_id"] = station_id
                rec["station_name"] = station_name
            raw_rows.extend(recs)
            current = nxt
            time.sleep(0.05)

    station_catalog = nearby[
        ["StationCode", "StationName", "Latitude", "Longitude", "distance_km"]
    ].rename(
        columns={
            "StationCode": "station_id",
            "StationName": "station_name",
            "Latitude": "lat",
            "Longitude": "lon",
        }
    )

    if not raw_rows:
        return station_catalog, pd.DataFrame(), pd.DataFrame()

    raw = pd.DataFrame(raw_rows)
    raw["time"] = raw.get("ReadingTime", pd.Series([None] * len(raw))).apply(parse_wsdot_time)
    raw = raw[raw["time"].notna()].copy()
    raw["date"] = pd.to_datetime(raw["time"]).dt.floor("D")

    numeric_map = {
        "TemperatureInFahrenheit": "temp_f",
        "PrecipitationInInches": "precip_in",
        "RelativeHumidity": "rh_pct",
        "BarometricPressure": "pressure",
        "WindSpeedInMPH": "wind_mph",
        "WindGustSpeedInMPH": "wind_gust_mph",
    }
    for src, dst in numeric_map.items():
        if src in raw.columns:
            raw[dst] = pd.to_numeric(raw[src], errors="coerce")

    keep = ["time", "date", "station_id", "station_name"] + [v for v in numeric_map.values() if v in raw.columns]
    raw = raw[keep].copy()

    daily = raw.groupby(["date", "station_id"], as_index=False).agg(
        temp_f_mean=("temp_f", "mean"),
        temp_f_min=("temp_f", "min"),
        temp_f_max=("temp_f", "max"),
        precip_in_sum=("precip_in", "sum"),
        rh_pct_mean=("rh_pct", "mean"),
        pressure_mean=("pressure", "mean"),
        wind_mph_mean=("wind_mph", "mean"),
        wind_gust_mph_max=("wind_gust_mph", "max"),
    )
    daily = daily.merge(station_catalog[["station_id", "station_name", "distance_km"]], on="station_id", how="left")
    return station_catalog, raw.sort_values("time"), daily.sort_values(["date", "station_id"])


def fetch_openmeteo_archive_daily(
    points: List[Dict[str, object]], start_date: date, end_date: date
) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    frames: List[pd.DataFrame] = []
    for p in points:
        params = {
            "latitude": p["lat"],
            "longitude": p["lon"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum,wind_speed_10m_mean,surface_pressure_mean",
            "timezone": "UTC",
        }
        payload = None
        for attempt in range(4):
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code >= 400:
                break
            payload = r.json().get("daily", {})
            if payload and "time" in payload:
                break
            time.sleep(1)
        if not payload or "time" not in payload:
            continue
        df = pd.DataFrame(payload).rename(
            columns={
                "time": "date",
                "temperature_2m_mean": "temp_c_mean",
                "temperature_2m_min": "temp_c_min",
                "temperature_2m_max": "temp_c_max",
                "precipitation_sum": "precip_mm_sum",
                "wind_speed_10m_mean": "wind_kph_mean",
                "surface_pressure_mean": "pressure_hpa_mean",
            }
        )
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.floor("D")
        df["station_id"] = str(p["station_id"])
        df["lat"] = float(p["lat"])
        df["lon"] = float(p["lon"])
        frames.append(df)
        time.sleep(0.2)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "station_id"]).reset_index(drop=True)


def fetch_openmeteo_forecasts(
    lat: float, lon: float, models: List[str], forecast_days: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_frames: List[pd.DataFrame] = []

    for model in models:
        params = {
            "latitude": lat,
            "longitude": lon,
            "models": model,
            "forecast_days": forecast_days,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,surface_pressure,freezing_level_height",
            "timezone": "UTC",
        }
        try:
            payload = _get_json(url, params=params, timeout=90)
        except Exception:
            continue
        hourly = payload.get("hourly", {})
        if "time" not in hourly:
            continue
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df = df[df["time"].notna()].copy()
        if df.empty:
            continue
        df["model"] = model
        df["date"] = df["time"].dt.floor("D")
        hourly_frames.append(df)

    hourly_frames = [df for df in hourly_frames if not df.empty]
    if not hourly_frames:
        return pd.DataFrame(), pd.DataFrame()

    hourly_all = pd.concat(hourly_frames, ignore_index=True)
    for col in [
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "surface_pressure",
        "freezing_level_height",
    ]:
        if col in hourly_all.columns:
            hourly_all[col] = pd.to_numeric(hourly_all[col], errors="coerce")

    daily = hourly_all.groupby(["date", "model"], as_index=False).agg(
        temp_c_mean=("temperature_2m", "mean"),
        temp_c_min=("temperature_2m", "min"),
        temp_c_max=("temperature_2m", "max"),
        precip_mm_sum=("precipitation", "sum"),
        wind_kph_mean=("wind_speed_10m", "mean"),
        pressure_hpa_mean=("surface_pressure", "mean"),
        freezing_level_m_mean=("freezing_level_height", "mean"),
    )
    return (
        hourly_all.sort_values(["model", "time"]).reset_index(drop=True),
        daily.sort_values(["date", "model"]).reset_index(drop=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Snoqualmie Pass weather station + model forecast pipeline."
    )
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="Historical start date (YYYY-MM-DD) for station data pulls.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Historical end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--wsdot-station-limit",
        type=int,
        default=8,
        help="How many nearest WSDOT stations to include.",
    )
    parser.add_argument(
        "--wsdot-chunk-days",
        type=int,
        default=7,
        help="WSDOT request chunk size in days (API supports ~1 week windows).",
    )
    parser.add_argument(
        "--forecast-days",
        type=int,
        default=16,
        help="Number of forecast days to request for each model.",
    )
    parser.add_argument(
        "--forecast-models",
        default=",".join(DEFAULT_FORECAST_MODELS),
        help="Comma-separated models (example: ecmwf_ifs025,gfs_seamless,hrrr_conus).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = PipelineStats()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    models = [m.strip() for m in str(args.forecast_models).split(",") if m.strip()]

    print("\n=== Snoqualmie Weather Data Pipeline ===")
    print(f"Historical window: {start_date} to {end_date}")
    print(f"Forecast models: {models}\n")

    access_code = os.getenv("WSDOT_ACCESS_CODE", "").strip()
    if access_code:
        print("[1/3] Fetching WSDOT RWIS historical data ...")
        station_catalog, wsdot_raw, wsdot_daily = fetch_wsdot_history(
            access_code=access_code,
            start_date=start_date,
            end_date=end_date,
            station_limit=max(1, int(args.wsdot_station_limit)),
            chunk_days=max(1, int(args.wsdot_chunk_days)),
        )
        if not station_catalog.empty:
            station_catalog.to_csv(OUT_DIR / "wsdot_station_catalog.csv", index=False)
            stats.wsdot_station_count = len(station_catalog)
        if not wsdot_raw.empty:
            wsdot_raw.to_csv(OUT_DIR / "wsdot_observations_raw.csv", index=False)
            stats.wsdot_raw_rows = len(wsdot_raw)
        if not wsdot_daily.empty:
            wsdot_daily.to_csv(OUT_DIR / "wsdot_daily_summary.csv", index=False)
            stats.wsdot_daily_rows = len(wsdot_daily)
        print(
            f"  WSDOT done: stations={stats.wsdot_station_count}, raw={stats.wsdot_raw_rows}, daily={stats.wsdot_daily_rows}"
        )
    else:
        print("[1/3] Skipping WSDOT RWIS: set WSDOT_ACCESS_CODE to enable DOT history.")

    print("[2/3] Fetching Open-Meteo nearby historical daily station data ...")
    openmeteo_station_daily = fetch_openmeteo_archive_daily(
        points=DEFAULT_OPEN_METEO_POINTS,
        start_date=start_date,
        end_date=end_date,
    )
    if not openmeteo_station_daily.empty:
        openmeteo_station_daily.to_csv(OUT_DIR / "openmeteo_station_daily.csv", index=False)
        stats.openmeteo_daily_rows = len(openmeteo_station_daily)
    print(f"  Open-Meteo historical rows: {stats.openmeteo_daily_rows}")

    print("[3/3] Fetching multi-model forecasts ...")
    fc_hourly, fc_daily = fetch_openmeteo_forecasts(
        lat=SNOQUALMIE_LAT,
        lon=SNOQUALMIE_LON,
        models=models,
        forecast_days=max(1, int(args.forecast_days)),
    )
    if not fc_hourly.empty:
        fc_hourly.to_csv(OUT_DIR / "model_forecast_hourly.csv", index=False)
        stats.forecast_hourly_rows = len(fc_hourly)
    if not fc_daily.empty:
        fc_daily.to_csv(OUT_DIR / "model_forecast_daily.csv", index=False)
        stats.forecast_daily_rows = len(fc_daily)
    print(f"  Forecast rows: hourly={stats.forecast_hourly_rows}, daily={stats.forecast_daily_rows}")

    manifest = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "historical_start_date": start_date.isoformat(),
        "historical_end_date": end_date.isoformat(),
        "forecast_models": models,
        "files": {
            "wsdot_station_catalog": "data/pipeline/wsdot_station_catalog.csv",
            "wsdot_observations_raw": "data/pipeline/wsdot_observations_raw.csv",
            "wsdot_daily_summary": "data/pipeline/wsdot_daily_summary.csv",
            "openmeteo_station_daily": "data/pipeline/openmeteo_station_daily.csv",
            "model_forecast_hourly": "data/pipeline/model_forecast_hourly.csv",
            "model_forecast_daily": "data/pipeline/model_forecast_daily.csv",
        },
        "row_counts": {
            "wsdot_station_count": stats.wsdot_station_count,
            "wsdot_raw_rows": stats.wsdot_raw_rows,
            "wsdot_daily_rows": stats.wsdot_daily_rows,
            "openmeteo_daily_rows": stats.openmeteo_daily_rows,
            "forecast_hourly_rows": stats.forecast_hourly_rows,
            "forecast_daily_rows": stats.forecast_daily_rows,
        },
    }
    (OUT_DIR / "pipeline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone. Output directory: data/pipeline/")
    print("  - pipeline_manifest.json")
    print("  - openmeteo_station_daily.csv")
    print("  - model_forecast_hourly.csv")
    print("  - model_forecast_daily.csv")
    if access_code:
        print("  - wsdot_station_catalog.csv")
        print("  - wsdot_observations_raw.csv")
        print("  - wsdot_daily_summary.csv")


if __name__ == "__main__":
    main()
