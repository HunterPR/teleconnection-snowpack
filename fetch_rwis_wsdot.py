"""
Fetch RWIS-style daily features from WSDOT Traveler Information API.

Requires:
  - Environment variable WSDOT_ACCESS_CODE

Outputs:
  - data/rwis_daily_features.csv
  - data/rwis_station_catalog.csv
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SNOQUALMIE_LAT = 47.424
SNOQUALMIE_LON = -121.413
BASE = "https://wsdot.wa.gov/Traffic/api"


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


def get_stations(access_code: str) -> pd.DataFrame:
    url = f"{BASE}/WeatherStations/WeatherStationsREST.svc/GetCurrentStationsAsJson"
    r = requests.get(url, params={"AccessCode": access_code}, timeout=60)
    r.raise_for_status()
    data = r.json()
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


def fetch_station_window(access_code: str, station_id: int, start_dt: datetime, end_dt: datetime) -> List[Dict]:
    url = f"{BASE}/WeatherInformation/WeatherInformationREST.svc/SearchWeatherInformationAsJson"
    params = {
        "AccessCode": access_code,
        "StationID": station_id,
        "SearchStartTime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "SearchEndTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def main() -> None:
    access_code = os.getenv("WSDOT_ACCESS_CODE", "").strip()
    if not access_code:
        print("Missing WSDOT_ACCESS_CODE environment variable.")
        print("Set it, then rerun: python fetch_rwis_wsdot.py")
        return

    print("\n=== Fetch WSDOT RWIS Features ===\n")
    stations = get_stations(access_code)
    if stations.empty:
        raise RuntimeError("No station metadata returned from WSDOT API.")

    nearby = stations.head(6).copy()
    print("Using stations:")
    for _, r in nearby.iterrows():
        print(f"  {int(r['StationCode'])}: {r['StationName']} ({r['distance_km']:.1f} km)")

    start_date = date(2020, 1, 1)
    end_date = date.today()
    chunk_days = 14

    all_rows: List[Dict] = []
    for _, row in nearby.iterrows():
        station_id = int(row["StationCode"])
        station_name = str(row["StationName"])
        print(f"\n[Station {station_id}] {station_name}")
        cur = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time())
        rows_before = len(all_rows)
        while cur < end_dt:
            nxt = min(cur + timedelta(days=chunk_days), end_dt)
            records = fetch_station_window(access_code, station_id, cur, nxt)
            for rec in records:
                rec["station_id"] = station_id
                rec["station_name"] = station_name
            all_rows.extend(records)
            cur = nxt
        print(f"  raw rows fetched: {len(all_rows) - rows_before}")

    if not all_rows:
        raise RuntimeError("No RWIS weather observations retrieved.")

    df = pd.DataFrame(all_rows)
    if "ReadingTime" not in df.columns:
        raise RuntimeError("RWIS payload missing ReadingTime.")
    df["time"] = df["ReadingTime"].apply(parse_wsdot_time)
    df = df[df["time"].notna()].copy()
    df["date"] = pd.to_datetime(df["time"]).dt.floor("D")

    numeric_map = {
        "TemperatureInFahrenheit": "rwis_temp_f",
        "PrecipitationInInches": "rwis_precip_in",
        "RelativeHumidity": "rwis_rh_pct",
        "BarometricPressure": "rwis_pressure",
        "WindSpeedInMPH": "rwis_wind_mph",
        "WindGustSpeedInMPH": "rwis_wind_gust_mph",
    }
    for src, dst in numeric_map.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    keep = ["date", "station_id", "station_name"] + [c for c in numeric_map.values() if c in df.columns]
    df = df[keep].copy()

    station_daily = df.groupby(["date", "station_id"], as_index=False).agg(
        rwis_temp_f_mean=("rwis_temp_f", "mean"),
        rwis_temp_f_min=("rwis_temp_f", "min"),
        rwis_temp_f_max=("rwis_temp_f", "max"),
        rwis_precip_in_sum=("rwis_precip_in", "sum"),
        rwis_rh_pct_mean=("rwis_rh_pct", "mean"),
        rwis_pressure_mean=("rwis_pressure", "mean"),
        rwis_wind_mph_mean=("rwis_wind_mph", "mean"),
        rwis_wind_gust_mph_max=("rwis_wind_gust_mph", "max"),
    )

    regional_daily = station_daily.groupby("date", as_index=False).agg(
        rwis_temp_f_mean=("rwis_temp_f_mean", "mean"),
        rwis_temp_f_min=("rwis_temp_f_min", "mean"),
        rwis_temp_f_max=("rwis_temp_f_max", "mean"),
        rwis_precip_in_sum=("rwis_precip_in_sum", "mean"),
        rwis_rh_pct_mean=("rwis_rh_pct_mean", "mean"),
        rwis_pressure_mean=("rwis_pressure_mean", "mean"),
        rwis_wind_mph_mean=("rwis_wind_mph_mean", "mean"),
        rwis_wind_gust_mph_max=("rwis_wind_gust_mph_max", "mean"),
        rwis_station_count=("station_id", "nunique"),
    )
    regional_daily = regional_daily.sort_values("date").reset_index(drop=True)

    station_catalog = nearby[["StationCode", "StationName", "Latitude", "Longitude", "distance_km"]].rename(
        columns={"StationCode": "station_id", "StationName": "station_name", "Latitude": "lat", "Longitude": "lon"}
    )

    regional_daily.to_csv(DATA_DIR / "rwis_daily_features.csv", index=False)
    station_catalog.to_csv(DATA_DIR / "rwis_station_catalog.csv", index=False)
    print(f"\nSaved data/rwis_daily_features.csv ({len(regional_daily)} rows)")
    print(f"Saved data/rwis_station_catalog.csv ({len(station_catalog)} stations)")


if __name__ == "__main__":
    main()
