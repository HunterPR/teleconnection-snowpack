"""
nowcast.py — Layer 2: Local telemetry nowcast module.
====================================================

Reads sub-hourly station data (ALP31, ALP44, SNO30, SNO38) and computes:
  - Daily snowfall (depth-gain method), liquid precip, temperature
  - Multi-elevation temperature profile -> freezing level, snow/rain line, inversions
  - Pressure trends -> ridge/trough detection
  - Current-month snowfall/SWE pacing for blending with Layer 1 (teleconnection) forecasts
  - Open-Meteo pressure-level vertical profiles (soundings) with wet-bulb
  - Forecast freezing level + snowmaking/snowfall windows

Station layout (west to east, low to high):
  SNO30  3010 ft  Snoqualmie Pass — temp, RH, snow depth, precip, pressure
  ALP31  3100 ft  Alpental Base   — temp, RH, snow depth, precip
  SNO38  3760 ft  Dodge Ridge     — temp, wind speed/dir/gust
  ALP44  4350 ft  Alpental Mid    — temp, RH, snow depth
  ALP55  5400 ft  Alpental Summit — temp, wind, snow depth (wind-scoured)
"""

import os
import glob
import calendar
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

# Station metadata: name, elevation (ft), key variables available
STATIONS = {
    "ALP31": {"name": "Alpental Base",   "elev_ft": 3100, "has": ["snow_depth", "precip", "temp", "rh"]},
    "SNO30": {"name": "Snoqualmie Pass", "elev_ft": 3010, "has": ["snow_depth", "precip", "temp", "rh", "pressure"]},
    "ALP44": {"name": "Alpental Mid",    "elev_ft": 4350, "has": ["snow_depth", "temp", "rh"]},
    "SNO38": {"name": "Dodge Ridge",     "elev_ft": 3760, "has": ["temp", "wind"]},
    "ALP55": {"name": "Alpental Summit", "elev_ft": 5400, "has": ["snow_depth", "temp", "wind"]},
}

# Columns to extract by type
COL_MAP = {
    "temp":       "air_temp_set_1",
    "rh":         "relative_humidity_set_1",
    "snow_depth": "snow_depth_set_1",
    "precip":     "precip_accum_one_hour_set_1",
    "pressure":   "sea_level_pressure_set_1d",
    "wind_speed": "wind_speed_set_1",
    "wind_dir":   "wind_direction_set_1",
    "wind_gust":  "wind_gust_set_1",
}


# ── Station data loading ──────────────────────────────────────────────────────

def find_station_file(station_id: str) -> str:
    """Find the most recent CSV file for a station."""
    pattern = os.path.join(BASE, f"{station_id}.*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = os.path.join(BASE, f"{station_id}*.csv")
        files = sorted(glob.glob(pattern))
    return files[-1] if files else ""


def load_station(station_id: str) -> pd.DataFrame:
    """Load a station CSV, handling MesoWest header format."""
    fpath = find_station_file(station_id)
    if not fpath:
        return pd.DataFrame()
    df = pd.read_csv(fpath, comment="#", skiprows=[1], low_memory=False)
    df.columns = df.columns.str.strip()
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date_Time"])
    # Convert numeric columns
    for key, col in COL_MAP.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["station"] = station_id
    return df


# ── Daily aggregation ─────────────────────────────────────────────────────────

def daily_snow_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily snowfall using the depth-gain method:
      snowfall_est = max(0, today_max_depth - yesterday_min_depth)

    This captures new snow before compaction destroys the signal.
    Also returns daily precip total, mean temp, mean depth.
    """
    if df.empty or "snow_depth_set_1" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = df["Date_Time"].dt.date

    agg = {"snow_depth_set_1": ["max", "min", "mean"]}
    if "precip_accum_one_hour_set_1" in df.columns:
        agg["precip_accum_one_hour_set_1"] = [("precip_total", lambda x: x.clip(lower=0).sum())]
    if "air_temp_set_1" in df.columns:
        agg["air_temp_set_1"] = ["mean", "min", "max"]

    daily = df.groupby("date").agg(agg)
    daily.columns = ["_".join(c).strip("_") for c in daily.columns]

    # Rename for clarity
    rename = {
        "snow_depth_set_1_max": "depth_max",
        "snow_depth_set_1_min": "depth_min",
        "snow_depth_set_1_mean": "depth_mean",
        "air_temp_set_1_mean": "temp_mean_f",
        "air_temp_set_1_min": "temp_min_f",
        "air_temp_set_1_max": "temp_max_f",
    }
    daily = daily.rename(columns=rename)

    # Fix precip column name (lambda creates ugly names)
    precip_cols = [c for c in daily.columns if "precip" in c.lower()]
    if precip_cols:
        daily = daily.rename(columns={precip_cols[0]: "precip_in"})

    # Depth-gain snowfall estimate
    daily["snowfall_est"] = (daily["depth_max"] - daily["depth_min"].shift(1)).clip(lower=0)

    daily.index = pd.to_datetime(daily.index)
    return daily


def compute_daily_all_stations() -> dict:
    """Load and aggregate daily stats for all stations with snow depth."""
    results = {}
    for sid in ["ALP31", "SNO30", "ALP44"]:
        raw = load_station(sid)
        if raw.empty:
            continue
        daily = daily_snow_stats(raw)
        if not daily.empty:
            results[sid] = daily
    return results


# ── Multi-elevation temperature profile ───────────────────────────────────────

def compute_freezing_level() -> pd.DataFrame:
    """
    Estimate freezing level (ft ASL) from multi-elevation temperature data.

    Uses ALP31 (3100'), SNO38/ALP44 (3760'/4350') to fit a lapse rate
    per time step and find where T = 32F.

    Also detects inversions (temperature increasing with elevation).
    """
    # Load hourly temps at each elevation
    profiles = []
    for sid in ["ALP31", "SNO30", "ALP44", "SNO38", "ALP55"]:
        raw = load_station(sid)
        if raw.empty or "air_temp_set_1" not in raw.columns:
            continue
        elev = STATIONS[sid]["elev_ft"]
        temp = raw[["Date_Time", "air_temp_set_1"]].copy()
        temp = temp.rename(columns={"air_temp_set_1": f"temp_{sid}"})
        temp[f"elev_{sid}"] = elev
        temp = temp.set_index("Date_Time")
        # Resample to hourly
        temp = temp.resample("1h").mean()
        profiles.append(temp)

    if len(profiles) < 2:
        return pd.DataFrame()

    merged = pd.concat(profiles, axis=1)

    # For each hour, fit lapse rate and find freezing level
    temp_cols = [c for c in merged.columns if c.startswith("temp_")]
    elev_cols = [c for c in merged.columns if c.startswith("elev_")]

    rows = []
    for idx, row in merged.iterrows():
        temps = []
        elevs = []
        for tc, ec in zip(temp_cols, elev_cols):
            if pd.notna(row.get(tc)) and pd.notna(row.get(ec)):
                temps.append(row[tc])
                elevs.append(row[ec])
        if len(temps) < 2:
            continue

        temps = np.array(temps)
        elevs = np.array(elevs)

        # Lapse rate: dT/dz (F per ft), negative = normal decrease
        if len(temps) >= 2:
            # Linear fit: T = a * elev + b
            coeffs = np.polyfit(elevs, temps, 1)
            lapse_rate = coeffs[0]  # F per ft
            # Freezing level: solve a * z + b = 32
            if abs(lapse_rate) > 1e-6:
                freeze_elev = (32.0 - coeffs[1]) / lapse_rate
            else:
                freeze_elev = np.nan
            # Inversion: lapse rate > 0 means temp increases with height
            inversion = lapse_rate > 0
            # Snow level ~ freezing level (simplified; wet-bulb would be better)
            snow_level = freeze_elev
        else:
            lapse_rate = np.nan
            freeze_elev = np.nan
            inversion = False
            snow_level = np.nan

        rows.append({
            "time": idx,
            "freezing_level_ft": freeze_elev,
            "snow_level_ft": snow_level,
            "lapse_rate_f_per_kft": lapse_rate * 1000,  # F per 1000 ft
            "inversion": inversion,
            "n_stations": len(temps),
        })

    return pd.DataFrame(rows).set_index("time")


# ── Pressure trends (ridge/trough) ───────────────────────────────────────────

def compute_pressure_trends(hours: int = 24) -> pd.DataFrame:
    """
    Compute SLP trends from SNO30 to detect approaching ridges/troughs.
      - Falling pressure (< -2 hPa / 12h): trough approaching (storms likely)
      - Rising pressure (> +2 hPa / 12h): ridge building (clearing)
      - Steady: neutral
    """
    raw = load_station("SNO30")
    if raw.empty or "sea_level_pressure_set_1d" not in raw.columns:
        return pd.DataFrame()

    pres = raw[["Date_Time", "sea_level_pressure_set_1d"]].copy()
    pres = pres.set_index("Date_Time").resample("1h").mean()
    pres.columns = ["slp_hpa"]

    # 3-hour and 12-hour trends
    pres["slp_3h_change"] = pres["slp_hpa"] - pres["slp_hpa"].shift(3)
    pres["slp_12h_change"] = pres["slp_hpa"] - pres["slp_hpa"].shift(12)
    pres["slp_24h_change"] = pres["slp_hpa"] - pres["slp_hpa"].shift(24)

    def classify(change):
        if pd.isna(change):
            return "unknown"
        if change < -3:
            return "trough_approaching"
        elif change < -1:
            return "falling"
        elif change > 3:
            return "ridge_building"
        elif change > 1:
            return "rising"
        return "steady"

    pres["pattern_12h"] = pres["slp_12h_change"].apply(classify)
    return pres


# ── Current month pacing ──────────────────────────────────────────────────────

def current_month_pace(year: int = None, month: int = None) -> dict:
    """
    Compute current-month snowfall and SWE pacing from station data.

    Returns dict with:
      - actual_snowfall_in: total depth-gain snowfall so far this month
      - actual_precip_in: total liquid precip so far
      - days_elapsed: days with data
      - days_in_month: total days in the month
      - pace_snowfall_in: projected end-of-month snowfall at current pace
      - swe_gain_est_in: estimated SWE gain (precip total or snowfall/10)
    """
    now = datetime.now(timezone.utc)
    if year is None:
        year = now.year
    if month is None:
        month = now.month

    days_in_month = calendar.monthrange(year, month)[1]

    # Use ALP31 as primary (best snow depth + precip coverage at pass level)
    raw = load_station("ALP31")
    if raw.empty:
        raw = load_station("SNO30")
    if raw.empty:
        return {"error": "No station data available"}

    daily = daily_snow_stats(raw)
    if daily.empty:
        return {"error": "No daily stats computed"}

    # Filter to target month
    month_data = daily[(daily.index.year == year) & (daily.index.month == month)]
    if month_data.empty:
        return {"error": f"No data for {year}-{month:02d}"}

    actual_snow = month_data["snowfall_est"].sum()
    days_elapsed = len(month_data)
    pace_snow = actual_snow * (days_in_month / max(1, days_elapsed))

    precip_total = month_data["precip_in"].sum() if "precip_in" in month_data.columns else np.nan
    # SWE gain estimate: liquid precip is the best proxy
    swe_gain = precip_total if pd.notna(precip_total) else actual_snow / 10.0

    depth_start = month_data["depth_mean"].iloc[0] if "depth_mean" in month_data.columns else np.nan
    depth_latest = month_data["depth_mean"].iloc[-1] if "depth_mean" in month_data.columns else np.nan

    return {
        "year": year,
        "month": month,
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "actual_snowfall_in": round(actual_snow, 1),
        "pace_snowfall_in": round(pace_snow, 1),
        "actual_precip_in": round(precip_total, 2) if pd.notna(precip_total) else None,
        "swe_gain_est_in": round(swe_gain, 2) if pd.notna(swe_gain) else None,
        "depth_start_in": round(depth_start, 1) if pd.notna(depth_start) else None,
        "depth_latest_in": round(depth_latest, 1) if pd.notna(depth_latest) else None,
        "station": raw["station"].iloc[0] if "station" in raw.columns else "unknown",
    }


# ── Blending: Layer 1 (teleconnection) + Layer 2 (telemetry) ─────────────────

def blend_forecast(layer1_snowfall: float, layer1_swe: float,
                   pace: dict, snotel_swe_start: float = None) -> dict:
    """
    Blend Layer 1 (monthly teleconnection forecast) with Layer 2 (station pace).

    For the current month:
      - weight_actual increases linearly with days elapsed (0 at start, 1 at end)
      - blended_snow = w * pace_snowfall + (1-w) * layer1_snowfall
      - blended_swe adjusts layer1 prediction by observed SWE gain rate

    For future months: returns layer1 values unchanged.
    """
    if "error" in pace:
        return {
            "blended_snowfall": layer1_snowfall,
            "blended_swe": layer1_swe,
            "blend_source": "layer1_only",
            "note": pace.get("error", ""),
        }

    days = pace["days_elapsed"]
    total = pace["days_in_month"]
    w = min(1.0, days / total)  # weight for actual data

    # Snowfall blend
    pace_snow = pace["pace_snowfall_in"]
    blended_snow = w * pace_snow + (1 - w) * layer1_snowfall

    # SWE blend: if we know SWE at month start, add observed gain
    if snotel_swe_start is not None and pace.get("swe_gain_est_in") is not None:
        swe_gain_pace = pace["swe_gain_est_in"] * (total / max(1, days))
        blended_swe = snotel_swe_start + swe_gain_pace
    else:
        blended_swe = layer1_swe

    return {
        "blended_snowfall": round(blended_snow, 1),
        "blended_swe": round(blended_swe, 1),
        "blend_weight_actual": round(w, 2),
        "layer1_snowfall": layer1_snowfall,
        "layer1_swe": layer1_swe,
        "pace_snowfall": pace_snow,
        "actual_snowfall_to_date": pace["actual_snowfall_in"],
        "blend_source": f"blended (w_actual={w:.0%})",
    }


# ── Open-Meteo pressure-level soundings ──────────────────────────────────────

SNOQUALMIE_LAT = 47.424
SNOQUALMIE_LON = -121.413

# Pressure levels (hPa) and approximate altitudes:
#   1000 ≈ 300 ft (sea level), 925 ≈ 2600 ft (below pass),
#   850 ≈ 4900 ft (above pass), 700 ≈ 9800 ft, 500 ≈ 18400 ft
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]


def stull_wetbulb_c(t_c: float, rh: float) -> float:
    """Compute wet-bulb temperature (C) from T(C) and RH(%) using Stull (2011)."""
    if np.isnan(t_c) or np.isnan(rh):
        return np.nan
    rh = max(5, min(99, rh))  # clamp for formula stability
    wb = (t_c * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
          + np.arctan(t_c + rh)
          - np.arctan(rh - 1.676331)
          + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
          - 4.686035)
    return wb


def fetch_openmeteo_sounding(
    lat: float = SNOQUALMIE_LAT,
    lon: float = SNOQUALMIE_LON,
    forecast_days: int = 5,
    models: list = None,
) -> pd.DataFrame:
    """
    Fetch vertical profile forecast from Open-Meteo at pressure levels.

    Returns hourly DataFrame with columns:
      time, model, level_hPa, temp_c, dewpoint_c, rh, wind_speed_kph,
      wind_dir, geopotential_m, wetbulb_c, freezing_level_m
    """
    if models is None:
        models = ["ecmwf_ifs025", "gfs_seamless"]

    url = "https://api.open-meteo.com/v1/forecast"

    # Build pressure-level variable names
    plevel_vars = []
    for lev in PRESSURE_LEVELS:
        plevel_vars.extend([
            f"temperature_{lev}hPa",
            f"dewpoint_{lev}hPa",
            f"relative_humidity_{lev}hPa",
            f"windspeed_{lev}hPa",
            f"winddirection_{lev}hPa",
            f"geopotential_height_{lev}hPa",
        ])

    # Surface variables (freezing level + 2m for context)
    surface_vars = [
        "freezing_level_height",
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
    ]

    all_frames = []
    for model in models:
        params = {
            "latitude": lat,
            "longitude": lon,
            "models": model,
            "forecast_days": forecast_days,
            "hourly": ",".join(plevel_vars + surface_vars),
            "timezone": "UTC",
        }
        payload = None
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=90)
                if r.status_code == 429:
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    print(f"   Sounding API error ({model}): HTTP {r.status_code}")
                    break
                payload = r.json().get("hourly", {})
                if payload and "time" in payload:
                    break
            except Exception as e:
                print(f"   Sounding fetch error ({model}): {e}")
                time.sleep(2)
        if not payload or "time" not in payload:
            continue

        times = pd.to_datetime(payload["time"], errors="coerce", utc=True)
        n = len(times)

        # Extract freezing level (surface variable)
        fzl = pd.to_numeric(pd.Series(payload.get("freezing_level_height", [None]*n)), errors="coerce")
        t2m = pd.to_numeric(pd.Series(payload.get("temperature_2m", [None]*n)), errors="coerce")
        rh2m = pd.to_numeric(pd.Series(payload.get("relative_humidity_2m", [None]*n)), errors="coerce")

        # Build one row per (time, level)
        for lev in PRESSURE_LEVELS:
            t_col = f"temperature_{lev}hPa"
            d_col = f"dewpoint_{lev}hPa"
            rh_col = f"relative_humidity_{lev}hPa"
            ws_col = f"windspeed_{lev}hPa"
            wd_col = f"winddirection_{lev}hPa"
            gh_col = f"geopotential_height_{lev}hPa"

            temp_c = pd.to_numeric(pd.Series(payload.get(t_col, [None]*n)), errors="coerce")
            dew_c = pd.to_numeric(pd.Series(payload.get(d_col, [None]*n)), errors="coerce")
            rh_vals = pd.to_numeric(pd.Series(payload.get(rh_col, [None]*n)), errors="coerce")
            ws_vals = pd.to_numeric(pd.Series(payload.get(ws_col, [None]*n)), errors="coerce")
            wd_vals = pd.to_numeric(pd.Series(payload.get(wd_col, [None]*n)), errors="coerce")
            gh_vals = pd.to_numeric(pd.Series(payload.get(gh_col, [None]*n)), errors="coerce")

            lev_df = pd.DataFrame({
                "time": times,
                "model": model,
                "level_hPa": lev,
                "temp_c": temp_c,
                "dewpoint_c": dew_c,
                "rh": rh_vals,
                "wind_speed_kph": ws_vals,
                "wind_dir": wd_vals,
                "geopotential_m": gh_vals,
                "freezing_level_m": fzl,
                "t2m_c": t2m,
                "rh2m": rh2m,
            })
            all_frames.append(lev_df)

        time.sleep(0.3)  # rate limit courtesy

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df = df.dropna(subset=["time"])

    # Compute wet-bulb at each level
    df["wetbulb_c"] = df.apply(lambda r: stull_wetbulb_c(r["temp_c"], r["rh"]), axis=1)
    # Surface wet-bulb
    df["wetbulb_2m_c"] = df.apply(lambda r: stull_wetbulb_c(r["t2m_c"], r["rh2m"]), axis=1)

    # Convert key values to F for US display
    df["temp_f"] = df["temp_c"] * 9/5 + 32
    df["wetbulb_f"] = df["wetbulb_c"] * 9/5 + 32
    df["dewpoint_f"] = df["dewpoint_c"] * 9/5 + 32
    df["geopotential_ft"] = df["geopotential_m"] * 3.28084
    df["freezing_level_ft"] = df["freezing_level_m"] * 3.28084

    return df.sort_values(["model", "time", "level_hPa"]).reset_index(drop=True)


def sounding_summary(sounding_df: pd.DataFrame) -> dict:
    """
    Summarize sounding data into a compact dict for nowcast output.

    Returns: freezing level forecast, snow level estimate, wet-bulb profile,
    snowfall/snowmaking windows, and wind at key levels.
    """
    if sounding_df.empty:
        return {"error": "No sounding data"}

    result = {}

    # Multi-model mean for current conditions (first 6 hours)
    now_utc = pd.Timestamp.now(tz="UTC")
    near = sounding_df[sounding_df["time"] <= now_utc + pd.Timedelta(hours=6)]
    if near.empty:
        near = sounding_df.head(len(PRESSURE_LEVELS) * 2)

    # Freezing level forecast (next 48h, 120h)
    fzl = sounding_df.groupby("time")["freezing_level_ft"].first().dropna()
    if not fzl.empty:
        next_48 = fzl[fzl.index <= now_utc + pd.Timedelta(hours=48)]
        next_120 = fzl[fzl.index <= now_utc + pd.Timedelta(hours=120)]
        result["freezing_level_forecast"] = {
            "current_ft": int(round(fzl.iloc[0])),
            "min_48h_ft": int(round(next_48.min())) if len(next_48) > 0 else None,
            "max_48h_ft": int(round(next_48.max())) if len(next_48) > 0 else None,
            "mean_48h_ft": int(round(next_48.mean())) if len(next_48) > 0 else None,
            "min_120h_ft": int(round(next_120.min())) if len(next_120) > 0 else None,
            "max_120h_ft": int(round(next_120.max())) if len(next_120) > 0 else None,
        }

    # Vertical profile snapshot (most recent, multi-model mean)
    snap = near.groupby("level_hPa").agg(
        temp_f=("temp_f", "mean"),
        wetbulb_f=("wetbulb_f", "mean"),
        dewpoint_f=("dewpoint_f", "mean"),
        rh=("rh", "mean"),
        wind_speed_kph=("wind_speed_kph", "mean"),
        wind_dir=("wind_dir", "mean"),
        geopotential_ft=("geopotential_ft", "mean"),
    ).round(1)
    profile = []
    for lev, row in snap.iterrows():
        profile.append({
            "level_hPa": int(lev),
            "altitude_ft": int(round(row["geopotential_ft"])),
            "temp_f": round(row["temp_f"], 1),
            "wetbulb_f": round(row["wetbulb_f"], 1),
            "dewpoint_f": round(row["dewpoint_f"], 1),
            "rh_pct": round(row["rh"], 0),
            "wind_kph": round(row["wind_speed_kph"], 0),
            "wind_dir": round(row["wind_dir"], 0),
        })
    result["vertical_profile"] = profile

    # Snow level estimate: where wet-bulb = 32F (0C) in the vertical
    # Interpolate from profile
    wb_profile = [(p["altitude_ft"], p["wetbulb_f"]) for p in profile if p["altitude_ft"] > 0]
    wb_profile.sort(key=lambda x: x[0])
    snow_level_ft = None
    for i in range(len(wb_profile) - 1):
        alt1, wb1 = wb_profile[i]
        alt2, wb2 = wb_profile[i+1]
        if (wb1 >= 32 and wb2 < 32) or (wb1 <= 32 and wb2 > 32):
            # Linear interpolation
            if abs(wb2 - wb1) > 0.01:
                frac = (32.0 - wb1) / (wb2 - wb1)
                snow_level_ft = int(round(alt1 + frac * (alt2 - alt1)))
            break
    if snow_level_ft is None and wb_profile:
        # All above or below 32F
        if all(wb < 32 for _, wb in wb_profile):
            snow_level_ft = 0  # snow to sea level
        elif all(wb >= 32 for _, wb in wb_profile):
            snow_level_ft = int(round(wb_profile[-1][0]))  # above highest level
    result["snow_level_ft"] = snow_level_ft

    # Snowfall windows: hours where freezing level < 3500 ft (pass + 500ft buffer)
    PASS_ELEV = 3022
    snowfall_mask = fzl < (PASS_ELEV + 500) * 0.3048 / 0.3048  # already in ft
    snowfall_mask = fzl < (PASS_ELEV + 500)
    snowfall_hours_48h = int(snowfall_mask[fzl.index <= now_utc + pd.Timedelta(hours=48)].sum())
    snowfall_hours_120h = int(snowfall_mask[fzl.index <= now_utc + pd.Timedelta(hours=120)].sum())
    result["snowfall_possible_hours"] = {
        "next_48h": snowfall_hours_48h,
        "next_120h": snowfall_hours_120h,
    }

    # Snowmaking windows: surface wet-bulb < 28F (-2.2C)
    wb2m = sounding_df.groupby("time")["wetbulb_2m_c"].first().dropna()
    if not wb2m.empty:
        wb2m_f = wb2m * 9/5 + 32
        snowmaking_good = wb2m_f < 28  # good snowmaking
        snowmaking_marginal = wb2m_f < 32  # marginal
        next_48_wb = wb2m_f[wb2m_f.index <= now_utc + pd.Timedelta(hours=48)]
        result["snowmaking_windows"] = {
            "good_hours_48h": int(snowmaking_good[wb2m_f.index <= now_utc + pd.Timedelta(hours=48)].sum()),
            "marginal_hours_48h": int(snowmaking_marginal[wb2m_f.index <= now_utc + pd.Timedelta(hours=48)].sum()),
            "current_wetbulb_f": round(float(wb2m_f.iloc[0]), 1),
        }

    # 850 hPa wind (storm track direction indicator)
    wind_850 = sounding_df[sounding_df["level_hPa"] == 850].copy()
    if not wind_850.empty:
        next_48_wind = wind_850[wind_850["time"] <= now_utc + pd.Timedelta(hours=48)]
        if not next_48_wind.empty:
            mean_dir = next_48_wind["wind_dir"].mean()
            mean_spd = next_48_wind["wind_speed_kph"].mean()
            result["wind_850hPa_48h"] = {
                "mean_dir_deg": int(round(mean_dir)),
                "mean_speed_kph": int(round(mean_spd)),
                "mean_speed_mph": int(round(mean_spd * 0.621371)),
            }

    return result


# ── Summary report ────────────────────────────────────────────────────────────

def nowcast_summary(year: int = None, month: int = None) -> dict:
    """
    Generate a complete Layer 2 nowcast summary for the current month.
    """
    now = datetime.now(timezone.utc)
    if year is None:
        year = now.year
    if month is None:
        month = now.month

    summary = {"timestamp": now.isoformat(), "year": year, "month": month}

    # Current month pacing
    pace = current_month_pace(year, month)
    summary["pace"] = pace

    # Freezing level (latest 48h)
    try:
        fl = compute_freezing_level()
        if not fl.empty:
            cutoff = fl.index.max() - pd.Timedelta(hours=48)
            recent = fl[fl.index >= cutoff]
            summary["freezing_level"] = {
                "current_ft": round(recent["freezing_level_ft"].iloc[-1]),
                "avg_48h_ft": round(recent["freezing_level_ft"].mean()),
                "inversions_48h": int(recent["inversion"].sum()),
                "lapse_rate_avg": round(recent["lapse_rate_f_per_kft"].mean(), 1),
            }
    except Exception as e:
        summary["freezing_level"] = {"error": str(e)}

    # Pressure trends (latest)
    try:
        pres = compute_pressure_trends()
        if not pres.empty:
            latest = pres.dropna(subset=["slp_hpa"]).iloc[-1]
            summary["pressure"] = {
                "slp_hpa": round(latest["slp_hpa"], 1),
                "change_12h": round(latest["slp_12h_change"], 1) if pd.notna(latest["slp_12h_change"]) else None,
                "change_24h": round(latest["slp_24h_change"], 1) if pd.notna(latest["slp_24h_change"]) else None,
                "pattern": latest["pattern_12h"],
            }
    except Exception as e:
        summary["pressure"] = {"error": str(e)}

    # Open-Meteo vertical profile (forecast soundings)
    try:
        sounding = fetch_openmeteo_sounding(forecast_days=5)
        if not sounding.empty:
            summary["sounding"] = sounding_summary(sounding)
            # Save detailed sounding CSV for dashboard
            snd_path = os.path.join(DATA, "sounding_forecast.csv")
            sounding.to_csv(snd_path, index=False)
    except Exception as e:
        summary["sounding"] = {"error": str(e)}

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Layer 2 Nowcast — Snoqualmie Pass")
    print("=" * 60)

    summary = nowcast_summary()

    # Pacing
    pace = summary.get("pace", {})
    if "error" not in pace:
        print(f"\n  Current month pacing ({pace.get('station','?')}):")
        print(f"    Days elapsed: {pace['days_elapsed']} / {pace['days_in_month']}")
        print(f"    Snowfall to date: {pace['actual_snowfall_in']}\"")
        print(f"    On pace for: {pace['pace_snowfall_in']}\"")
        print(f"    Liquid precip: {pace.get('actual_precip_in', '?')}\"")
        print(f"    SWE gain est: {pace.get('swe_gain_est_in', '?')}\"")
        print(f"    Snow depth: {pace.get('depth_start_in','?')}\" -> {pace.get('depth_latest_in','?')}\"")
    else:
        print(f"\n  Pacing: {pace['error']}")

    # Freezing level
    fl = summary.get("freezing_level", {})
    if "error" not in fl and fl:
        print(f"\n  Freezing level:")
        print(f"    Current: {fl.get('current_ft', '?')}' ASL")
        print(f"    48h avg: {fl.get('avg_48h_ft', '?')}' ASL")
        print(f"    Inversions (48h): {fl.get('inversions_48h', '?')}")
        print(f"    Lapse rate: {fl.get('lapse_rate_avg', '?')} F/1000ft")

    # Pressure
    pres = summary.get("pressure", {})
    if "error" not in pres and pres:
        print(f"\n  Pressure (SNO30):")
        print(f"    SLP: {pres.get('slp_hpa', '?')} hPa")
        print(f"    12h change: {pres.get('change_12h', '?')} hPa -> {pres.get('pattern', '?')}")
        print(f"    24h change: {pres.get('change_24h', '?')} hPa")

    # Sounding / vertical profile
    snd = summary.get("sounding", {})
    if "error" not in snd and snd:
        print(f"\n  Forecast Sounding (Open-Meteo ECMWF+GFS):")

        # Freezing level forecast
        fzl = snd.get("freezing_level_forecast", {})
        if fzl:
            print(f"    Freezing level now: {fzl.get('current_ft', '?')}' ASL")
            print(f"    48h range: {fzl.get('min_48h_ft', '?')}' - {fzl.get('max_48h_ft', '?')}' "
                  f"(mean {fzl.get('mean_48h_ft', '?')}')")
            print(f"    5-day range: {fzl.get('min_120h_ft', '?')}' - {fzl.get('max_120h_ft', '?')}'")

        # Snow level
        sl = snd.get("snow_level_ft")
        if sl is not None:
            print(f"    Snow level (wet-bulb): {sl}' ASL")

        # Vertical profile table
        vp = snd.get("vertical_profile", [])
        if vp:
            print(f"\n    {'Level':>6}  {'Alt ft':>7}  {'Temp F':>6}  {'WetBulb':>7}  {'RH%':>4}  {'Wind':>10}")
            print(f"    {'------':>6}  {'-------':>7}  {'------':>6}  {'-------':>7}  {'----':>4}  {'----------':>10}")
            for p in vp:
                wdir = int(p['wind_dir'])
                wspd = int(p['wind_kph'] * 0.621371)
                print(f"    {p['level_hPa']:>4} hPa  {p['altitude_ft']:>7,}  {p['temp_f']:>6.1f}  {p['wetbulb_f']:>7.1f}  {p['rh_pct']:>3.0f}%  {wdir:>3} deg {wspd:>3} mph")

        # Snowfall / snowmaking windows
        sf = snd.get("snowfall_possible_hours", {})
        if sf:
            print(f"\n    Snowfall possible hours: {sf.get('next_48h', '?')}/48h, {sf.get('next_120h', '?')}/120h")
        sm = snd.get("snowmaking_windows", {})
        if sm:
            print(f"    Snowmaking (WB<28F): {sm.get('good_hours_48h', '?')}/48h | "
                  f"Marginal (WB<32F): {sm.get('marginal_hours_48h', '?')}/48h")
            print(f"    Current surface wet-bulb: {sm.get('current_wetbulb_f', '?')}F")

        # 850 hPa wind
        w850 = snd.get("wind_850hPa_48h", {})
        if w850:
            print(f"    850 hPa wind (48h avg): {w850.get('mean_dir_deg', '?')} deg "
                  f"@ {w850.get('mean_speed_mph', '?')} mph")
    elif "error" in snd:
        print(f"\n  Sounding: {snd['error']}")

    # Save nowcast_pace.json for forecast.py Layer 2 blending
    pace = summary.get("pace", {})
    pace_out = os.path.join(DATA, "nowcast_pace.json")
    pace_data = {}
    if os.path.exists(pace_out):
        try:
            with open(pace_out) as f:
                pace_data = json.load(f)
        except Exception:
            pass
    if "error" not in pace:
        pace_data[str(pace["month"])] = pace
    # Add sounding-derived forecast data (freezing level, snow level, windows)
    snd = summary.get("sounding", {})
    if "error" not in snd and snd:
        pace_data["sounding"] = {
            "freezing_level_forecast": snd.get("freezing_level_forecast"),
            "snow_level_ft": snd.get("snow_level_ft"),
            "snowfall_possible_hours": snd.get("snowfall_possible_hours"),
            "snowmaking_windows": snd.get("snowmaking_windows"),
            "wind_850hPa_48h": snd.get("wind_850hPa_48h"),
        }
    with open(pace_out, "w") as f:
        json.dump(pace_data, f, indent=2, default=str)
    print(f"\n  Saved pace -> {pace_out}")

    # Save full nowcast
    out = os.path.join(DATA, "nowcast.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved nowcast -> {out}")
