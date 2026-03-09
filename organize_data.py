"""
Organize and standardize datasets for Snoqualmie Pass forecasting.

This script does not delete or overwrite source files. It creates:
  - data/processed/snoqualmie_daily_targets.csv
  - data/processed/buoy_daily_features.csv
  - data/processed/met_daily_features.csv
  - data/processed/ndbc_multi_daily_features.csv
  - data/processed/rwis_daily_features.csv
  - data/processed/pipeline_wsdot_daily_features.csv
  - data/processed/pipeline_openmeteo_daily_features.csv
  - data/processed/pipeline_model_forecast_daily.csv
  - data/processed/custom_daily_features.csv
  - data/processed/custom_source_manifest.csv
  - data/processed/synoptic_daily_features.csv
  - data/processed/teleconnection_monthly_features.csv
  - data/processed/nearby_snotel_monthly_features.csv
  - data/processed/streamflow_monthly_features.csv
  - data/processed/snoqualmie_model_daily.csv
  - data/processed/data_manifest.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PIPELINE_DIR = DATA_DIR / "pipeline"
CUSTOM_DIR = DATA_DIR / "custom_sources"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MAX_HOURLY_SNOW_IN = 8.0
MAX_HOURLY_PRECIP_IN = 4.0
MAX_DAILY_SNOWFALL_IN = 40.0
MONTH_NAME_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def choose_existing(paths: Iterable[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"No source file found among: {[str(p) for p in paths]}")


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_date_series(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        out = out.dt.tz_convert(None)
    except Exception:
        pass
    return out.dt.floor("D")


def load_wide_monthly(path: Path, value_name: str, year_col: str = "year") -> pd.DataFrame:
    """
    Convert wide monthly tables (year + 12 month columns) into long format:
    year, month, <value_name>.
    """
    if not path.exists():
        return pd.DataFrame(columns=["year", "month", value_name])

    df = pd.read_csv(path, low_memory=False)
    if year_col not in df.columns:
        return pd.DataFrame(columns=["year", "month", value_name])

    keep_cols = [year_col]
    month_map: Dict[str, int] = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl == year_col.lower():
            continue
        for token, month_num in MONTH_NAME_TO_NUM.items():
            if token in cl:
                month_map[col] = month_num
                keep_cols.append(col)
                break

    if not month_map:
        return pd.DataFrame(columns=["year", "month", value_name])

    long_rows = []
    work = df[keep_cols].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    for _, row in work.iterrows():
        year = row[year_col]
        if pd.isna(year):
            continue
        year = int(year)
        for col, month in month_map.items():
            val = pd.to_numeric(row[col], errors="coerce")
            long_rows.append({"year": year, "month": int(month), value_name: val})

    out = pd.DataFrame(long_rows)
    out = out.dropna(subset=["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out


def build_snoqualmie_daily_targets(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    if "time" not in df.columns:
        raise ValueError(f"'time' column missing in {source}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df[df["time"].notna()].copy()
    df["date"] = df["time"].dt.date

    numeric_cols = [
        "air_temp_set_1_F",
        "snow_depth_set_1_in",
        "precip_accum_one_hour_set_1_in",
        "snow_interval_set_1_in",
        "relative_humidity_set_1_%",
        "pressure_set_1d_mb",
        "sea_level_pressure_set_1d_mb",
    ]
    to_numeric(df, numeric_cols)

    if "snow_interval_set_1_in" in df.columns:
        # Negative deltas can be sensor noise or compaction effects.
        df["snow_interval_set_1_in"] = df["snow_interval_set_1_in"].clip(lower=0)
        # Remove impossible spikes before daily aggregation.
        df.loc[df["snow_interval_set_1_in"] > MAX_HOURLY_SNOW_IN, "snow_interval_set_1_in"] = pd.NA
    if "precip_accum_one_hour_set_1_in" in df.columns:
        df["precip_accum_one_hour_set_1_in"] = df["precip_accum_one_hour_set_1_in"].clip(lower=0)
        df.loc[df["precip_accum_one_hour_set_1_in"] > MAX_HOURLY_PRECIP_IN, "precip_accum_one_hour_set_1_in"] = pd.NA

    # Derive practical weather-operation metrics (snowmaking and snowfall potential)
    # from hourly station observations where available.
    if {"air_temp_set_1_F", "relative_humidity_set_1_%"}.issubset(df.columns):
        temp_c = (df["air_temp_set_1_F"] - 32.0) * (5.0 / 9.0)
        rh = df["relative_humidity_set_1_%"].clip(lower=0, upper=100)
        # Stull (2011) wet-bulb approximation (C); accurate enough for ops thresholds.
        wet_bulb_c = (
            temp_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(temp_c + rh)
            - np.arctan(rh - 1.676331)
            + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
            - 4.686035
        )
        df["wet_bulb_set_1_f"] = (wet_bulb_c * 9.0 / 5.0) + 32.0
        df["snowmaking_good_hour"] = (
            (df["air_temp_set_1_F"] <= 32.0) & (df["wet_bulb_set_1_f"] <= 28.0)
        ).astype(int)
        df["snowmaking_marginal_hour"] = (
            (df["air_temp_set_1_F"] <= 34.0) & (df["wet_bulb_set_1_f"] <= 30.0)
        ).astype(int)
        df["snowfall_possible_hour"] = (
            (df["air_temp_set_1_F"] <= 34.0) & (df["relative_humidity_set_1_%"] >= 70.0)
        ).astype(int)
        if "precip_accum_one_hour_set_1_in" in df.columns:
            df["snowfall_likely_hour"] = (
                (df["air_temp_set_1_F"] <= 34.0)
                & (df["relative_humidity_set_1_%"] >= 70.0)
                & (df["precip_accum_one_hour_set_1_in"] > 0)
            ).astype(int)

    agg = {
        "air_temp_set_1_F": ["mean", "min", "max"],
        "snow_depth_set_1_in": ["mean", "min", "max"],
        "precip_accum_one_hour_set_1_in": "sum",
        "snow_interval_set_1_in": "sum",
        "relative_humidity_set_1_%": "mean",
        "pressure_set_1d_mb": "mean",
        "sea_level_pressure_set_1d_mb": "mean",
        "wet_bulb_set_1_f": ["mean", "min", "max"],
        "snowmaking_good_hour": "sum",
        "snowmaking_marginal_hour": "sum",
        "snowfall_possible_hour": "sum",
        "snowfall_likely_hour": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    daily = df.groupby("date", as_index=False).agg(agg)

    daily.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in daily.columns
    ]
    daily = daily.rename(
        columns={
            "date": "date",
            "air_temp_set_1_F_mean": "target_mean_temp_f",
            "air_temp_set_1_F_min": "target_min_temp_f",
            "air_temp_set_1_F_max": "target_max_temp_f",
            "snow_depth_set_1_in_mean": "target_mean_snow_depth_in",
            "snow_depth_set_1_in_min": "target_min_snow_depth_in",
            "snow_depth_set_1_in_max": "target_max_snow_depth_in",
            "precip_accum_one_hour_set_1_in_sum": "target_precip_24h_in",
            "snow_interval_set_1_in_sum": "raw_snow_interval_24h_in",
            "relative_humidity_set_1_%_mean": "target_mean_rh_pct",
            "pressure_set_1d_mb_mean": "target_mean_pressure_mb",
            "sea_level_pressure_set_1d_mb_mean": "target_mean_slp_mb",
            "wet_bulb_set_1_f_mean": "target_mean_wet_bulb_f",
            "wet_bulb_set_1_f_min": "target_min_wet_bulb_f",
            "wet_bulb_set_1_f_max": "target_max_wet_bulb_f",
            "snowmaking_good_hour_sum": "target_snowmaking_good_hours_24h",
            "snowmaking_marginal_hour_sum": "target_snowmaking_marginal_hours_24h",
            "snowfall_possible_hour_sum": "target_snowfall_possible_hours_24h",
            "snowfall_likely_hour_sum": "target_snowfall_likely_hours_24h",
        }
    )

    # Robust snowfall target: positive day-over-day change in daily max snow depth.
    if "target_max_snow_depth_in" in daily.columns:
        depth_delta = daily["target_max_snow_depth_in"].diff()
        daily["target_snowfall_24h_in"] = depth_delta.clip(lower=0)
        daily.loc[daily["target_snowfall_24h_in"] > MAX_DAILY_SNOWFALL_IN, "target_snowfall_24h_in"] = pd.NA
        if len(daily) > 0:
            daily.loc[daily.index[0], "target_snowfall_24h_in"] = pd.NA

    # Kuchera-ratio snowfall estimate from liquid precip + mean temp (for reference; not a substitute for depth-delta target).
    if "target_precip_24h_in" in daily.columns and "target_mean_temp_f" in daily.columns:
        try:
            from snow_ratio import liquid_to_snow_inches
            prec = pd.to_numeric(daily["target_precip_24h_in"], errors="coerce").fillna(0)
            temp = pd.to_numeric(daily["target_mean_temp_f"], errors="coerce")
            temp = temp.fillna(30.0)
            daily["target_snowfall_kuchera_in"] = liquid_to_snow_inches(prec.values, temp.values)
        except Exception:
            pass

    if "air_temp_set_1_F" in df.columns:
        freezing_hours = (
            df.assign(is_freezing=df["air_temp_set_1_F"] <= 32.0)
            .groupby("date", as_index=False)["is_freezing"]
            .sum()
            .rename(columns={"is_freezing": "target_freezing_hours_24h"})
        )
        daily = daily.merge(freezing_hours, on="date", how="left")

    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


# Snoqualmie Pass station IDs used for multi-station snowfall (priority order).
# ALP31 and SNO30 get higher weight when present (pass-representative).
SNOQUALMIE_PASS_STATION_STEMS = ["sno38", "alp31", "alp44", "alp55", "alp43", "sno30"]
# Weights for weighted-mean pass snowfall: ALP31 and SNO30 emphasized when available.
PASS_SNOWFALL_WEIGHTS = {"alp31": 1.5, "sno30": 1.5}
DEFAULT_PASS_SNOWFALL_WEIGHT = 1.0


def _detect_snow_column(df: pd.DataFrame) -> str | None:
    """Return first column that looks like snow depth or new snow (for daily snowfall derivation)."""
    candidates = [
        "snow_depth_set_1_in",
        "snow_depth_in",
        "snow_depth",
        "snow_interval_set_1_in",
        "snow_interval_in",
        "new_snow_in",
        "new_snow",
        "depth_in",
        "depth",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for key in ["snow_depth", "snow_interval", "new_snow", "snowfall"]:
        if key in lowered:
            return str(lowered[key])
    for col in df.columns:
        cl = str(col).strip().lower()
        if "snow" in cl and ("depth" in cl or "interval" in cl or "new" in cl):
            return str(col)
    return None


def build_snoqualmie_pass_snowfall_from_stations() -> pd.DataFrame | None:
    """
    Build a single pass-representative daily snowfall series from multiple stations
    (SNO38, ALP31, ALP44, ALP55, SNO30) when CSVs are present in data/custom_sources/.

    Expects files named e.g. sno38.csv, alp31.csv with a date/datetime column and a
    snow depth or new-snow column. Daily snowfall = day-over-day increase in daily
    max snow depth (or sum of interval/new_snow if that column is used). Combines
    stations by taking the mean of available values per date (extrapolation to pass).
    Returns None if no station files found or no valid series.
    """
    if not CUSTOM_DIR.exists():
        return None
    station_dailies: List[pd.DataFrame] = []
    for stem in SNOQUALMIE_PASS_STATION_STEMS:
        path = CUSTOM_DIR / f"{stem}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        dt_col = _detect_datetime_column(df)
        snow_col = _detect_snow_column(df)
        if dt_col is None or snow_col is None:
            continue
        work = df.copy()
        work["date"] = normalize_date_series(work[dt_col])
        work = work[work["date"].notna()].copy()
        work[snow_col] = pd.to_numeric(work[snow_col], errors="coerce")
        work = work[work[snow_col].notna()].copy()
        if work.empty:
            continue
        cl = str(snow_col).lower()
        if "interval" in cl or "new_snow" in cl or "new " in cl:
            daily = work.groupby("date", as_index=False)[snow_col].sum()
            daily = daily.rename(columns={snow_col: "snowfall_in"})
        else:
            daily_max = work.groupby("date", as_index=False)[snow_col].max()
            daily_max = daily_max.sort_values("date").reset_index(drop=True)
            daily_max["snowfall_in"] = daily_max[snow_col].diff().clip(lower=0)
            daily_max.loc[daily_max["snowfall_in"] > MAX_DAILY_SNOWFALL_IN, "snowfall_in"] = pd.NA
            if len(daily_max) > 0:
                daily_max.loc[daily_max.index[0], "snowfall_in"] = pd.NA
            daily = daily_max[["date", "snowfall_in"]].copy()
        daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
        daily = daily.dropna(subset=["snowfall_in"]).copy()
        if daily.empty:
            continue
        daily = daily.rename(columns={"snowfall_in": stem})
        station_dailies.append(daily)
    if not station_dailies:
        return None
    merged = station_dailies[0].copy()
    first_stem = merged.columns[1]
    for tbl in station_dailies[1:]:
        stem = tbl.columns[1]
        merged = merged.merge(tbl, on="date", how="outer")
    snow_cols = [c for c in merged.columns if c in SNOQUALMIE_PASS_STATION_STEMS]
    if not snow_cols:
        snow_cols = [first_stem]
    # Weighted mean: ALP31 and SNO30 get higher weight (pass-representative).
    def weighted_mean(row):
        total, denom = 0.0, 0.0
        for c in snow_cols:
            v = row[c]
            if pd.notna(v):
                w = PASS_SNOWFALL_WEIGHTS.get(c.lower(), DEFAULT_PASS_SNOWFALL_WEIGHT)
                total += v * w
                denom += w
        return total / denom if denom > 0 else np.nan

    merged["target_snowfall_24h_in"] = merged.apply(weighted_mean, axis=1)
    merged["n_stations_snowfall"] = merged[snow_cols].notna().sum(axis=1)
    out = merged[["date", "target_snowfall_24h_in", "n_stations_snowfall"]].copy()
    out = out.sort_values("date").reset_index(drop=True)
    return out


def build_buoy_daily_features(source: Path) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    if "time" not in df.columns:
        raise ValueError(f"'time' column missing in {source}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df[df["time"].notna()].copy()
    df["date"] = df["time"].dt.date

    numeric_cols = ["max_wspd", "max_gst", "max_wvht", "min_pres", "WSPD", "GST", "WVHT", "PRES"]
    to_numeric(df, numeric_cols)

    # Support both "already-daily max_" buoy files and raw hourly buoy files.
    feature_specs: Dict[str, List[str]] = {
        "max_wspd": ["mean", "max"],
        "max_gst": ["mean", "max"],
        "max_wvht": ["mean", "max"],
        "min_pres": ["mean", "min"],
        "WSPD": ["mean", "max"],
        "GST": ["mean", "max"],
        "WVHT": ["mean", "max"],
        "PRES": ["mean", "min"],
    }
    use_specs = {k: v for k, v in feature_specs.items() if k in df.columns}
    if not use_specs:
        raise ValueError(f"No buoy numeric columns found in {source}")

    daily = df.groupby("date", as_index=False).agg(use_specs)
    daily.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in daily.columns
    ]
    rename = {c: f"buoy_{c}" for c in daily.columns if c != "date"}
    daily = daily.rename(columns=rename)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def build_met_daily_features() -> pd.DataFrame:
    path = DATA_DIR / "met_daily_features.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_ndbc_multi_daily_features() -> pd.DataFrame:
    candidates = [
        DATA_DIR / "ndbc_historical_daily_features.csv",
        DATA_DIR / "ndbc_multi_daily_features.csv",
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_rwis_daily_features() -> pd.DataFrame:
    path = DATA_DIR / "rwis_daily_features.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_synoptic_daily_features() -> pd.DataFrame:
    path = DATA_DIR / "synoptic_daily_features.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_pipeline_wsdot_daily_features() -> pd.DataFrame:
    """
    Load optional WSDOT daily summaries produced by build_snoqualmie_weather_pipeline.py
    and collapse station rows into regional daily features.
    """
    path = PIPELINE_DIR / "wsdot_daily_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    numeric_cols = [
        "temp_f_mean",
        "temp_f_min",
        "temp_f_max",
        "precip_in_sum",
        "rh_pct_mean",
        "pressure_mean",
        "wind_mph_mean",
        "wind_gust_mph_max",
    ]
    to_numeric(df, numeric_cols)
    agg = {
        "temp_f_mean": "mean",
        "temp_f_min": "mean",
        "temp_f_max": "mean",
        "precip_in_sum": "mean",
        "rh_pct_mean": "mean",
        "pressure_mean": "mean",
        "wind_mph_mean": "mean",
        "wind_gust_mph_max": "mean",
        "station_id": "nunique",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    daily = df.groupby("date", as_index=False).agg(agg)
    rename = {
        "temp_f_mean": "pipe_wsdot_temp_f_mean",
        "temp_f_min": "pipe_wsdot_temp_f_min",
        "temp_f_max": "pipe_wsdot_temp_f_max",
        "precip_in_sum": "pipe_wsdot_precip_in_sum",
        "rh_pct_mean": "pipe_wsdot_rh_pct_mean",
        "pressure_mean": "pipe_wsdot_pressure_mean",
        "wind_mph_mean": "pipe_wsdot_wind_mph_mean",
        "wind_gust_mph_max": "pipe_wsdot_wind_gust_mph_max",
        "station_id": "pipe_wsdot_station_count",
    }
    daily = daily.rename(columns=rename)
    return daily.sort_values("date").reset_index(drop=True)


def build_pipeline_openmeteo_daily_features() -> pd.DataFrame:
    """
    Load optional Open-Meteo station history produced by
    build_snoqualmie_weather_pipeline.py and aggregate by date.
    """
    path = PIPELINE_DIR / "openmeteo_station_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    numeric_cols = [
        "temp_c_mean",
        "temp_c_min",
        "temp_c_max",
        "precip_mm_sum",
        "wind_kph_mean",
        "pressure_hpa_mean",
    ]
    to_numeric(df, numeric_cols)
    agg = {
        "temp_c_mean": "mean",
        "temp_c_min": "mean",
        "temp_c_max": "mean",
        "precip_mm_sum": "sum",
        "wind_kph_mean": "mean",
        "pressure_hpa_mean": "mean",
        "station_id": "nunique",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}
    daily = df.groupby("date", as_index=False).agg(agg)
    rename = {
        "temp_c_mean": "pipe_openmeteo_temp_c_mean",
        "temp_c_min": "pipe_openmeteo_temp_c_min",
        "temp_c_max": "pipe_openmeteo_temp_c_max",
        "precip_mm_sum": "pipe_openmeteo_precip_mm_sum",
        "wind_kph_mean": "pipe_openmeteo_wind_kph_mean",
        "pressure_hpa_mean": "pipe_openmeteo_pressure_hpa_mean",
        "station_id": "pipe_openmeteo_station_count",
    }
    daily = daily.rename(columns=rename)
    return daily.sort_values("date").reset_index(drop=True)


def build_pipeline_model_forecast_daily() -> pd.DataFrame:
    """
    Load optional model forecast daily file from the unified pipeline.
    """
    path = PIPELINE_DIR / "model_forecast_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["date", "model"])
    df = pd.read_csv(path, low_memory=False)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["date", "model"])
    df["date"] = normalize_date_series(df["date"])
    df = df[df["date"].notna()].copy()
    numeric_cols = [
        "temp_c_mean",
        "temp_c_min",
        "temp_c_max",
        "precip_mm_sum",
        "wind_kph_mean",
        "pressure_hpa_mean",
        "freezing_level_m_mean",
    ]
    to_numeric(df, numeric_cols)
    return df.sort_values(["date", "model"]).reset_index(drop=True)


def _sanitize_token(name: str, limit: int = 40) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")
    if not token:
        token = "src"
    return token[:limit]


def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "date",
        "datetime",
        "time",
        "timestamp",
        "readingtime",
        "observation_time",
    ]
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        if key in lowered:
            return str(lowered[key])
    for col in df.columns:
        cl = str(col).strip().lower()
        if "date" in cl or "time" in cl:
            return str(col)
    return None


def _agg_kind(col: str) -> str:
    cl = str(col).lower()
    if "min" in cl:
        return "min"
    if "max" in cl:
        return "max"
    sum_tokens = ["precip", "snow", "rain", "accum", "amount", "total", "new_snow"]
    if any(t in cl for t in sum_tokens):
        return "sum"
    return "mean"


def build_custom_daily_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingest arbitrary CSV files from data/custom_sources and convert them into
    normalized daily features keyed by date.
    """
    if not CUSTOM_DIR.exists():
        return pd.DataFrame(columns=["date"]), pd.DataFrame(columns=["source_file", "rows_in", "rows_daily", "date_column"])

    csv_paths = sorted(CUSTOM_DIR.rglob("*.csv"))
    if not csv_paths:
        return pd.DataFrame(columns=["date"]), pd.DataFrame(columns=["source_file", "rows_in", "rows_daily", "date_column"])

    merged: pd.DataFrame | None = None
    manifest_rows: List[Dict[str, str]] = []
    for path in csv_paths:
        try:
            src = pd.read_csv(path, low_memory=False)
        except Exception:
            manifest_rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "rows_in": "0",
                    "rows_daily": "0",
                    "date_column": "",
                    "status": "read_error",
                }
            )
            continue
        if src.empty:
            continue

        dt_col = _detect_datetime_column(src)
        if dt_col is None:
            manifest_rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "rows_in": str(len(src)),
                    "rows_daily": "0",
                    "date_column": "",
                    "status": "missing_datetime_column",
                }
            )
            continue

        work = src.copy()
        work["date"] = normalize_date_series(work[dt_col])
        work = work[work["date"].notna()].copy()
        if work.empty:
            manifest_rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "rows_in": str(len(src)),
                    "rows_daily": "0",
                    "date_column": dt_col,
                    "status": "no_valid_dates",
                }
            )
            continue

        candidate_cols = [c for c in work.columns if c not in {dt_col, "date"}]
        numeric_cols: List[str] = []
        for col in candidate_cols:
            series = pd.to_numeric(work[col], errors="coerce")
            if series.notna().sum() >= 10:
                work[col] = series
                numeric_cols.append(col)
        if not numeric_cols:
            manifest_rows.append(
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "rows_in": str(len(src)),
                    "rows_daily": "0",
                    "date_column": dt_col,
                    "status": "no_numeric_columns",
                }
            )
            continue

        agg_map = {col: _agg_kind(col) for col in numeric_cols}
        daily = work.groupby("date", as_index=False).agg(agg_map)
        source_prefix = _sanitize_token(path.stem)
        rename_map = {col: f"custom_{source_prefix}_{_sanitize_token(col, limit=32)}" for col in numeric_cols}
        daily = daily.rename(columns=rename_map)
        daily = daily.sort_values("date").reset_index(drop=True)

        merged = daily if merged is None else merged.merge(daily, on="date", how="outer")
        manifest_rows.append(
            {
                "source_file": str(path.relative_to(ROOT)),
                "rows_in": str(len(src)),
                "rows_daily": str(len(daily)),
                "date_column": dt_col,
                "status": "ok",
            }
        )

    if merged is None:
        merged = pd.DataFrame(columns=["date"])
    manifest = pd.DataFrame(manifest_rows)
    return merged.sort_values("date").reset_index(drop=True), manifest


def season_to_month(season: str) -> int:
    season = (season or "").strip().upper()
    mapping = {
        "DJF": 1,
        "JFM": 2,
        "FMA": 3,
        "MAM": 4,
        "AMJ": 5,
        "MJJ": 6,
        "JJA": 7,
        "JAS": 8,
        "ASO": 9,
        "SON": 10,
        "OND": 11,
        "NDJ": 12,
    }
    return mapping.get(season, pd.NA)


def build_teleconnection_monthly_features() -> pd.DataFrame:
    tele_paths = {
        "ao": DATA_DIR / "ao.csv",
        "nao": DATA_DIR / "nao.csv",
        "pna": DATA_DIR / "pna.csv",
        "pdo": DATA_DIR / "pdo.csv",
        "oni": DATA_DIR / "oni.csv",
        "mjo": DATA_DIR / "mjo_rmm.csv",
    }

    parts = []
    for key, path in tele_paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if key == "oni":
            if {"year", "season", "oni_anomaly"}.issubset(df.columns):
                df["month"] = df["season"].apply(season_to_month)
                df = df[["year", "month", "oni_anomaly"]]
                df["oni_anomaly"] = pd.to_numeric(df["oni_anomaly"], errors="coerce")
                df = df[df["month"].notna()]
                df["month"] = df["month"].astype(int)
                parts.append(df)
        elif key == "mjo":
            needed = {"year", "month", "rmm1", "rmm2", "amplitude"}
            if needed.issubset(df.columns):
                for col in ["rmm1", "rmm2", "amplitude"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                m = (
                    df.groupby(["year", "month"], as_index=False)[["rmm1", "rmm2", "amplitude"]]
                    .mean()
                    .rename(
                        columns={
                            "rmm1": "mjo_rmm1_mean",
                            "rmm2": "mjo_rmm2_mean",
                            "amplitude": "mjo_amplitude_mean",
                        }
                    )
                )
                parts.append(m)
        else:
            val_col = key
            if {"year", "month", val_col}.issubset(df.columns):
                df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
                parts.append(df[["year", "month", val_col]])

    # Additional transformed indices already available in workspace.
    extra_base = DATA_DIR / "PSL CSV Files"
    extras = [
        ("qbo", extra_base / "transformed_qbo.csv"),
        ("np", extra_base / "transformed_np.csv"),
        ("pmm", extra_base / "transformed_pmm.csv"),
        ("solar", extra_base / "transformed_solar.csv"),
        ("wp", extra_base / "wp.csv"),
    ]
    for name, path in extras:
        extra = load_wide_monthly(path, name)
        if not extra.empty:
            parts.append(extra)

    if not parts:
        return pd.DataFrame(columns=["year", "month"])

    out = parts[0].copy()
    for p in parts[1:]:
        out = out.merge(p, on=["year", "month"], how="outer")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year", "month"]).copy()
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    out = out.sort_values(["year", "month"]).reset_index(drop=True)
    return out


def build_streamflow_monthly_features() -> pd.DataFrame:
    candidates = [
        ROOT / "monthly_streamflow_usgs_1950_2024.csv",
        ROOT / "monthly_bor_streamflow.csv",
    ]
    monthly_frames = []
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if {"year", "month"}.issubset(df.columns):
            monthly_frames.append(df)
    if not monthly_frames:
        return pd.DataFrame(columns=["year", "month"])

    out = monthly_frames[0].copy()
    for f in monthly_frames[1:]:
        overlap = [c for c in f.columns if c in out.columns and c not in {"year", "month"}]
        if overlap:
            f = f.rename(columns={c: f"{c}_bor" for c in overlap})
        out = out.merge(f, on=["year", "month"], how="outer")
    out = out.sort_values(["year", "month"]).reset_index(drop=True)
    return out


def build_nearby_snotel_monthly_features() -> pd.DataFrame:
    station_files = [
        ("stampede", ROOT / "stampede_monthly_snotel.csv"),
        ("olallie", ROOT / "olallie_monthly_snotel.csv"),
        ("tinkham", ROOT / "tinkhamcreek_monthly_snotel.csv"),
    ]
    parts = []
    for station_name, path in station_files:
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        if not {"year", "month"}.issubset(df.columns):
            continue
        rename_cols = {
            c: f"{station_name}_{c.lower()}"
            for c in df.columns
            if c not in {"year", "month"}
        }
        df = df.rename(columns=rename_cols)
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["year", "month"])

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["year", "month"], how="outer")
    out = out.sort_values(["year", "month"]).reset_index(drop=True)
    return out


def add_lag_and_rolling_features(
    df: pd.DataFrame, columns: List[str], lags: List[int], windows: List[int]
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("date").copy()
    for col in columns:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for lag in lags:
            out[f"{col}_lag{lag}d"] = series.shift(lag)
        for win in windows:
            out[f"{col}_roll{win}d_mean"] = series.rolling(win, min_periods=max(2, win // 2)).mean()
            out[f"{col}_roll{win}d_max"] = series.rolling(win, min_periods=max(2, win // 2)).max()
    return out


def build_model_daily(
    daily_targets: pd.DataFrame,
    buoy_daily: pd.DataFrame,
    met_daily: pd.DataFrame,
    ndbc_daily: pd.DataFrame,
    rwis_daily: pd.DataFrame,
    pipeline_wsdot_daily: pd.DataFrame,
    pipeline_openmeteo_daily: pd.DataFrame,
    custom_daily: pd.DataFrame,
    synoptic_daily: pd.DataFrame,
    tele_monthly: pd.DataFrame,
    stream_monthly: pd.DataFrame,
    snotel_monthly: pd.DataFrame,
) -> pd.DataFrame:
    model = daily_targets.copy()
    model = model.merge(buoy_daily, on="date", how="left")
    if not met_daily.empty:
        model = model.merge(met_daily, on="date", how="left")
    if not ndbc_daily.empty:
        model = model.merge(ndbc_daily, on="date", how="left")
    if not rwis_daily.empty:
        model = model.merge(rwis_daily, on="date", how="left")
    if not pipeline_wsdot_daily.empty:
        model = model.merge(pipeline_wsdot_daily, on="date", how="left")
    if not pipeline_openmeteo_daily.empty:
        model = model.merge(pipeline_openmeteo_daily, on="date", how="left")
    if not custom_daily.empty:
        model = model.merge(custom_daily, on="date", how="left")
    if not synoptic_daily.empty:
        model = model.merge(synoptic_daily, on="date", how="left")
    model["year"] = model["date"].dt.year
    model["month"] = model["date"].dt.month
    if not tele_monthly.empty:
        model = model.merge(tele_monthly, on=["year", "month"], how="left")
    if not snotel_monthly.empty:
        model = model.merge(snotel_monthly, on=["year", "month"], how="left")
    if not stream_monthly.empty:
        model = model.merge(stream_monthly, on=["year", "month"], how="left")

    # Add lag/rolling marine features so forecast tools can exploit lead-lag signals.
    buoy_core = [
        "buoy_max_wspd_mean",
        "buoy_max_gst_mean",
        "buoy_max_wvht_mean",
        "buoy_min_pres_mean",
    ]
    model = add_lag_and_rolling_features(
        model,
        columns=buoy_core,
        lags=[1, 2, 3, 5, 7, 10, 14],
        windows=[3, 7],
    )

    # Drop columns that are entirely missing in the merged date range.
    all_nan_cols = [c for c in model.columns if c != "date" and model[c].isna().all()]
    if all_nan_cols:
        model = model.drop(columns=all_nan_cols)
    model = model.sort_values("date").reset_index(drop=True)
    return model


def write_manifest(entries: List[Dict[str, str]]) -> None:
    pd.DataFrame(entries).to_csv(PROCESSED_DIR / "data_manifest.csv", index=False)


def main() -> None:
    sno_source = choose_existing([ROOT / "sno30_manual_clean.csv", ROOT / "sno30_for_model.csv"])
    buoy_source = choose_existing([ROOT / "buoy_fixed.csv", ROOT / "buoy_1975_present_extended.csv", ROOT / "buoy_clean.csv"])

    daily_targets = build_snoqualmie_daily_targets(sno_source)
    pass_snowfall = build_snoqualmie_pass_snowfall_from_stations()
    pass_monthly = pd.DataFrame(columns=["year", "month", "snow_inches_pass"])
    if pass_snowfall is not None and not pass_snowfall.empty:
        pass_snowfall = pass_snowfall.copy()
        pass_snowfall["date"] = pd.to_datetime(pass_snowfall["date"]).dt.normalize()
        daily_targets["date"] = pd.to_datetime(daily_targets["date"]).dt.normalize()
        daily_targets = daily_targets.merge(
            pass_snowfall[["date", "target_snowfall_24h_in", "n_stations_snowfall"]],
            on="date",
            how="left",
            suffixes=("", "_pass"),
        )
        if "target_snowfall_24h_in_pass" in daily_targets.columns:
            daily_targets["target_snowfall_24h_in"] = daily_targets["target_snowfall_24h_in_pass"].combine_first(
                daily_targets["target_snowfall_24h_in"]
            )
            daily_targets = daily_targets.drop(columns=["target_snowfall_24h_in_pass"])
        if "n_stations_snowfall" in daily_targets.columns:
            daily_targets["n_stations_snowfall"] = daily_targets["n_stations_snowfall"].fillna(0).astype(int)
        print(f"  Snoqualmie Pass snowfall: merged multi-station series ({pass_snowfall['date'].min()} to {pass_snowfall['date'].max()}, {len(pass_snowfall)} days)")
        # Monthly aggregate for forecast pipeline (DOT/ALP pass-first snow_inches)
        pass_snowfall["date"] = pd.to_datetime(pass_snowfall["date"])
        pass_snowfall["year"] = pass_snowfall["date"].dt.year
        pass_snowfall["month"] = pass_snowfall["date"].dt.month
        pass_monthly = (
            pass_snowfall.groupby(["year", "month"], as_index=False)["target_snowfall_24h_in"]
            .sum()
            .rename(columns={"target_snowfall_24h_in": "snow_inches_pass"})
        )
        pass_monthly["year"] = pass_monthly["year"].astype(int)
        pass_monthly["month"] = pass_monthly["month"].astype(int)
    buoy_daily = build_buoy_daily_features(buoy_source)
    met_daily = build_met_daily_features()
    ndbc_daily = build_ndbc_multi_daily_features()
    rwis_daily = build_rwis_daily_features()
    pipeline_wsdot_daily = build_pipeline_wsdot_daily_features()
    pipeline_openmeteo_daily = build_pipeline_openmeteo_daily_features()
    pipeline_forecast_daily = build_pipeline_model_forecast_daily()
    custom_daily, custom_manifest = build_custom_daily_features()
    synoptic_daily = build_synoptic_daily_features()
    tele_monthly = build_teleconnection_monthly_features()
    snotel_monthly = build_nearby_snotel_monthly_features()
    stream_monthly = build_streamflow_monthly_features()
    model_daily = build_model_daily(
        daily_targets,
        buoy_daily,
        met_daily,
        ndbc_daily,
        rwis_daily,
        pipeline_wsdot_daily,
        pipeline_openmeteo_daily,
        custom_daily,
        synoptic_daily,
        tele_monthly,
        stream_monthly,
        snotel_monthly,
    )

    outputs = {
        "snoqualmie_daily_targets.csv": daily_targets,
        "buoy_daily_features.csv": buoy_daily,
        "met_daily_features.csv": met_daily,
        "ndbc_multi_daily_features.csv": ndbc_daily,
        "rwis_daily_features.csv": rwis_daily,
        "pipeline_wsdot_daily_features.csv": pipeline_wsdot_daily,
        "pipeline_openmeteo_daily_features.csv": pipeline_openmeteo_daily,
        "pipeline_model_forecast_daily.csv": pipeline_forecast_daily,
        "custom_daily_features.csv": custom_daily,
        "custom_source_manifest.csv": custom_manifest,
        "synoptic_daily_features.csv": synoptic_daily,
        "teleconnection_monthly_features.csv": tele_monthly,
        "nearby_snotel_monthly_features.csv": snotel_monthly,
        "streamflow_monthly_features.csv": stream_monthly,
        "snoqualmie_model_daily.csv": model_daily,
    }
    if not pass_monthly.empty:
        outputs["pass_monthly_snowfall.csv"] = pass_monthly

    for name, df in outputs.items():
        out_path = PROCESSED_DIR / name
        df.to_csv(out_path, index=False)
        print(f"Saved {name}: {len(df)} rows x {len(df.columns)} cols")

    manifest = [
        {"layer": "source", "name": "snoqualmie_hourly_source", "path": str(sno_source.relative_to(ROOT)), "rows": str(len(pd.read_csv(sno_source, low_memory=False)))},
        {"layer": "source", "name": "buoy_source", "path": str(buoy_source.relative_to(ROOT)), "rows": str(len(pd.read_csv(buoy_source, low_memory=False)))},
        {"layer": "output", "name": "daily_targets", "path": "data/processed/snoqualmie_daily_targets.csv", "rows": str(len(daily_targets))},
        {"layer": "output", "name": "buoy_daily_features", "path": "data/processed/buoy_daily_features.csv", "rows": str(len(buoy_daily))},
        {"layer": "output", "name": "met_daily_features", "path": "data/processed/met_daily_features.csv", "rows": str(len(met_daily))},
        {"layer": "output", "name": "ndbc_multi_daily_features", "path": "data/processed/ndbc_multi_daily_features.csv", "rows": str(len(ndbc_daily))},
        {"layer": "output", "name": "rwis_daily_features", "path": "data/processed/rwis_daily_features.csv", "rows": str(len(rwis_daily))},
        {"layer": "output", "name": "pipeline_wsdot_daily_features", "path": "data/processed/pipeline_wsdot_daily_features.csv", "rows": str(len(pipeline_wsdot_daily))},
        {"layer": "output", "name": "pipeline_openmeteo_daily_features", "path": "data/processed/pipeline_openmeteo_daily_features.csv", "rows": str(len(pipeline_openmeteo_daily))},
        {"layer": "output", "name": "pipeline_model_forecast_daily", "path": "data/processed/pipeline_model_forecast_daily.csv", "rows": str(len(pipeline_forecast_daily))},
        {"layer": "output", "name": "custom_daily_features", "path": "data/processed/custom_daily_features.csv", "rows": str(len(custom_daily))},
        {"layer": "output", "name": "custom_source_manifest", "path": "data/processed/custom_source_manifest.csv", "rows": str(len(custom_manifest))},
        {"layer": "output", "name": "synoptic_daily_features", "path": "data/processed/synoptic_daily_features.csv", "rows": str(len(synoptic_daily))},
        {"layer": "output", "name": "teleconnection_monthly_features", "path": "data/processed/teleconnection_monthly_features.csv", "rows": str(len(tele_monthly))},
        {"layer": "output", "name": "nearby_snotel_monthly_features", "path": "data/processed/nearby_snotel_monthly_features.csv", "rows": str(len(snotel_monthly))},
        {"layer": "output", "name": "streamflow_monthly_features", "path": "data/processed/streamflow_monthly_features.csv", "rows": str(len(stream_monthly))},
        {"layer": "output", "name": "model_daily", "path": "data/processed/snoqualmie_model_daily.csv", "rows": str(len(model_daily))},
    ]
    if not pass_monthly.empty:
        manifest.append({"layer": "output", "name": "pass_monthly_snowfall", "path": "data/processed/pass_monthly_snowfall.csv", "rows": str(len(pass_monthly))})
    write_manifest(manifest)
    print("Saved data_manifest.csv")


if __name__ == "__main__":
    main()
