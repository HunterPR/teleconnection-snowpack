"""
Organize and standardize datasets for Snoqualmie Pass forecasting.

This script does not delete or overwrite source files. It creates:
  - data/processed/snoqualmie_daily_targets.csv
  - data/processed/buoy_daily_features.csv
  - data/processed/met_daily_features.csv
  - data/processed/ndbc_multi_daily_features.csv
  - data/processed/rwis_daily_features.csv
  - data/processed/synoptic_daily_features.csv
  - data/processed/teleconnection_monthly_features.csv
  - data/processed/nearby_snotel_monthly_features.csv
  - data/processed/streamflow_monthly_features.csv
  - data/processed/snoqualmie_model_daily.csv
  - data/processed/data_manifest.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
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

    agg = {
        "air_temp_set_1_F": ["mean", "min", "max"],
        "snow_depth_set_1_in": ["mean", "min", "max"],
        "precip_accum_one_hour_set_1_in": "sum",
        "snow_interval_set_1_in": "sum",
        "relative_humidity_set_1_%": "mean",
        "pressure_set_1d_mb": "mean",
        "sea_level_pressure_set_1d_mb": "mean",
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
        }
    )

    # Robust snowfall target: positive day-over-day change in daily max snow depth.
    if "target_max_snow_depth_in" in daily.columns:
        depth_delta = daily["target_max_snow_depth_in"].diff()
        daily["target_snowfall_24h_in"] = depth_delta.clip(lower=0)
        daily.loc[daily["target_snowfall_24h_in"] > MAX_DAILY_SNOWFALL_IN, "target_snowfall_24h_in"] = pd.NA
        if len(daily) > 0:
            daily.loc[daily.index[0], "target_snowfall_24h_in"] = pd.NA

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


def build_model_daily(
    daily_targets: pd.DataFrame,
    buoy_daily: pd.DataFrame,
    met_daily: pd.DataFrame,
    ndbc_daily: pd.DataFrame,
    rwis_daily: pd.DataFrame,
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
    buoy_daily = build_buoy_daily_features(buoy_source)
    met_daily = build_met_daily_features()
    ndbc_daily = build_ndbc_multi_daily_features()
    rwis_daily = build_rwis_daily_features()
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
        "synoptic_daily_features.csv": synoptic_daily,
        "teleconnection_monthly_features.csv": tele_monthly,
        "nearby_snotel_monthly_features.csv": snotel_monthly,
        "streamflow_monthly_features.csv": stream_monthly,
        "snoqualmie_model_daily.csv": model_daily,
    }

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
        {"layer": "output", "name": "synoptic_daily_features", "path": "data/processed/synoptic_daily_features.csv", "rows": str(len(synoptic_daily))},
        {"layer": "output", "name": "teleconnection_monthly_features", "path": "data/processed/teleconnection_monthly_features.csv", "rows": str(len(tele_monthly))},
        {"layer": "output", "name": "nearby_snotel_monthly_features", "path": "data/processed/nearby_snotel_monthly_features.csv", "rows": str(len(snotel_monthly))},
        {"layer": "output", "name": "streamflow_monthly_features", "path": "data/processed/streamflow_monthly_features.csv", "rows": str(len(stream_monthly))},
        {"layer": "output", "name": "model_daily", "path": "data/processed/snoqualmie_model_daily.csv", "rows": str(len(model_daily))},
    ]
    write_manifest(manifest)
    print("Saved data_manifest.csv")


if __name__ == "__main__":
    main()
