"""
Backfill historical NDBC marine predictors for Snoqualmie forecasting.

Outputs:
  - data/ndbc_historical_daily_features.csv
  - data/ndbc_historical_station_coverage.csv
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NDBC_STATIONS = ["46005", "46041", "46087", "46088", "46050"]
START_YEAR = 2003
END_YEAR = date.today().year

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def build_url(station: str, year: int) -> str:
    return f"https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"


def parse_ndbc_year_text(text: str, station: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    header = lines[0].split()
    # Skip units row.
    data_lines = lines[2:]
    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=r"\s+",
        names=header,
        on_bad_lines="skip",
    )

    required = ["#YY", "MM", "DD"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame()

    year = pd.to_numeric(df["#YY"], errors="coerce")
    month = pd.to_numeric(df["MM"], errors="coerce")
    day = pd.to_numeric(df["DD"], errors="coerce")
    df["date"] = pd.to_datetime({"year": year, "month": month, "day": day}, errors="coerce")
    df = df[df["date"].notna()].copy()
    if df.empty:
        return df

    for col in ["WSPD", "GST", "WVHT", "PRES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["station"] = station
    return df


def fetch_station_year(station: str, year: int) -> Tuple[pd.DataFrame, str]:
    url = build_url(station, year)
    try:
        r = requests.get(url, timeout=60, headers=HEADERS)
        if r.status_code != 200:
            return pd.DataFrame(), f"HTTP {r.status_code}"
        df = parse_ndbc_year_text(r.text, station)
        if df.empty:
            return pd.DataFrame(), "empty_or_unparseable"
        return df, "ok"
    except Exception as exc:
        return pd.DataFrame(), f"error:{exc}"


def main() -> None:
    print("\n=== Backfill Historical NDBC Features ===\n")
    yearly_frames: List[pd.DataFrame] = []
    coverage_rows: List[Dict[str, str]] = []

    for station in NDBC_STATIONS:
        print(f"[{station}]")
        ok_years = 0
        for year in range(START_YEAR, END_YEAR + 1):
            df, status = fetch_station_year(station, year)
            coverage_rows.append(
                {
                    "station": station,
                    "year": str(year),
                    "status": status,
                    "rows": str(len(df)),
                }
            )
            if not df.empty:
                yearly_frames.append(df)
                ok_years += 1
        print(f"  years_with_data: {ok_years}")

    if not yearly_frames:
        raise RuntimeError("No historical NDBC data downloaded.")

    full = pd.concat(yearly_frames, ignore_index=True)
    # Per-station daily summaries.
    per_station_daily = full.groupby(["date", "station"], as_index=False).agg(
        max_wspd=("WSPD", "max"),
        max_gst=("GST", "max"),
        max_wvht=("WVHT", "max"),
        min_pres=("PRES", "min"),
        mean_wspd=("WSPD", "mean"),
        mean_wvht=("WVHT", "mean"),
        mean_pres=("PRES", "mean"),
    )
    # Regional aggregate across stations for each day.
    regional = per_station_daily.groupby("date", as_index=False).agg(
        ndbc_max_wspd_mean=("max_wspd", "mean"),
        ndbc_max_gst_mean=("max_gst", "mean"),
        ndbc_max_wvht_mean=("max_wvht", "mean"),
        ndbc_min_pres_mean=("min_pres", "mean"),
        ndbc_mean_wspd_mean=("mean_wspd", "mean"),
        ndbc_mean_wvht_mean=("mean_wvht", "mean"),
        ndbc_mean_pres_mean=("mean_pres", "mean"),
        ndbc_station_count=("station", "nunique"),
    )
    regional = regional.sort_values("date").reset_index(drop=True)

    coverage = pd.DataFrame(coverage_rows)
    regional.to_csv(DATA_DIR / "ndbc_historical_daily_features.csv", index=False)
    coverage.to_csv(DATA_DIR / "ndbc_historical_station_coverage.csv", index=False)

    print(f"\nSaved data/ndbc_historical_daily_features.csv ({len(regional)} rows)")
    print(f"Saved data/ndbc_historical_station_coverage.csv ({len(coverage)} rows)")


if __name__ == "__main__":
    main()
