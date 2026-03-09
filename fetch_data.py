"""
fetch_data.py
Downloads teleconnection indices and Snoqualmie Pass snowpack data.

Teleconnections:
  - ENSO / ONI  (NOAA CPC)
  - PDO         (NOAA NCEI)
  - PNA         (NOAA CPC)
  - AO          (NOAA CPC)
  - NAO         (NOAA CPC)
  - MJO (RMM)   (Bureau of Meteorology, Australia)

Snowpack / Snowfall:
  - Snoqualmie Pass SNOTEL #908 (NRCS) — monthly snow depth, SWE, precipitation
"""

import os
import io
import requests
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch(url: str, label: str) -> str:
    """GET a URL and return the text body, or raise with a friendly message."""
    print(f"  Fetching {label} ...", end=" ", flush=True)
    r = requests.get(url, timeout=60, headers=HEADERS)
    r.raise_for_status()
    print("OK")
    return r.text


def save_csv(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(DATA_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  Saved -> {path}  ({len(df)} rows)")


# ── ENSO / ONI ────────────────────────────────────────────────────────────────
# Fixed-width table: SEAS  YR  TOTAL  ANOM
# Three-month running means of ERSSTv5 SST anomalies in the Nino 3.4 region.

def fetch_oni() -> pd.DataFrame:
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    raw = fetch(url, "ENSO / ONI")
    df = pd.read_csv(io.StringIO(raw), delim_whitespace=True)
    # Columns: SEAS  YR  TOTAL  ANOM
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"YR": "year", "SEAS": "season",
                             "TOTAL": "sst_total", "ANOM": "oni_anomaly"})
    return df


# ── RONI (Relative Oceanic Niño Index) ─────────────────────────────────────────
# CPC adopted RONI for ENSO monitoring; same season layout as ONI.
# https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt

SEAS_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def fetch_roni() -> pd.DataFrame:
    url = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
    raw = fetch(url, "RONI (Relative Oceanic Niño Index)")
    df = pd.read_csv(io.StringIO(raw), delim_whitespace=True)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"YR": "year", "SEAS": "season", "ANOM": "roni"})
    df["month"] = df["season"].map(SEAS_TO_MONTH)
    df = df.dropna(subset=["month"]).copy()
    df["month"] = df["month"].astype(int)
    df["roni"] = pd.to_numeric(df["roni"], errors="coerce")
    return df[["year", "month", "roni"]]


# ── PDO ───────────────────────────────────────────────────────────────────────
# NOAA PSL monthly PDO index.
# Format: first line = "  YYYY YYYY" (year range), then rows of year + 12 values.
# Source: https://psl.noaa.gov/data/correlation/pdo.data

def fetch_pdo() -> pd.DataFrame:
    url = "https://psl.noaa.gov/data/correlation/pdo.data"
    raw = fetch(url, "PDO")
    records = []
    for line in raw.splitlines():
        parts = line.split()
        # Need at least year + several monthly values (skip 2-token year-range header)
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        # Expect up to 12 monthly values after the year
        for m, v in enumerate(parts[1:13], start=1):
            try:
                fv = float(v)
                # PSL missing values: -9.99, -9.9, -99.9, -9999
                if abs(fv) >= 9.9:
                    fv = float("nan")
            except ValueError:
                fv = float("nan")
            records.append({"year": year, "month": m, "pdo": fv})
    return pd.DataFrame(records)


# ── PNA ───────────────────────────────────────────────────────────────────────
# NOAA CPC monthly PNA index — fixed-width wide table (years as rows, months as cols).

def fetch_pna() -> pd.DataFrame:
    url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
           "norm.pna.monthly.b5001.current.ascii.table")
    raw = fetch(url, "PNA")
    df = _parse_cpc_wide_table(raw, "pna")
    return df


# ── AO ────────────────────────────────────────────────────────────────────────
# NOAA CPC monthly AO index — space-delimited, year + 12 monthly values.

def fetch_ao() -> pd.DataFrame:
    url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
           "daily_ao_index/monthly.ao.index.b50.current.ascii")
    raw = fetch(url, "AO")
    df = _parse_cpc_monthly_ascii(raw, "ao")
    return df


# ── NAO ───────────────────────────────────────────────────────────────────────
# NOAA CPC monthly NAO index — same wide-table format as PNA.

def fetch_nao() -> pd.DataFrame:
    url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
           "norm.nao.monthly.b5001.current.ascii.table")
    raw = fetch(url, "NAO")
    df = _parse_cpc_wide_table(raw, "nao")
    return df


# ── MJO (RMM indices) ─────────────────────────────────────────────────────────
# Bureau of Meteorology real-time RMM1 / RMM2 / phase / amplitude — daily.
# Try HTTPS first; fall back to HTTP (BoM sometimes blocks one or the other).

def fetch_mjo() -> pd.DataFrame:
    for url in [
        "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
        "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
    ]:
        try:
            raw = fetch(url, "MJO (RMM)")
            break
        except Exception:
            pass
    else:
        raise RuntimeError("Could not fetch MJO data from BOM (both HTTP and HTTPS failed)")

    # Format: 2 header lines (start with spaces/text), then data rows:
    #   year  month  day  RMM1  RMM2  phase  amplitude  method_string
    # Values may use scientific notation (e.g. 9.38282013E-02).
    records = []
    for line in raw.splitlines():
        parts = line.split()
        # Need at least year month day rmm1 rmm2 phase amplitude (7 fields)
        if len(parts) < 7:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            rmm1  = float(parts[3])
            rmm2  = float(parts[4])
            phase = int(parts[5])
            amp   = float(parts[6])
        except ValueError:
            continue
        # Replace BOM missing values (1.E36 or 999)
        if abs(rmm1) > 900 or abs(rmm2) > 900 or abs(amp) > 900:
            rmm1 = rmm2 = amp = float("nan")
        records.append({
            "year": year, "month": month, "day": day,
            "rmm1": rmm1, "rmm2": rmm2, "phase": phase, "amplitude": amp,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df[["date", "year", "month", "day", "rmm1", "rmm2", "phase", "amplitude"]]
    return df


# ── Snoqualmie Pass SNOTEL #908 ───────────────────────────────────────────────
# NRCS SNOTEL Report Generator — monthly snow depth (in), SWE (in), precipitation (in).

def fetch_snoqualmie() -> pd.DataFrame:
    # Station 908, WA, SNTL — monthly period-of-record
    url = (
        "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
        "customMultiTimeSeriesGroupByStationReport/monthly/"
        "908:WA:SNTL%7Cid=%22%22%7Cname/"
        "POR_BEGIN,POR_END/"
        "SNWD::value,WTEQ::value,PREC::value"
    )
    raw = fetch(url, "Snoqualmie Pass SNOTEL #908")
    # Strip comment lines (start with '#')
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    # Rename columns to something friendly
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if "date" in cl:
            col_map[c] = "date"
        elif "snow depth" in cl or "snwd" in cl:
            col_map[c] = "snow_depth_in"
        elif "snow water" in cl or "wteq" in cl:
            col_map[c] = "swe_in"
        elif "precipitation" in cl or "prec" in cl:
            col_map[c] = "precip_in"
    df = df.rename(columns=col_map)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
    return df


# ── CPC table parsers ─────────────────────────────────────────────────────────

MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

def _parse_cpc_wide_table(raw: str, varname: str) -> pd.DataFrame:
    """
    Parse CPC-style wide ASCII table where each row is a year and columns
    are the 12 monthly values (plus optional annual column).
    """
    lines = [l for l in raw.splitlines() if l.strip()]
    records = []
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        year = int(parts[0])
        vals = parts[1:13]  # up to 12 monthly values
        for m, v in enumerate(vals, start=1):
            try:
                fv = float(v)
            except ValueError:
                fv = float("nan")
            records.append({"year": year, "month": m, varname: fv})
    return pd.DataFrame(records)


def _parse_cpc_monthly_ascii(raw: str, varname: str) -> pd.DataFrame:
    """
    Parse CPC monthly ASCII files that have the format:
        year  month  value
    or
        year  val1 val2 ... val12
    Handles both layouts.
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    records = []
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].replace("-","").isdigit():
            continue
        if len(parts) == 3:
            try:
                year, month, val = int(parts[0]), int(parts[1]), float(parts[2])
                records.append({"year": year, "month": month, varname: val})
            except ValueError:
                pass
        elif len(parts) >= 13:
            try:
                year = int(parts[0])
                for m, v in enumerate(parts[1:13], start=1):
                    records.append({"year": year, "month": m, varname: float(v)})
            except ValueError:
                pass
    return pd.DataFrame(records)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Teleconnection & Snowpack Data Fetcher ===\n")

    fetchers = [
        ("oni.csv",          fetch_oni,         "ENSO / ONI"),
        ("roni.csv",         fetch_roni,        "RONI (Relative Oceanic Niño Index)"),
        ("pdo.csv",          fetch_pdo,         "PDO"),
        ("pna.csv",          fetch_pna,         "PNA"),
        ("ao.csv",           fetch_ao,          "AO"),
        ("nao.csv",          fetch_nao,         "NAO"),
        ("mjo_rmm.csv",      fetch_mjo,         "MJO (RMM)"),
        ("snoqualmie_snotel.csv", fetch_snoqualmie, "Snoqualmie Pass SNOTEL"),
    ]

    results = {}
    errors = []

    for filename, func, label in fetchers:
        print(f"[{label}]")
        try:
            df = func()
            save_csv(df, filename)
            results[label] = df
            print()
        except Exception as e:
            print(f"  ERROR: {e}\n")
            errors.append((label, e))

    print("=" * 48)
    print("Summary")
    print("=" * 48)
    for label, df in results.items():
        print(f"  {label:<30} {len(df):>6} rows  ->  data/{label.lower().replace(' ','_').replace('/','_')}.csv")
    if errors:
        print("\nFailed:")
        for label, e in errors:
            print(f"  {label}: {e}")
    print()


if __name__ == "__main__":
    main()
