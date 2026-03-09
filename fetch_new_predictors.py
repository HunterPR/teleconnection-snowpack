"""
fetch_new_predictors.py
=======================
Downloads and processes new predictor data for the Snoqualmie snowpack model.

Teleconnection indices (long-record ML predictors):
  - EPO  (East Pacific Oscillation) -- direct NE Pacific ridge/trough measure
  - Nino4 anomaly                   -- central Pacific SST, CP vs EP ENSO flavour
  - AMO  (Atlantic Multidecadal Oscillation) -- ~60-80 yr cycle; 1856-present
  - Z500 NE Pacific blocking index  -- from NCEP/NCAR reanalysis (requires netCDF4)

Regional snowpack data (NRCS SNOTEL):
  - Stampede Pass   (#740:WA:SNTL, 3,850 ft)  — nearest SNOTEL to Snoqualmie Pass
  - Paradise        (#679:WA:SNTL, 5,440 ft)  — Mt Rainier, long record

Current conditions (display only — too short for ML training):
  - NWAC weather stations near Snoqualmie Pass (via api.nwac.us)
  - WSDOT Snoqualmie Pass road/weather conditions

Outputs (all in data/):
  epo.csv                  year, month, epo
  nino4_anom.csv           year, month, nino4_anom
  amo.csv                  year, month, amo
  snotel_stampede.csv      year, month, WTEQ_stampede, SNWD_stampede, ...
  snotel_paradise.csv      year, month, WTEQ_paradise, ...
  z500_nepac.csv           year, month, z500_nepac_anom   (if netCDF4 available)
  hyak_snowfall.csv        year, season_total (if hyak.net accessible)
  nwac_current.json        latest NWAC station readings (display only)
  wsdot_passes.json        latest WSDOT pass conditions (display only)
  data_manifest.json       source URLs, fetch timestamps, year ranges for each file
"""

import os
import sys
import time
import json
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from io import StringIO

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

MISSING = [-99.90, -99.99, -9999.0, -999.0]

# ── helpers ───────────────────────────────────────────────────────────────────

def mask_missing(series: pd.Series) -> pd.Series:
    for m in MISSING:
        series = series.where(~series.between(m - 0.1, m + 0.1), other=np.nan)
    return series


def parse_psl_wide(url: str, col_name: str) -> pd.DataFrame:
    """
    Parse PSL fixed-width year×12-month format into long (year, month, value).
    First line: start_year  end_year
    Subsequent lines: year  jan feb mar ... dec
    Missing flag: -99.90 or -99.99
    """
    print(f"  Fetching {url} ...")
    try:
        r = requests.get(url, timeout=30, verify=False)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    lines = r.text.strip().splitlines()
    records = []
    for line in lines[1:]:          # skip header row (start_yr  end_yr)
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            yr = int(float(parts[0]))
            vals = [float(v) for v in parts[1:13]]
        except ValueError:
            continue
        for mo, v in enumerate(vals, start=1):
            records.append({"year": yr, "month": mo, col_name: v})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df[col_name] = mask_missing(df[col_name])
    df = df.dropna(subset=[col_name])
    df[["year","month"]] = df[["year","month"]].astype(int)
    print(f"  Got {len(df)} rows ({df['year'].min()}-{df['year'].max()})")
    return df


# ── 1. EPO (East Pacific Oscillation) ────────────────────────────────────────

def fetch_epo() -> pd.DataFrame:
    """
    EPO: 500mb dipole over NE Pacific (55-65N,160W-125W minus 20-35N,160W-125W).
    Negative EPO = ridge over NE Pacific = bad for PNW snow.
    This is the most direct measure of the user's 'ridge in the wrong place' concern.
    """
    print("\n[EPO] Fetching East Pacific Oscillation ...")
    df = parse_psl_wide("https://psl.noaa.gov/data/correlation/epo.data", "epo")
    if not df.empty:
        out = os.path.join(DATA, "epo.csv")
        df.to_csv(out, index=False)
        print(f"  Saved {out}")
    return df


# ── 2. Nino 4 anomaly (Central Pacific SST) ───────────────────────────────────

def fetch_nino4() -> pd.DataFrame:
    """
    Nino 4 region (5S-5N, 160E-150W) absolute SST.
    We compute anomalies relative to 1991-2020 climatology.
    Distinguishes Central Pacific (CP / Modoki) La Nina from Eastern Pacific (EP) La Nina.
    CP La Nina tends to have weaker PNW response; EP La Nina is more favorable.
    """
    print("\n[Nino4] Fetching Nino 4 SST and computing anomaly ...")
    df = parse_psl_wide("https://psl.noaa.gov/data/correlation/nina4.data", "nino4_sst")
    if df.empty:
        return df

    # Compute 1991-2020 monthly climatology
    clim = (df[(df["year"] >= 1991) & (df["year"] <= 2020)]
            .groupby("month")["nino4_sst"].mean()
            .rename("clim"))
    df = df.merge(clim, on="month")
    df["nino4_anom"] = df["nino4_sst"] - df["clim"]
    df = df[["year","month","nino4_anom"]].copy()

    out = os.path.join(DATA, "nino4_anom.csv")
    df.to_csv(out, index=False)
    print(f"  Anomaly computed, saved {out}")
    return df


# ── 3. Additional SNOTEL stations ─────────────────────────────────────────────

SNOTEL_STATIONS = {
    # Stations used for current-conditions SNOTEL display (not circularity-removed from ML)
    # The ML model now uses Z500 instead of neighboring SNOTEL SWE to avoid circularity.
    # These are kept for the dashboard's "current conditions" display.
    "stampede":  {"triplet": "740:WA:SNTL",  "name": "Stampede Pass (nearest to Snoqualmie)", "elev_ft": 3850},
    "paradise":  {"triplet": "679:WA:SNTL",  "name": "Paradise (Mt Rainier, long record)",    "elev_ft": 5440},
}

SNOTEL_ELEMENTS = "WTEQ::value,SNWD::value,PREC::value,TAVG::value,TMAX::value,TMIN::value"
SNOTEL_START    = "1979-10-01"
SNOTEL_END      = "2025-09-30"


def fetch_snotel_station(key: str, info: dict) -> pd.DataFrame:
    """
    Download monthly SNOTEL data from NRCS report generator.
    Returns long-format DataFrame with year, month, and measurement columns.
    Uses verify=False because NRCS uses a cert chain the Windows CA store doesn't trust.
    """
    triplet = info["triplet"]
    name    = info["name"]
    print(f"\n[SNOTEL] Fetching {name} ({triplet}) ...")

    url = (
        f"https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
        f"customMultiTimeSeriesGroupByStationReport/monthly/start_of_period/"
        f"{triplet}/"
        f"{SNOTEL_START},{SNOTEL_END}/"
        f"{SNOTEL_ELEMENTS}"
    )

    try:
        r = requests.get(url, timeout=45, verify=False)
        r.raise_for_status()
    except Exception as e:
        print(f"  ERROR fetching {name}: {e}")
        return pd.DataFrame()

    # NRCS CSV has comment lines starting with '#'
    lines = [l for l in r.text.splitlines() if not l.startswith("#")]
    if len(lines) < 3:
        print(f"  No data returned for {name}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(StringIO("\n".join(lines)))
    except Exception as e:
        print(f"  Parse error for {name}: {e}")
        return pd.DataFrame()

    # The first column is typically a date like "2010-01-01"
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Rename measurement columns to standardized names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "snow water equivalent" in cl or "wteq" in cl:
            col_map[c] = f"WTEQ_{key}"
        elif "snow depth" in cl or "snwd" in cl:
            col_map[c] = f"SNWD_{key}"
        elif "precipitation accum" in cl or "prec" in cl:
            col_map[c] = f"PREC_{key}"
        elif "air temperature average" in cl or "tavg" in cl:
            col_map[c] = f"TAVG_{key}"
        elif "air temperature maximum" in cl or "tmax" in cl:
            col_map[c] = f"TMAX_{key}"
        elif "air temperature minimum" in cl or "tmin" in cl:
            col_map[c] = f"TMIN_{key}"
    df = df.rename(columns=col_map)

    keep = ["year","month"] + [v for v in col_map.values() if v in df.columns]
    df = df[keep].copy()

    # Replace NRCS sentinel values
    for c in df.columns:
        if c not in ["year","month"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = os.path.join(DATA, f"snotel_{key}.csv")
    df.to_csv(out, index=False)
    print(f"  {len(df)} rows | cols: {list(df.columns)} | saved {out}")
    return df


def fetch_all_snotel() -> dict:
    results = {}
    for key, info in SNOTEL_STATIONS.items():
        df = fetch_snotel_station(key, info)
        if not df.empty:
            results[key] = df
        time.sleep(1)   # be polite to NRCS servers
    return results


# ── 4. Z500 NE Pacific blocking index ─────────────────────────────────────────

def fetch_z500_nepac() -> pd.DataFrame:
    """
    Attempt to compute a NE Pacific 500mb geopotential height anomaly index
    from NCEP/NCAR reanalysis monthly means.

    Region: 45-65 N, 130-165 W  (Gulf of Alaska / NE Pacific ridge zone)
    A positive anomaly in this region = ridge = bad for PNW snow.

    Requires: netCDF4 or xarray + scipy
    Falls back gracefully if libraries are not installed.
    """
    print("\n[Z500] Attempting NE Pacific 500mb height index ...")

    try:
        import netCDF4 as nc
        USE_NETCDF4 = True
    except ImportError:
        USE_NETCDF4 = False

    try:
        import xarray as xr
        USE_XARRAY = True
    except ImportError:
        USE_XARRAY = False

    if not USE_NETCDF4 and not USE_XARRAY:
        print("  netCDF4 and xarray are both unavailable.")
        print("  To enable Z500: pip install netCDF4  OR  pip install xarray scipy")
        print("  Skipping Z500 index.")
        return pd.DataFrame()

    # Use OPeNDAP to stream only the NE Pacific subset (~10 KB vs ~200 MB full file)
    # THREDDS OPeNDAP endpoint for NCEP/NCAR reanalysis
    opendap_url = (
        "https://psl.noaa.gov/thredds/dodsC/Datasets/"
        "ncep.reanalysis.derived/pressure/hgt.mon.mean.nc"
    )
    nc_path = os.path.join(DATA, "hgt.mon.mean.nc")   # cached local copy if present

    try:
        if USE_XARRAY:
            # Try OPeNDAP first (streams only the requested subset — no local file needed)
            source = opendap_url
            cached = False
            if os.path.exists(nc_path):
                # Validate cached file before using it
                try:
                    xr.open_dataset(nc_path, engine="netcdf4").close()
                    source = nc_path
                    cached = True
                    print(f"  Using validated local cache: {nc_path}")
                except Exception:
                    print("  Cached file invalid — will stream via OPeNDAP")
                    os.remove(nc_path)

            if not cached:
                print(f"  Streaming NE Pacific subset via OPeNDAP (no full download needed)")
                print(f"  URL: {opendap_url}")

            engine = "netcdf4" if USE_NETCDF4 else "scipy"
            ds = xr.open_dataset(source, engine=engine)

            # Select 500mb level and NE Pacific region
            # Lats are typically descending (90->-90), lons 0->360
            hgt = ds["hgt"].sel(
                level=500,
                lat=slice(65, 45),    # descending lat slice: 65N to 45N
                lon=slice(195, 230),  # 165W-130W in 0-360 convention
            )
            weights  = np.cos(np.deg2rad(hgt.lat))
            hgt_mean = hgt.weighted(weights).mean(("lat", "lon"))
            df_z     = hgt_mean.to_dataframe(name="z500_nepac").reset_index()
            df_z["year"]  = pd.to_datetime(df_z["time"]).dt.year
            df_z["month"] = pd.to_datetime(df_z["time"]).dt.month
            df_z = df_z[["year", "month", "z500_nepac"]].copy()
            ds.close()

        else:  # netCDF4 direct (OPeNDAP via netCDF4 library)
            source = opendap_url if not os.path.exists(nc_path) else nc_path
            print(f"  Opening via netCDF4: {source}")
            ds = nc.Dataset(source)
            lat      = ds.variables["lat"][:]
            lon      = ds.variables["lon"][:]
            time_var = ds.variables["time"]
            times    = nc.num2date(time_var[:], time_var.units)
            levels   = ds.variables["level"][:]
            lvl_idx  = int(np.where(levels == 500)[0][0])
            lat_mask = (lat >= 45) & (lat <= 65)
            lon_mask = (lon >= 195) & (lon <= 230)
            lat_idx  = np.where(lat_mask)[0]
            lon_idx  = np.where(lon_mask)[0]
            # Stream only the needed spatial subset (OPeNDAP handles server-side slicing)
            hgt_all = ds.variables["hgt"][:, lvl_idx, lat_idx, :][:, :, lon_idx]
            records = []
            for t_idx, t in enumerate(times):
                region = hgt_all[t_idx]
                wts    = np.cos(np.deg2rad(lat[lat_idx]))[:, None] * np.ones_like(region)
                val    = float(np.average(region, weights=wts))
                records.append({"year": t.year, "month": t.month, "z500_nepac": val})
            df_z = pd.DataFrame(records)
            ds.close()

        # Compute anomaly relative to 1981-2010 climatology
        clim = (df_z[(df_z["year"] >= 1981) & (df_z["year"] <= 2010)]
                .groupby("month")["z500_nepac"].mean()
                .rename("z500_clim"))
        df_z = df_z.merge(clim, on="month")
        df_z["z500_nepac_anom"] = df_z["z500_nepac"] - df_z["z500_clim"]
        df_z = df_z[["year","month","z500_nepac_anom"]].copy()

        out = os.path.join(DATA, "z500_nepac.csv")
        df_z.to_csv(out, index=False)
        print(f"  Z500 NE Pacific index computed: {len(df_z)} rows | saved {out}")
        return df_z

    except Exception as e:
        print(f"  Z500 fetch failed: {e}")
        print("  Skipping Z500 index (install netCDF4 or xarray to enable).")
        return pd.DataFrame()


# ── 5. Hyak.net historical snowfall scrape ────────────────────────────────────

def fetch_hyak() -> pd.DataFrame:
    """
    Scrape http://hyak.net/snowfallhist.html for historical seasonal snowfall.
    Data goes back to ~1930, giving pre-SNOTEL snowfall context.
    The site occasionally refuses connections — fails gracefully.
    """
    print("\n[Hyak] Attempting hyak.net historical snowfall scrape ...")
    url = "http://hyak.net/snowfallhist.html"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        print(f"  hyak.net unavailable: {e}")
        print("  Skipping — try again later.")
        return pd.DataFrame()

    try:
        import re
        # HTML structure:
        #   <center>1930-31</center></td>
        #   <td bgcolor="#..."><center>361</center></td>
        # Capture season string and first numeric value (seasonal total)
        rows = re.findall(
            r'<center>(\d{4}-\d{2})</center>.*?'
            r'<td[^>]*>\s*<center>([\d.]+)</center>',
            r.text, re.DOTALL
        )

        if not rows:
            print("  Could not parse hyak.net table structure.")
            print("  Page returned but format unexpected.")
            return pd.DataFrame()

        records = []
        for season, total in rows:
            yr_start = int(season[:4])
            records.append({
                "season_start": yr_start,
                "season":       season,
                "hyak_snow_total_in": float(total),
            })
        df = pd.DataFrame(records).sort_values("season_start")
        out = os.path.join(DATA, "hyak_snowfall.csv")
        df.to_csv(out, index=False)
        print(f"  Scraped {len(df)} seasons ({df['season_start'].min()}-{df['season_start'].max()})")
        print(f"  Saved {out}")
        return df

    except Exception as e:
        print(f"  Parse error: {e}")
        return pd.DataFrame()


# ── 6. AMO (Atlantic Multidecadal Oscillation) ────────────────────────────────

def fetch_amo() -> pd.DataFrame:
    """
    Atlantic Multidecadal Oscillation (AMO) unsmoothed monthly index.
    Source: NOAA PSL Climate Indices — data back to 1856.

    Physical link to PNW snow: AMO modulates the Pacific-North American pattern
    through Atlantic SST influence on the hemispheric Rossby wave pattern.
    Warm AMO phase (positive) tends to weakly suppress PNW precipitation on
    decadal timescales; cold AMO (negative) is mildly favorable.
    """
    print("\n[AMO] Fetching Atlantic Multidecadal Oscillation ...")
    df = parse_psl_wide("https://psl.noaa.gov/data/correlation/amon.us.long.data", "amo")
    if not df.empty:
        out = os.path.join(DATA, "amo.csv")
        df.to_csv(out, index=False)
        print(f"  Saved {out}")
    return df


# ── 7. NWAC current weather stations ──────────────────────────────────────────

def fetch_nwac_current() -> dict:
    """
    Fetch current weather conditions from NWAC (Northwest Avalanche Center)
    stations near Snoqualmie Pass.

    Uses the public NWAC data portal API.
    Returns a dict of station readings (for display only — too short for ML).
    Saves to data/nwac_current.json.
    """
    print("\n[NWAC] Fetching current station conditions ...")

    # Discover stations from NWAC API (may be restricted; fall back to known IDs)
    stations_url = "https://api.nwac.us/api/v1/station/"
    FALLBACK_STATIONS = [
        # Known NWAC stations near Snoqualmie Pass / Central Cascades
        {"id": 1, "stid": "snq_pass", "name": "Snoqualmie Pass",      "elevation": 3000},
        {"id": 2, "stid": "alpental", "name": "Alpental (Snoqualmie)", "elevation": 2900},
        {"id": 3, "stid": "crystal",  "name": "Crystal Mountain",      "elevation": 4400},
        {"id": 4, "stid": "stevens",  "name": "Stevens Pass",          "elevation": 4061},
    ]
    try:
        r = requests.get(stations_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        all_stations = r.json().get("results", r.json() if isinstance(r.json(), list) else [])
    except Exception as e:
        print(f"  NWAC API not accessible ({e}); using fallback station list")
        all_stations = FALLBACK_STATIONS

    # Filter to stations near Snoqualmie Pass / Central Cascades WA
    TARGET_NAMES = [
        "snoqualmie", "alpental", "commonwealth", "hyak",
        "stampede", "crystal", "chinook", "cayuse",
    ]
    nearby = []
    for s in all_stations:
        name_lower = str(s.get("name", "")).lower()
        if any(t in name_lower for t in TARGET_NAMES):
            nearby.append(s)

    if not nearby:
        # Fall back: show all WA stations
        nearby = [s for s in all_stations if "WA" in str(s.get("stid", "")) or
                  str(s.get("state", "")).upper() == "WA"][:20]

    print(f"  Found {len(all_stations)} total stations, {len(nearby)} near Snoqualmie/Central Cascades")

    # Fetch latest observation for each nearby station
    results = []
    for s in nearby[:15]:   # cap at 15 to avoid rate-limiting
        sid = s.get("id") or s.get("stid")
        if not sid:
            continue
        try:
            obs_url = f"https://api.nwac.us/api/v1/observation/?station_id={sid}&limit=1&ordering=-observation_date"
            ro = requests.get(obs_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            ro.raise_for_status()
            obs_data = ro.json()
            obs_list = obs_data.get("results", obs_data if isinstance(obs_data, list) else [])
            latest = obs_list[0] if obs_list else {}
            results.append({
                "station_id":   sid,
                "station_name": s.get("name", "Unknown"),
                "elevation_ft": s.get("elevation", s.get("elev")),
                "lat":          s.get("latitude", s.get("lat")),
                "lon":          s.get("longitude", s.get("lon")),
                "obs_time":     latest.get("observation_date") or latest.get("datetime"),
                "snow_depth_in": latest.get("snow_depth_in") or latest.get("snow_depth"),
                "air_temp_f":   latest.get("air_temperature") or latest.get("temp"),
                "wind_speed_mph": latest.get("wind_speed") or latest.get("wspd"),
                "precip_24h_in": latest.get("precipitation_24_hour") or latest.get("precip_24hr"),
            })
            time.sleep(0.3)
        except Exception as e:
            print(f"  Could not fetch obs for station {sid}: {e}")

    out = os.path.join(DATA, "nwac_current.json")
    payload = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "stations": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved {len(results)} station readings -> {out}")
    return payload


# ── 8. WSDOT mountain pass conditions ─────────────────────────────────────────

def fetch_wsdot_passes() -> dict:
    """
    Fetch current mountain pass conditions from WSDOT.
    Attempts the public JSON API first; falls back to a lightweight HTML scrape.
    Focuses on I-90 Snoqualmie Pass and nearby passes.
    Saves to data/wsdot_passes.json.
    """
    print("\n[WSDOT] Fetching mountain pass conditions ...")

    PASS_NAMES = ["snoqualmie", "stevens", "white pass", "chinook", "blewett",
                  "north cascades", "north cascade", "crystal", "tiger", "cayuse"]
    results = []

    # WSDOT public ArcGIS FeatureServer — no key required
    arcgis_url = (
        "https://data.wsdot.wa.gov/arcgis/rest/services/TravelInformation/"
        "TravelInfoMtPassReports/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&f=json"
    )
    try:
        r = requests.get(arcgis_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
        features = data.get("features", [])
        for feat in features:
            attrs = feat.get("attributes", {})
            name = str(attrs.get("PassName", attrs.get("MountainPassName", ""))).lower()
            if not name or not any(t in name for t in PASS_NAMES):
                continue
            results.append({
                "name":           attrs.get("PassName") or attrs.get("MountainPassName"),
                "elevation_ft":   attrs.get("Elevation") or attrs.get("ElevationInFeet"),
                "weather_desc":   attrs.get("Weather") or attrs.get("WeatherCondition"),
                "road_condition": attrs.get("RoadCondition"),
                "northbound":     attrs.get("PublicMessage1"),
                "southbound":     attrs.get("PublicMessage2"),
                "report_time":    attrs.get("DisplayDate") or attrs.get("DateUpdated"),
            })
        print(f"  WSDOT ArcGIS: {len(features)} passes total, {len(results)} matching")
    except Exception as e:
        print(f"  WSDOT ArcGIS failed ({e}), trying KML fallback ...")
        # KML endpoint — also public, no key
        try:
            kml_url = "https://wsdot.wa.gov/traffic/api/mountainpassconditions/kml.aspx"
            rk = requests.get(kml_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            rk.raise_for_status()
            import re
            # Very light KML parse: extract name + description blocks
            names    = re.findall(r'<name>(.*?)</name>', rk.text, re.DOTALL)
            descs    = re.findall(r'<description>(.*?)</description>', rk.text, re.DOTALL)
            for nm, desc in zip(names, descs):
                if any(t in nm.lower() for t in PASS_NAMES):
                    temp_m = re.search(r'(\d{1,3})\s*[°˚]?\s*F', desc)
                    results.append({
                        "name":     nm.strip(),
                        "source":   "WSDOT KML",
                        "temp_f":   int(temp_m.group(1)) if temp_m else None,
                        "raw_desc": re.sub(r'<[^>]+>', ' ', desc).strip()[:300],
                    })
            print(f"  WSDOT KML: found {len(results)} matching passes")
        except Exception as e2:
            print(f"  WSDOT KML also failed: {e2}")

    out = os.path.join(DATA, "wsdot_passes.json")
    payload = {
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "source_url":  arcgis_url,
        "passes": results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  Saved {len(results)} pass records -> {out}")
    return payload


# ── 9. Summit at Snoqualmie ski area snow report ───────────────────────────────

def fetch_summit_snow_report() -> dict:
    """
    Scrape current snow conditions from Summit at Snoqualmie ski area.
    Returns current base depth and 24h/72h snowfall.
    Saves to data/summit_snow_report.json.
    """
    print("\n[Summit] Fetching Summit at Snoqualmie snow report ...")
    url = "https://www.summitatsnoqualmie.com/mountain-report"
    result = {"fetched_utc": datetime.now(timezone.utc).isoformat(), "source_url": url}
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        import re
        text = r.text

        # Base depth
        base_m = re.search(r'base[^<]{0,40}?(\d{1,4})["\s]*(?:inch|in|")', text, re.IGNORECASE)
        if base_m:
            result["base_depth_in"] = int(base_m.group(1))

        # 24h snowfall
        h24_m = re.search(r'24[- ]?hour[^<]{0,30}?(\d{1,3})["\s]*(?:inch|in|")', text, re.IGNORECASE)
        if h24_m:
            result["snowfall_24h_in"] = int(h24_m.group(1))

        # 72h snowfall
        h72_m = re.search(r'72[- ]?hour[^<]{0,30}?(\d{1,3})["\s]*(?:inch|in|")', text, re.IGNORECASE)
        if h72_m:
            result["snowfall_72h_in"] = int(h72_m.group(1))

        # Try structured JSON-LD if present
        jld = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', text, re.DOTALL)
        for j in jld:
            try:
                obj = json.loads(j)
                if "snowConditions" in str(obj) or "snowfall" in str(obj).lower():
                    result["json_ld"] = obj
            except Exception:
                pass

        print(f"  Summit snow report: {result}")
    except Exception as e:
        print(f"  Could not fetch Summit snow report: {e}")
        result["error"] = str(e)

    out = os.path.join(DATA, "summit_snow_report.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return result


# ── 11. NDBC buoy monthly aggregates ───────────────────────────────────────────

def fetch_ndbc_monthly() -> pd.DataFrame:
    """
    Aggregate already-downloaded NDBC daily features to monthly means.
    Reads data/ndbc_historical_daily_features.csv (produced by fetch_ndbc_historical.py).
    Outputs data/ndbc_monthly.csv with columns: year, month, buoy_wvht, buoy_pres, buoy_wspd.

    Physical meaning:
      buoy_wvht  - NE Pacific significant wave height (storm track intensity proxy)
      buoy_pres  - NE Pacific sea-level pressure (Aleutian Low strength proxy)
      buoy_wspd  - NE Pacific surface wind speed

    These are valid upstream predictors: large-scale ocean/atmosphere state,
    not local Cascade weather.
    """
    print("\n[NDBC Monthly] Aggregating buoy daily -> monthly ...")
    daily_path = os.path.join(DATA, "ndbc_historical_daily_features.csv")
    if not os.path.exists(daily_path):
        print("   ndbc_historical_daily_features.csv not found.")
        print("   Run fetch_ndbc_historical.py first.")
        return pd.DataFrame()

    df = pd.read_csv(daily_path, parse_dates=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Replace NDBC missing sentinel (9999.0 fill value) with NaN
    for col in ["ndbc_mean_wvht_mean", "ndbc_mean_pres_mean", "ndbc_mean_wspd_mean",
                "ndbc_min_pres_mean", "ndbc_max_wvht_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Pressure: valid 870-1084 hPa. Wave height: valid 0-30 m. Wind: 0-90 m/s.
            if "pres" in col:
                df.loc[df[col] > 1100, col] = float("nan")
            elif "wvht" in col:
                df.loc[df[col] > 50, col] = float("nan")
            elif "wspd" in col:
                df.loc[df[col] > 100, col] = float("nan")

    monthly = df.groupby(["year", "month"], as_index=False).agg(
        buoy_wvht=("ndbc_mean_wvht_mean",  "mean"),  # mean sig wave height (m)
        buoy_pres=("ndbc_mean_pres_mean",  "mean"),  # mean SLP at buoys (hPa)
        buoy_wspd=("ndbc_mean_wspd_mean",  "mean"),  # mean wind speed (m/s)
        buoy_wvht_max=("ndbc_max_wvht_mean", "mean"),  # mean of daily max wave height
        buoy_pres_min=("ndbc_min_pres_mean", "mean"),  # mean of daily min pres (storm depth)
        buoy_storm_days=("ndbc_min_pres_mean",
                         lambda x: (pd.to_numeric(x, errors="coerce") < 1000).sum()),
    )

    out = os.path.join(DATA, "ndbc_monthly.csv")
    monthly.to_csv(out, index=False)
    yr_min, yr_max = int(monthly["year"].min()), int(monthly["year"].max())
    print(f"   Saved ndbc_monthly.csv: {len(monthly)} rows ({yr_min}-{yr_max})")
    return monthly


# ── 12. Synoptic gradient monthly aggregates ────────────────────────────────────

def fetch_synoptic_monthly() -> pd.DataFrame:
    """
    Aggregate synoptic_daily_features.csv (Open-Meteo reanalysis) to monthly means.
    Reads data/synoptic_daily_features.csv (produced by fetch_synoptic_features.py).
    Outputs data/synoptic_monthly.csv.

    Key derived features:
      syn_hgt500_gradient - 500mb height: offshore minus cascade (ridge/trough indicator)
      syn_slp_gradient    - SLP: offshore minus cascade (storm track strength)

    These are reanalysis-based circulation proxies, NOT local weather observations.
    Valid upstream predictors for Cascade snowpack.
    """
    print("\n[Synoptic Monthly] Aggregating synoptic daily -> monthly ...")
    daily_path = os.path.join(DATA, "synoptic_daily_features.csv")
    if not os.path.exists(daily_path):
        print("   synoptic_daily_features.csv not found.")
        print("   Run fetch_synoptic_features.py first.")
        return pd.DataFrame()

    df = pd.read_csv(daily_path)
    # Handle timezone-aware date strings from Open-Meteo (strip tz, keep wall-clock time)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)
    df = df[df["date"].notna()].copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    agg_dict = {}
    col_map = {
        "syn_hgt500_gradient": "syn_hgt500_gradient_offshore_minus_cascade",
        "syn_slp_gradient":    "syn_slp_gradient_offshore_minus_cascade",
        "syn_thickness":       "syn_thickness_proxy_500_850",
    }
    for out_col, src_col in col_map.items():
        if src_col in df.columns:
            df[src_col] = pd.to_numeric(df[src_col], errors="coerce")
            agg_dict[out_col] = (src_col, "mean")

    if not agg_dict:
        print("   No matching gradient columns found in synoptic_daily_features.csv")
        return pd.DataFrame()

    monthly = df.groupby(["year", "month"], as_index=False).agg(**agg_dict)

    out = os.path.join(DATA, "synoptic_monthly.csv")
    monthly.to_csv(out, index=False)
    yr_min, yr_max = int(monthly["year"].min()), int(monthly["year"].max())
    print(f"   Saved synoptic_monthly.csv: {len(monthly)} rows ({yr_min}-{yr_max})")
    return monthly


# ── 13. NE Pacific SLP monthly anomaly (Aleutian Low index) ────────────────────

def fetch_slp_nepac() -> pd.DataFrame:
    """
    Fetch NCEP/NCAR reanalysis monthly mean SLP for the NE Pacific (Aleutian Low region)
    via OPeNDAP. Compute area-weighted mean and anomaly (1981-2010 climatology).

    Region: 45-65N, 160-210E (160W-150W in 0-360 convention) — Aleutian Low core.
    Positive anomaly = weaker Aleutian Low = ridge = less Pacific moisture -> bad for PNW snow.
    Negative anomaly = stronger Aleutian Low = deeper trough = storm track active -> good for snow.

    Data covers 1948-present, fully overlapping our 1950-2024 training record.
    """
    print("\n[SLP NePac] Fetching monthly NE Pacific SLP from NCEP/NCAR OPeNDAP ...")

    try:
        import xarray as xr
        USE_XARRAY = True
    except ImportError:
        USE_XARRAY = False

    try:
        import netCDF4 as nc
        USE_NETCDF4 = not USE_XARRAY
    except ImportError:
        USE_NETCDF4 = False

    if not USE_XARRAY and not USE_NETCDF4:
        print("   xarray/netCDF4 unavailable — skipping SLP NePac fetch.")
        return pd.DataFrame()

    opendap_url = (
        "https://psl.noaa.gov/thredds/dodsC/Datasets/"
        "ncep.reanalysis.derived/surface/slp.mon.mean.nc"
    )

    try:
        if USE_XARRAY:
            engine = "netcdf4"
            ds = xr.open_dataset(opendap_url, engine=engine)
            # SLP variable is 'slp', units are Pascals in NCEP/NCAR
            slp = ds["slp"].sel(
                lat=slice(65, 45),    # descending lat: 65N to 45N
                lon=slice(160, 210),  # 160E-210E (= 160W-150W) for Aleutian Low
            )
            weights  = np.cos(np.deg2rad(slp.lat))
            slp_mean = slp.weighted(weights).mean(("lat", "lon"))
            df_slp   = slp_mean.to_dataframe(name="slp_nepac").reset_index()
            df_slp["year"]  = pd.to_datetime(df_slp["time"]).dt.year
            df_slp["month"] = pd.to_datetime(df_slp["time"]).dt.month
            df_slp["slp_nepac"] = df_slp["slp_nepac"] / 100.0  # Pa -> hPa
            ds.close()
        else:
            ds = nc.Dataset(opendap_url)
            lat      = ds.variables["lat"][:]
            lon      = ds.variables["lon"][:]
            time_var = ds.variables["time"]
            times    = nc.num2date(time_var[:], time_var.units)
            lat_mask = (lat >= 45) & (lat <= 65)
            lon_mask = (lon >= 160) & (lon <= 210)
            lat_idx  = np.where(lat_mask)[0]
            lon_idx  = np.where(lon_mask)[0]
            slp_all  = ds.variables["slp"][:, lat_idx, :][:, :, lon_idx]
            weights  = np.cos(np.deg2rad(lat[lat_idx]))
            slp_wt   = np.nansum(slp_all * weights[None, :, None], axis=(1, 2)) / np.nansum(weights)
            df_slp   = pd.DataFrame({
                "year":  [t.year  for t in times],
                "month": [t.month for t in times],
                "slp_nepac": slp_wt / 100.0,  # Pa -> hPa
            })
            ds.close()

        # Compute 1981-2010 climatology anomaly
        clim = df_slp[(df_slp["year"] >= 1981) & (df_slp["year"] <= 2010)].groupby("month")["slp_nepac"].mean()
        df_slp["slp_nepac_anom"] = df_slp["slp_nepac"] - df_slp["month"].map(clim)
        df_slp = df_slp[["year", "month", "slp_nepac", "slp_nepac_anom"]].copy()

        out = os.path.join(DATA, "slp_nepac.csv")
        df_slp.to_csv(out, index=False)
        yr_min, yr_max = int(df_slp["year"].min()), int(df_slp["year"].max())
        print(f"   Saved slp_nepac.csv: {len(df_slp)} rows ({yr_min}-{yr_max})")
        return df_slp

    except Exception as e:
        print(f"   SLP fetch failed: {e}")
        return pd.DataFrame()


# ── 14. 500mb height gradient: NE Pacific offshore vs Cascade crest ───────────

def fetch_hgt500_gradient() -> pd.DataFrame:
    """
    Compute monthly 500mb geopotential height gradient:
        offshore_mean(50-58N, 220-235E) minus cascade_mean(45-50N, 235-243E)

    Positive gradient = higher heights upstream = onshore flow = Pacific storms
    Negative gradient = ridge building over Cascades = blocking = bad for PNW snow

    Distinct from z500_nepac_anom (broad NE Pacific mean): captures whether the
    ridge/trough tilts toward the Cascades vs is centred offshore.

    Source: NCEP/NCAR Reanalysis monthly `hgt.mon.mean.nc` via OPeNDAP.
    Coverage: 1948-present, fully overlapping training record.
    """
    print("\n[HGT500 Gradient] Fetching 500mb height gradient from NCEP/NCAR ...")

    try:
        import xarray as xr
        USE_XARRAY = True
    except ImportError:
        USE_XARRAY = False

    if not USE_XARRAY:
        try:
            import netCDF4 as nc
        except ImportError:
            print("   xarray/netCDF4 unavailable — skipping hgt500 gradient fetch.")
            return pd.DataFrame()

    url = (
        "https://psl.noaa.gov/thredds/dodsC/Datasets/"
        "ncep.reanalysis.derived/pressure/hgt.mon.mean.nc"
    )

    for attempt in range(3):
        try:
            if attempt > 0:
                print(f"   Retry {attempt}/2 after delay ...")
                time.sleep(4)
            if USE_XARRAY:
                ds = xr.open_dataset(url, engine="netcdf4")
                h = ds["hgt"].sel(level=500.0)
                # Offshore box: Gulf of Alaska region (upstream)
                h_off  = h.sel(lat=slice(58, 50), lon=slice(220, 235))
                w_off  = np.cos(np.deg2rad(h_off.lat))
                off_mean = h_off.weighted(w_off).mean(("lat", "lon"))
                # Cascade box: Washington State / Cascades (target area)
                h_cas  = h.sel(lat=slice(50, 45), lon=slice(235, 243))
                w_cas  = np.cos(np.deg2rad(h_cas.lat))
                cas_mean = h_cas.weighted(w_cas).mean(("lat", "lon"))
                gradient = off_mean - cas_mean
                df_g = gradient.to_dataframe(name="hgt500_gradient").reset_index()
                df_g["year"]  = pd.to_datetime(df_g["time"]).dt.year
                df_g["month"] = pd.to_datetime(df_g["time"]).dt.month
                df_g = df_g[["year", "month", "hgt500_gradient"]].copy()
                ds.close()
            else:
                import netCDF4 as nc
                ds = nc.Dataset(url)
                lat      = ds.variables["lat"][:]
                lon      = ds.variables["lon"][:]
                lev      = ds.variables["level"][:]
                time_var = ds.variables["time"]
                times    = nc.num2date(time_var[:], time_var.units)
                lev_idx  = np.where(np.abs(lev - 500) < 1)[0][0]
                off_lat = (lat >= 50) & (lat <= 58)
                off_lon = (lon >= 220) & (lon <= 235)
                cas_lat = (lat >= 45) & (lat <= 50)
                cas_lon = (lon >= 235) & (lon <= 243)
                hgt_all = ds.variables["hgt"][:, lev_idx, :, :]
                w_off = np.cos(np.deg2rad(lat[off_lat]))
                w_cas = np.cos(np.deg2rad(lat[cas_lat]))
                off_vals = np.nansum(
                    hgt_all[:, off_lat, :][:, :, off_lon] * w_off[None, :, None],
                    axis=(1, 2)) / np.nansum(w_off)
                cas_vals = np.nansum(
                    hgt_all[:, cas_lat, :][:, :, cas_lon] * w_cas[None, :, None],
                    axis=(1, 2)) / np.nansum(w_cas)
                df_g = pd.DataFrame({
                    "year":  [t.year  for t in times],
                    "month": [t.month for t in times],
                    "hgt500_gradient": off_vals - cas_vals,
                })
                ds.close()

            out = os.path.join(DATA, "hgt500_gradient.csv")
            df_g.to_csv(out, index=False)
            yr_min, yr_max = int(df_g["year"].min()), int(df_g["year"].max())
            print(f"   Saved hgt500_gradient.csv: {len(df_g)} rows ({yr_min}-{yr_max})")
            return df_g

        except Exception as e:
            print(f"   hgt500 gradient attempt {attempt+1} failed: {e}")
            if attempt == 2:
                print("   Giving up after 3 attempts.")
                return pd.DataFrame()


# ── 15. Nino1+2 and TNI (Trans-Nino Index) ────────────────────────────────────

def fetch_nino12_tni() -> pd.DataFrame:
    """
    Fetch Nino1+2 SST anomaly (0-10S, 90W-80W) from NOAA PSL.
    Compute TNI = nino12_anom - nino4_anom (Trans-Nino Index).

    Physical link:
      Nino1+2: eastern Pacific SST — traditional EP El Nino signal
      Nino4:   central Pacific SST — CP (Modoki) El Nino signal
      TNI > 0: EP El Nino pattern -> stronger subtropical ridge -> LESS PNW snow
      TNI < 0: CP-dominated La Nina/Modoki -> favorable for PNW ridging breakdown

    EP El Nino (high TNI) has much stronger suppressive effect on Cascade snow
    than CP La Nina, so this index differentiates ENSO flavour for PNW forecasting.

    Source: NOAA PSL Climate Indices.
    Coverage: 1950-present.
    """
    print("\n[Nino1+2 / TNI] Fetching Nino1+2 anomaly from NOAA PSL ...")

    # PSL filename for the Nino1+2 region is nina1.data (covers 0-10S, 90W-80W)
    # The file contains absolute SST; compute anomaly using 1991-2020 climatology.
    df_raw = parse_psl_wide(
        "https://psl.noaa.gov/data/correlation/nina1.data", "nino12_sst"
    )
    if df_raw.empty:
        print("   Nino1+2 fetch failed.")
        return pd.DataFrame()

    # Compute 1991-2020 monthly climatology anomaly
    clim = (df_raw[(df_raw["year"] >= 1991) & (df_raw["year"] <= 2020)]
            .groupby("month")["nino12_sst"].mean()
            .rename("clim"))
    df_n12 = df_raw.merge(clim, on="month")
    df_n12["nino12_anom"] = df_n12["nino12_sst"] - df_n12["clim"]
    df_n12 = df_n12[["year", "month", "nino12_anom"]].copy()

    out_n12 = os.path.join(DATA, "nino12_anom.csv")
    df_n12.to_csv(out_n12, index=False)
    print(f"   Saved nino12_anom.csv: {len(df_n12)} rows "
          f"({int(df_n12['year'].min())}-{int(df_n12['year'].max())})")

    # Compute TNI = nino12_anom - nino4_anom
    nino4_path = os.path.join(DATA, "nino4_anom.csv")
    if os.path.exists(nino4_path):
        df_n4 = pd.read_csv(nino4_path)
        df_n4[["year", "month"]] = df_n4[["year", "month"]].astype(int)
        df_tni = df_n12.merge(df_n4[["year", "month", "nino4_anom"]],
                              on=["year", "month"], how="inner")
        df_tni["tni"] = df_tni["nino12_anom"] - df_tni["nino4_anom"]
        out_tni = os.path.join(DATA, "tni.csv")
        df_tni[["year", "month", "tni"]].to_csv(out_tni, index=False)
        print(f"   Saved tni.csv: {len(df_tni)} rows "
              f"({int(df_tni['year'].min())}-{int(df_tni['year'].max())})")
    else:
        print("   nino4_anom.csv not found — TNI not computed (run fetch_nino4 first)")

    return df_n12


# ── 10. Data manifest (provenance tracking) ────────────────────────────────────

DATA_SOURCES = {
    "ao.csv":           {"name": "AO (Arctic Oscillation)",      "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table",  "update": "~monthly"},
    "nao.csv":          {"name": "NAO (N. Atlantic Oscillation)", "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table", "update": "~monthly"},
    "pna.csv":          {"name": "PNA (Pacific/N. America)",      "url": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table", "update": "~monthly"},
    "pdo.csv":          {"name": "PDO (Pacific Decadal Osc.)",    "url": "https://psl.noaa.gov/pdo/",   "update": "~monthly"},
    "oni.csv":          {"name": "ONI / ENSO34",                  "url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", "update": "~weekly"},
    "epo.csv":          {"name": "EPO (E. Pacific Osc.)",         "url": "https://psl.noaa.gov/data/correlation/epo.data",          "update": "~monthly"},
    "nino4_anom.csv":   {"name": "Nino4 SST Anomaly",             "url": "https://psl.noaa.gov/data/correlation/nina4.data",        "update": "~monthly"},
    "amo.csv":          {"name": "AMO (Atlantic Multidecadal)",   "url": "https://psl.noaa.gov/data/correlation/amon.us.long.data", "update": "~monthly"},
    "z500_nepac.csv":   {"name": "Z500 NE Pacific Anomaly",       "url": "NCEP/NCAR Reanalysis OPeNDAP (psl.noaa.gov)",            "update": "~1 month lag"},
    "snoqualmie_snotel.csv": {"name": "Snoqualmie Pass SNOTEL #908", "url": "https://wcc.sc.egov.usda.gov/reportGenerator/", "update": "daily"},
    "ndbc_monthly.csv":    {"name": "NDBC Buoy Monthly (NE Pacific)",  "url": "NDBC historical stdmet (buoys 46005/46041/46087/46088/46050)", "update": "monthly"},
    "synoptic_monthly.csv": {"name": "Synoptic Gradients Monthly",     "url": "Open-Meteo Archive API (offshore/cascade gradient)", "update": "monthly"},
    "slp_nepac.csv":        {"name": "NE Pacific SLP Anomaly",         "url": "NCEP/NCAR Reanalysis OPeNDAP (psl.noaa.gov) — Aleutian Low", "update": "~1 month lag"},
    "hgt500_gradient.csv":  {"name": "500mb Height Gradient (NE Pac -> Cascade)", "url": "NCEP/NCAR Reanalysis OPeNDAP (psl.noaa.gov) hgt.mon.mean.nc", "update": "~1 month lag"},
    "nino12_anom.csv":      {"name": "Nino1+2 SST Anomaly (Eastern Pacific)",     "url": "https://psl.noaa.gov/data/correlation/nina1+2.data",    "update": "~monthly"},
    "tni.csv":              {"name": "TNI (Trans-Nino Index = Nino1+2 - Nino4)",  "url": "Derived: nino12_anom - nino4_anom",                      "update": "~monthly"},
}


def write_data_manifest(results: dict):
    """
    Write a JSON manifest recording each data file's source, fetch timestamp,
    and year range. Used by the dashboard's Data Provenance panel.
    """
    manifest = {}
    for fname, meta in DATA_SOURCES.items():
        path = os.path.join(DATA, fname)
        entry = {
            "name":       meta["name"],
            "source_url": meta["url"],
            "update_freq": meta["update"],
            "file":       fname,
            "exists":     os.path.exists(path),
            "last_modified": None,
            "year_range": None,
            "n_rows": None,
            "fetch_session": datetime.now(timezone.utc).isoformat(),
        }
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            entry["last_modified"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            try:
                df = pd.read_csv(path)
                if "year" in df.columns:
                    entry["year_range"] = f"{int(df['year'].min())}–{int(df['year'].max())}"
                    entry["n_rows"] = len(df)
                    # Flag staleness: data should have entries within last 3 months
                    if "month" in df.columns:
                        latest_ym = int(df["year"].max()) * 12 + int(df[df["year"]==df["year"].max()]["month"].max())
                        now_ym    = datetime.now().year * 12 + datetime.now().month
                        months_old = now_ym - latest_ym
                        entry["months_since_latest_data"] = months_old
                        entry["stale"] = months_old > 4
            except Exception:
                pass
        # Merge in any custom fetch results (e.g., from this session's fetch)
        if fname in results:
            entry["fetch_ok"] = not (isinstance(results[fname], pd.DataFrame) and results[fname].empty)
        manifest[fname] = entry

    out = os.path.join(DATA, "data_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Data manifest written -> {out}")
    return manifest


# ── 6. Summary merge helper ────────────────────────────────────────────────────

def build_summary(results: dict):
    """Print a quick summary of what was fetched and write data manifest."""
    print("\n" + "="*60)
    print("  FETCH SUMMARY")
    print("="*60)

    def status(key, name):
        v = results.get(key)
        if v is None or (isinstance(v, pd.DataFrame) and v.empty):
            print(f"  {name:<40} FAILED / SKIPPED")
        elif isinstance(v, pd.DataFrame):
            yr_range = f"{int(v['year'].min())}-{int(v['year'].max())}" if "year" in v.columns else ""
            print(f"  {name:<40} OK  ({len(v)} rows {yr_range})")
        else:
            print(f"  {name:<40} OK")

    status("epo",              "EPO")
    status("nino4",            "Nino4 anomaly")
    status("amo",              "AMO (Atlantic Multidecadal Osc.)")
    status("z500",             "Z500 NE Pacific")
    status("slp_nepac",        "SLP NE Pacific (Aleutian Low)")
    status("ndbc_monthly",     "NDBC Buoy Monthly (NE Pacific)")
    status("synoptic_monthly", "Synoptic Gradients Monthly")
    status("hgt500_gradient",  "500mb Height Gradient (offshore/cascade)")
    status("nino12_tni",       "Nino1+2 + TNI")
    status("hyak",             "Hyak.net snowfall")
    for k in results.get("snotel_keys", []):
        status(f"snotel_{k}", f"SNOTEL {k}")

    nwac = results.get("nwac")
    if nwac and nwac.get("stations"):
        print(f"  {'NWAC current conditions':<40} {len(nwac['stations'])} stations")
    else:
        print(f"  {'NWAC current conditions':<40} FAILED / RESTRICTED")

    wsdot = results.get("wsdot")
    if wsdot and wsdot.get("passes"):
        print(f"  {'WSDOT pass conditions':<40} {len(wsdot['passes'])} passes")
    else:
        print(f"  {'WSDOT pass conditions':<40} FAILED / EMPTY")

    summit = results.get("summit")
    if summit and not summit.get("error"):
        print(f"  {'Summit snow report':<40} OK  ({summit})")
    else:
        print(f"  {'Summit snow report':<40} FAILED")

    # Write provenance manifest
    df_results = {k: v for k, v in results.items() if isinstance(v, pd.DataFrame)}
    write_data_manifest(df_results)

    print()
    print("  Next step: run forecast.py to retrain with new data")
    print("  (AMO + Z500 auto-incorporated via patch_fresh_telecons)")
    print("="*60)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Fetching New Predictors for Snoqualmie Snowpack Model")
    print("="*60)

    results = {}

    # ── Long-record ML predictors ─────────────────────────────────────────────
    results["epo"]             = fetch_epo()
    results["nino4"]           = fetch_nino4()
    results["amo"]             = fetch_amo()
    results["z500"]            = fetch_z500_nepac()
    results["slp_nepac"]       = fetch_slp_nepac()
    results["hyak"]            = fetch_hyak()

    # ── Marine + synoptic monthly aggregates ──────────────────────────────────
    results["ndbc_monthly"]      = fetch_ndbc_monthly()
    results["synoptic_monthly"]  = fetch_synoptic_monthly()

    # ── Additional teleconnection indices ──────────────────────────────────────
    results["hgt500_gradient"] = fetch_hgt500_gradient()
    results["nino12_tni"]      = fetch_nino12_tni()

    # ── SNOTEL stations (for current conditions display) ──────────────────────
    snotel_dict = fetch_all_snotel()
    results["snotel_keys"] = list(snotel_dict.keys())
    for k, df in snotel_dict.items():
        results[f"snotel_{k}"] = df

    # ── Current conditions (display only — not used as ML predictors) ─────────
    results["nwac"]   = fetch_nwac_current()
    results["wsdot"]  = fetch_wsdot_passes()
    results["summit"] = fetch_summit_snow_report()

    build_summary(results)


if __name__ == "__main__":
    main()
