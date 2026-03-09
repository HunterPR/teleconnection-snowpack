"""
forecast.py
===========
Snoqualmie Pass snowpack forecasting tool.

Pipeline:
  1. Load historical training data (Merged_Dataset.csv: 1950-2024)
  2. Patch with fresh teleconnection indices and Z500 NE Pacific data
  3. Engineer lagged features (0, 1, 2, 3 month lags)
  4. Train Ridge + Random Forest models (leave-one-year-out CV)
  5. Identify analog years (nearest-neighbor in teleconnection space)
  6. Forecast Feb-Apr 2026 SWE and snowfall
  7. Save results + generate plots

Targets:
  - WTEQ        : Snow water equivalent at Snoqualmie Pass (inches)
  - snow_inches : Monthly snowfall at Snoqualmie Pass (inches)

Key teleconnections used (from prior correlation analysis):
  ao, enso34, pdo, pna, qbo, np, pmm, wp, solar, epo, nino4_anom, z500_nepac_anom, amo
  + MJO indices: index4_140e, index5_160e, index6_120w, index7_40w
"""

import os
import json
import warnings

# Avoid joblib/loky CPU detection issues on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
WINTER_MONTHS   = [10, 11, 12, 1, 2, 3, 4]  # Oct-Apr
CORE_TELE = ["ao", "roni", "pdo", "pna", "qbo", "np", "pmm", "wp", "solar",
             "index4_140e_mjo", "index5_160e_mjo", "index6_120w_mjo", "index7_40w_mjo",
             "np_x_pna",        # ridge-blocking interaction term
             "epo",             # East Pacific Oscillation: NE Pacific ridge/trough
             "nino4_anom",      # Central Pacific SST anomaly (CP vs EP ENSO flavour)
             "z500_nepac_anom", # NE Pacific 500mb geopotential height anomaly (direct circulation driver)
             "amo",             # Atlantic Multidecadal Oscillation: ~60-80 yr hemispheric cycle
             # ── Marine & synoptic predictors (2007+, will be NaN-imputed for earlier years) ──
             "buoy_wvht",       # NE Pacific sig wave height: storm track intensity
             "buoy_pres",       # NE Pacific SLP at buoys: Aleutian Low strength proxy
             "buoy_wspd",       # NE Pacific surface wind speed
             "buoy_storm_days", # Days/month with SLP < 1000 hPa: active storm track count
             "syn_slp_gradient",     # SLP gradient offshore minus cascade (onshore flow strength, 2003+)
             # ── NE Pacific SLP anomaly (Aleutian Low index, 1948+) ──────────────────────
             "slp_nepac_anom",  # Aleutian Low strength anomaly: negative = deeper = more storms
             # ── 500mb height gradient: offshore (GoA) vs Cascade crest (1948+) ───────────
             "hgt500_gradient", # 500mb height offshore - cascade: positive = onshore flow
             # ── ENSO flavour indices (EP vs CP discrimination) ───────────────────────────
             "nino12_anom",     # Eastern Pacific SST anomaly (EP El Nino signal)
             "tni",             # Trans-Nino Index = nino12 - nino4 (EP vs CP ENSO type)
             ]

# Lags to build features for (months before the target month)
LAGS = [0, 1, 2, 3]

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# Blend snowfall forecast with climatology to reduce variance (experimental). 0 = model only, 0.5 = half climatology.
SNOW_CLIM_BLEND = 0.35

# Blend WTEQ forecast with climatology.  Best tune_backtest config:
#   Ridge+GBR | all features | clim=60% -> skill=+1.8%, corr=0.863
# Set to 0.0 to disable; loaded from tune_backtest_results.csv if available.
WTEQ_CLIM_BLEND = 0.60

# Preferred WTEQ model subset from tuning (None = full ensemble)
WTEQ_MODEL_SUBSET = ["Ridge", "GBR"]

def _apply_tune_config():
    """Load best tune_backtest config and update module-level WTEQ_CLIM_BLEND / WTEQ_MODEL_SUBSET."""
    global WTEQ_CLIM_BLEND, WTEQ_MODEL_SUBSET
    path = os.path.join(DATA, "tune_backtest_results.csv")
    if not os.path.exists(path):
        return
    try:
        rdf = pd.read_csv(path)
        if rdf.empty or "skill" not in rdf.columns:
            return
        rdf = rdf.sort_values("skill", ascending=False).reset_index(drop=True)
        row = rdf.iloc[0]
        label = str(row.get("label", ""))
        skill = row.get("skill", 0)

        # Parse from dedicated columns if present, else parse label
        if "clim_blend" in rdf.columns and pd.notna(row.get("clim_blend")):
            WTEQ_CLIM_BLEND = float(row["clim_blend"])
        else:
            # Parse "clim=60%" from label
            import re
            m = re.search(r"clim=(\d+)%", label)
            if m:
                WTEQ_CLIM_BLEND = int(m.group(1)) / 100.0

        if "model_names" in rdf.columns and pd.notna(row.get("model_names")):
            mn_str = str(row["model_names"])
            if mn_str != "ensemble":
                WTEQ_MODEL_SUBSET = [s.strip() for s in mn_str.split("|") if s.strip()]
            else:
                WTEQ_MODEL_SUBSET = None
        else:
            # Parse model names from label (before first " | ")
            parts = label.split(" | ")
            if parts:
                model_str = parts[0].split("(")[0].strip()  # strip alpha param
                if model_str == "ensemble":
                    WTEQ_MODEL_SUBSET = None
                else:
                    WTEQ_MODEL_SUBSET = [s.strip() for s in model_str.split("+") if s.strip()]

        print(f"   [tune] Best config: models={'+'.join(WTEQ_MODEL_SUBSET) if WTEQ_MODEL_SUBSET else 'ensemble'}, "
              f"clim_blend={WTEQ_CLIM_BLEND:.0%} (skill={skill:.1%})")
    except Exception:
        pass  # fall back to hardcoded defaults


# Teleconnection subsets used by tune_backtest (for loading best config)
TUNE_TELE_SUBSETS = {
    "core4":  ["ao", "roni", "pdo", "pna"],
    "core6":  ["ao", "roni", "pdo", "pna", "np", "epo"],
    "core8":  ["ao", "roni", "pdo", "pna", "np", "epo", "z500_nepac_anom"],
    "core10": ["ao", "roni", "pdo", "pna", "np", "epo", "z500_nepac_anom", "amo", "nao"],
}

# ── 1. Load & extend base dataset ─────────────────────────────────────────────

def load_base() -> pd.DataFrame:
    print("[1] Loading Merged_Dataset.csv ...")
    merged_path = os.path.join(BASE, "Merged_Dataset.csv")
    if not os.path.exists(merged_path):
        print("   Merged_Dataset.csv not found; building from data/ ...")
        try:
            from build_merged_dataset import build
            df = build()
            df.to_csv(merged_path, index=False)
            print(f"   Built and saved {merged_path} ({len(df)} rows)")
        except Exception as e:
            raise FileNotFoundError(
                "Merged_Dataset.csv not found and build_merged_dataset failed. "
                "Run: python fetch_data.py && python build_merged_dataset.py"
            ) from e
    df = pd.read_csv(merged_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    # Rename MJO columns to consistent short names
    renames = {
        "index4_140e_mjo": "index4_140e_mjo",
        "index5_160e_mjo": "index5_160e_mjo",
        "index6_120w_mjo": "index6_120w_mjo",
        "index7_40w_mjo":  "index7_40w_mjo",
    }
    df = df.rename(columns=renames)
    print(f"   {len(df)} rows, {len(df.columns)} cols | years {df.year.min()}-{df.year.max()}")
    return df


def patch_fresh_telecons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend / overwrite teleconnection columns with freshly downloaded data
    (which is more current and uses the official source formats).
    """
    print("[2] Patching with fresh teleconnection downloads ...")
    updates = {}

    # ONI → enso34
    oni = pd.read_csv(os.path.join(DATA, "oni.csv"))
    # ONI is 3-month seasons; convert to approximate calendar month
    # Season codes: DJF→Jan, JFM→Feb, FMA→Mar, MAM→Apr, AMJ→May,
    #               MJJ→Jun, JJA→Jul, JAS→Aug, ASO→Sep, SON→Oct, OND→Nov, NDJ→Dec
    seas_to_month = {
        "DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
        "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12,
    }
    oni["month"] = oni["season"].map(seas_to_month)
    oni = oni.rename(columns={"oni_anomaly": "enso34"})[["year","month","enso34"]].dropna()
    updates["enso34"] = oni

    # RONI (Relative Oceanic Niño Index) — primary ENSO predictor in models
    roni_path = os.path.join(DATA, "roni.csv")
    if os.path.exists(roni_path):
        roni = pd.read_csv(roni_path)
        roni[["year","month"]] = roni[["year","month"]].astype(int)
        if "roni" in roni.columns:
            updates["roni"] = roni[["year","month","roni"]]
            print(f"   RONI file found: {len(roni)} rows")
        else:
            updates["roni"] = oni.rename(columns={"enso34": "roni"})[["year","month","roni"]]
    else:
        updates["roni"] = oni.rename(columns={"enso34": "roni"})[["year","month","roni"]]
        print("   roni.csv not found — using ONI as roni fallback (run fetch_data.py for RONI)")

    # PDO
    pdo = pd.read_csv(os.path.join(DATA, "pdo.csv")).rename(columns={"pdo":"pdo"})
    pdo[["year","month"]] = pdo[["year","month"]].astype(int)
    updates["pdo"] = pdo[["year","month","pdo"]]

    # PNA
    pna = pd.read_csv(os.path.join(DATA, "pna.csv"))
    pna[["year","month"]] = pna[["year","month"]].astype(int)
    updates["pna"] = pna[["year","month","pna"]]

    # AO
    ao = pd.read_csv(os.path.join(DATA, "ao.csv"))
    ao[["year","month"]] = ao[["year","month"]].astype(int)
    updates["ao"] = ao[["year","month","ao"]]

    # NAO → no direct column in base, add as new feature
    nao = pd.read_csv(os.path.join(DATA, "nao.csv"))
    nao[["year","month"]] = nao[["year","month"]].astype(int)
    updates["nao"] = nao[["year","month","nao"]]

    # EPO (optional — created by fetch_new_predictors.py)
    epo_path = os.path.join(DATA, "epo.csv")
    if os.path.exists(epo_path):
        epo = pd.read_csv(epo_path)
        epo[["year","month"]] = epo[["year","month"]].astype(int)
        updates["epo"] = epo[["year","month","epo"]]
        print(f"   EPO file found: {len(epo)} rows")
    else:
        print("   epo.csv not found — run fetch_new_predictors.py to enable EPO feature")

    # Nino4 anomaly (optional — created by fetch_new_predictors.py)
    nino4_path = os.path.join(DATA, "nino4_anom.csv")
    if os.path.exists(nino4_path):
        nino4 = pd.read_csv(nino4_path)
        nino4[["year","month"]] = nino4[["year","month"]].astype(int)
        updates["nino4_anom"] = nino4[["year","month","nino4_anom"]]
        print(f"   nino4_anom file found: {len(nino4)} rows")
    else:
        print("   nino4_anom.csv not found — run fetch_new_predictors.py to enable Nino4 feature")

    # Z500 NE Pacific anomaly (500mb geopotential height, 45-65N 195-230E)
    z500_path = os.path.join(DATA, "z500_nepac.csv")
    if os.path.exists(z500_path):
        z500 = pd.read_csv(z500_path)
        z500[["year","month"]] = z500[["year","month"]].astype(int)
        updates["z500_nepac_anom"] = z500[["year","month","z500_nepac_anom"]]
        print(f"   z500_nepac_anom file found: {len(z500)} rows")
    else:
        print("   z500_nepac.csv not found — run fetch_new_predictors.py to compute Z500")

    # AMO (Atlantic Multidecadal Oscillation) — ~60-80 yr hemispheric cycle
    amo_path = os.path.join(DATA, "amo.csv")
    if os.path.exists(amo_path):
        amo = pd.read_csv(amo_path)
        amo[["year","month"]] = amo[["year","month"]].astype(int)
        updates["amo"] = amo[["year","month","amo"]]
        print(f"   amo file found: {len(amo)} rows")
    else:
        print("   amo.csv not found — run fetch_new_predictors.py to enable AMO feature")

    for col, src in updates.items():
        # Merge into df — update existing rows, add new rows
        src = src.copy()
        # Mark which rows exist in base
        df_years = set(zip(df["year"], df["month"]))
        src_new = src[~src.apply(lambda r: (r["year"], r["month"]) in df_years, axis=1)]
        # Update existing values
        merged = df.merge(src.rename(columns={col: col + "_fresh"}), on=["year","month"], how="left")
        if col in merged.columns:
            merged[col] = merged[col + "_fresh"].combine_first(merged[col])
        else:
            merged[col] = merged[col + "_fresh"]
        merged = merged.drop(columns=[col + "_fresh"])
        # Append new rows
        if not src_new.empty:
            # Create skeleton rows for genuinely new year/month combos
            extra = src_new.copy()
            for c in df.columns:
                if c not in extra.columns:
                    extra[c] = np.nan
            extra = extra[df.columns]
            merged = pd.concat([merged, extra], ignore_index=True)
        df = merged
        print(f"   patched {col}: {len(src)} source rows")

    return df


def patch_fresh_snotel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend WTEQ and snow_depth columns with fresh Snoqualmie SNOTEL data.
    After this, apply_sno_pass_first() overwrites WTEQ/snow_inches with pass-preferred
    values (Stampede or 20-year-corrected 908; DOT/ALP snowfall when available).
    """
    print("[3] Patching with fresh Snoqualmie SNOTEL data ...")
    snq = pd.read_csv(os.path.join(DATA, "snoqualmie_snotel.csv"))
    snq["year"]  = snq["year"].astype(int)
    snq["month"] = snq["month"].astype(int)
    # Map SNOTEL columns → Merged_Dataset columns
    # WTEQ = swe_in (Snoqualmie SNOTEL station #908)
    snq = snq.rename(columns={"swe_in": "WTEQ_snotel908", "snow_depth_in": "SNWD_snotel908"})

    # Merge into base
    df = df.merge(snq[["year","month","WTEQ_snotel908","SNWD_snotel908"]],
                  on=["year","month"], how="left")

    # Fill WTEQ from the SNOTEL station where the base column is missing
    if "WTEQ" in df.columns:
        df["WTEQ"] = df["WTEQ"].combine_first(df["WTEQ_snotel908"])
    else:
        df["WTEQ"] = df["WTEQ_snotel908"]

    print(f"   SNOTEL WTEQ patched: {df['WTEQ'].notna().sum()} non-null rows")
    return df


def patch_historical_snowfall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restore historical monthly snowfall from transformed_snow.csv (1950-2022).
    This fills the snow_inches column in the base dataset which Cursor's rebuild
    left empty. The pass-first correction (apply_sno_pass_first) will override
    2005+ rows with DOT/ALP pass station data where available.
    """
    fpath = os.path.join(DATA, "PSL CSV Files", "transformed_snow.csv")
    if not os.path.exists(fpath):
        print("   transformed_snow.csv not found — snow_inches not restored")
        return df
    raw = pd.read_csv(fpath)
    # Melt wide format (year + 12 monthly columns) to long format (year, month, snow_inches)
    month_map = {
        "january(snow_inches)": 1, "february(snow_inches)": 2,
        "march(snow_inches)": 3, "april(snow_inches)": 4,
        "may(snow_inches)": 5, "june(snow_inches)": 6,
        "july(snow_inches)": 7, "august(snow_inches)": 8,
        "september(snow_inches)": 9, "october(snow_inches)": 10,
        "november(snow_inches)": 11, "december(snow_inches)": 12,
    }
    rows = []
    for _, r in raw.iterrows():
        yr = int(r["year"])
        for col, mo in month_map.items():
            if col in r.index and pd.notna(r[col]):
                rows.append({"year": yr, "month": mo, "snow_hist": float(r[col])})
    hist = pd.DataFrame(rows)

    # Fill snow_inches: historical first, then existing data overrides
    if "snow_inches" not in df.columns:
        df["snow_inches"] = np.nan
    df = df.merge(hist, on=["year", "month"], how="left")
    df["snow_inches"] = df["snow_inches"].combine_first(df["snow_hist"])
    df = df.drop(columns=["snow_hist"])
    n_valid = df["snow_inches"].notna().sum()
    yr_range = f"{int(hist['year'].min())}-{int(hist['year'].max())}"
    print(f"   Restored historical snowfall: {n_valid} non-null rows ({yr_range})")
    return df


def apply_sno_pass_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sno-Pass-first targets: prefer Stampede Pass SNOTEL (or 20-yr corrected 908) for WTEQ,
    and DOT/ALP pass snowfall for snow_inches when available. Users discredit raw SNOTEL;
    this keeps outputs defensible as pass-representative.
    """
    try:
        from sno_pass_correction import build_pass_first_wteq, build_pass_first_snow_inches
    except ImportError:
        return df
    print("[3b] Applying Sno-Pass-first targets (Stampede / DOT-ALP preferred) ...")
    df = build_pass_first_wteq(df)
    n_wteq = df["WTEQ"].notna().sum()
    df = build_pass_first_snow_inches(df)
    n_snow = df["snow_inches"].notna().sum() if "snow_inches" in df.columns else 0
    print(f"   Pass-first WTEQ: {n_wteq} non-null | snow_inches: {n_snow} non-null")
    return df


def patch_ndbc_buoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge monthly NDBC NE Pacific buoy features into the training dataframe.
    Source: data/ndbc_monthly.csv (generated by fetch_new_predictors.py).
    Covers 2007+; earlier rows will have NaN (imputed during training).
    Valid upstream predictors: ocean/atmosphere state upstream of the Cascades.
    """
    fpath = os.path.join(DATA, "ndbc_monthly.csv")
    if not os.path.exists(fpath):
        return df
    buoy = pd.read_csv(fpath)
    buoy["year"]  = buoy["year"].astype(int)
    buoy["month"] = buoy["month"].astype(int)
    cols = [c for c in ["buoy_wvht", "buoy_pres", "buoy_wspd",
                         "buoy_wvht_max", "buoy_pres_min", "buoy_storm_days"]
            if c in buoy.columns]
    df = df.merge(buoy[["year", "month"] + cols], on=["year", "month"], how="left")
    n_valid = df["buoy_wvht"].notna().sum() if "buoy_wvht" in df.columns else 0
    print(f"   patched buoy: {n_valid} non-null rows "
          f"({int(buoy['year'].min())}-{int(buoy['year'].max())})")
    return df


def patch_synoptic_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge monthly synoptic gradient features into the training dataframe.
    Source: data/synoptic_monthly.csv (generated by fetch_new_predictors.py).
    Covers 2003+; earlier rows will have NaN.
    syn_hgt500_gradient: 500mb height offshore minus cascade (ridge indicator)
    syn_slp_gradient:    SLP gradient offshore minus cascade (onshore flow)
    """
    fpath = os.path.join(DATA, "synoptic_monthly.csv")
    if not os.path.exists(fpath):
        return df
    syn = pd.read_csv(fpath)
    syn["year"]  = syn["year"].astype(int)
    syn["month"] = syn["month"].astype(int)
    cols = [c for c in ["syn_hgt500_gradient", "syn_slp_gradient", "syn_thickness"]
            if c in syn.columns]
    df = df.merge(syn[["year", "month"] + cols], on=["year", "month"], how="left")
    # Report the column that actually has data (syn_slp_gradient, not syn_hgt500_gradient)
    n_valid = df["syn_slp_gradient"].notna().sum() if "syn_slp_gradient" in df.columns else 0
    print(f"   patched synoptic gradients: {n_valid} non-null rows "
          f"({int(syn['year'].min())}-{int(syn['year'].max())})")
    return df


def patch_slp_nepac(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NE Pacific monthly SLP anomaly (Aleutian Low index) from NCEP/NCAR reanalysis.
    Source: data/slp_nepac.csv (generated by fetch_new_predictors.py).
    Covers 1948+, fully overlapping training record.
    slp_nepac_anom: negative = stronger Aleutian Low = more storms = good for PNW snow.
    """
    fpath = os.path.join(DATA, "slp_nepac.csv")
    if not os.path.exists(fpath):
        return df
    slp = pd.read_csv(fpath)
    slp["year"]  = slp["year"].astype(int)
    slp["month"] = slp["month"].astype(int)
    # Drop pre-existing column to avoid _x/_y suffix collision on merge
    if "slp_nepac_anom" in df.columns:
        df = df.drop(columns=["slp_nepac_anom"])
    df = df.merge(slp[["year", "month", "slp_nepac_anom"]], on=["year", "month"], how="left")
    n_valid = df["slp_nepac_anom"].notna().sum()
    print(f"   patched slp_nepac: {n_valid} non-null rows "
          f"({int(slp['year'].min())}-{int(slp['year'].max())})")
    return df


def patch_hgt500_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge 500mb height gradient (offshore GoA minus Cascade crest) from NCEP/NCAR.
    Source: data/hgt500_gradient.csv (generated by fetch_new_predictors.py).
    Covers 1948+, fully overlapping training record.
    hgt500_gradient > 0: onshore flow (Pacific storms), < 0: Cascade ridge blocking.
    """
    fpath = os.path.join(DATA, "hgt500_gradient.csv")
    if not os.path.exists(fpath):
        return df
    g = pd.read_csv(fpath)
    g["year"]  = g["year"].astype(int)
    g["month"] = g["month"].astype(int)
    if "hgt500_gradient" not in g.columns:
        return df
    if "hgt500_gradient" in df.columns:
        df = df.drop(columns=["hgt500_gradient"])
    df = df.merge(g[["year", "month", "hgt500_gradient"]], on=["year", "month"], how="left")
    if "hgt500_gradient" in df.columns:
        n_valid = df["hgt500_gradient"].notna().sum()
        print(f"   patched hgt500_gradient: {n_valid} non-null rows "
              f"({int(g['year'].min())}-{int(g['year'].max())})")
    else:
        df["hgt500_gradient"] = np.nan
        print("   patched hgt500_gradient: column missing after merge (added as NaN)")
    return df


def patch_nino12_tni(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Nino1+2 SST anomaly and TNI (Trans-Nino Index) into training dataframe.
    Source: data/nino12_anom.csv and data/tni.csv (generated by fetch_new_predictors.py).
    nino12_anom: eastern Pacific SST; TNI = nino12_anom - nino4_anom (ENSO flavour).
    Covers 1950+.
    """
    n12_path = os.path.join(DATA, "nino12_anom.csv")
    tni_path = os.path.join(DATA, "tni.csv")
    if os.path.exists(n12_path):
        n12 = pd.read_csv(n12_path)
        n12[["year", "month"]] = n12[["year", "month"]].astype(int)
        if "nino12_anom" in df.columns:
            df = df.drop(columns=["nino12_anom"])
        df = df.merge(n12[["year", "month", "nino12_anom"]], on=["year", "month"], how="left")
        n_valid = df["nino12_anom"].notna().sum()
        print(f"   patched nino12_anom: {n_valid} non-null rows")
    if os.path.exists(tni_path):
        tni = pd.read_csv(tni_path)
        tni[["year", "month"]] = tni[["year", "month"]].astype(int)
        if "tni" in df.columns:
            df = df.drop(columns=["tni"])
        df = df.merge(tni[["year", "month", "tni"]], on=["year", "month"], how="left")
        n_valid = df["tni"].notna().sum()
        print(f"   patched tni: {n_valid} non-null rows")
    return df


def patch_additional_snotel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge additional SNOTEL station WTEQ columns into df as regional predictors.
    Only loads files that exist (created by fetch_new_predictors.py).
    """
    print("[3b] Patching additional SNOTEL stations as regional predictors ...")
    stations = {
        "stevens":   "WTEQ_stevens",
        "whitepass": "WTEQ_whitepass",
        "lyman":     "WTEQ_lyman",
        "corral":    "WTEQ_corral",
    }
    loaded = []
    for key, col_name in stations.items():
        path = os.path.join(DATA, f"snotel_{key}.csv")
        if not os.path.exists(path):
            continue
        try:
            sno = pd.read_csv(path)
            sno["year"]  = sno["year"].astype(int)
            sno["month"] = sno["month"].astype(int)
            # The fetch_new_predictors.py names columns WTEQ_{key}
            wteq_src = f"WTEQ_{key}"
            if wteq_src not in sno.columns:
                # Fallback: find any WTEQ-like column
                wteq_src = next((c for c in sno.columns if "WTEQ" in c.upper()), None)
            if wteq_src is None:
                print(f"   No WTEQ column in snotel_{key}.csv — skipping")
                continue
            sno = sno.rename(columns={wteq_src: col_name})
            # Avoid duplicate column if already present
            if col_name in df.columns:
                df = df.drop(columns=[col_name])
            df = df.merge(sno[["year","month",col_name]], on=["year","month"], how="left")
            n_valid = df[col_name].notna().sum()
            loaded.append(col_name)
            print(f"   Merged {col_name}: {n_valid} non-null rows")
        except Exception as e:
            print(f"   Could not load snotel_{key}.csv: {e}")

    if not loaded:
        print("   No additional SNOTEL files found (run fetch_new_predictors.py first)")
    else:
        print(f"   Added SNOTEL predictors: {loaded}")
    return df


# ── 2. Build feature matrix ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, add lagged teleconnection values.
    Lag N means the teleconnection value from N months before this row's month.
    Also adds np_x_pna: the NP × PNA interaction term, a proxy for the
    ridge-blocking pattern that suppresses Cascades snowfall even during La Niña.
    High NP + high PNA = ridge over NE Pacific = jet deflects north of Cascades.
    """
    print("[4] Engineering lagged features ...")
    df = df.sort_values(["year","month"]).reset_index(drop=True)

    # Compute ridge-blocking interaction: NP × PNA
    # Both positive = Aleutian Low weak AND ridge over NE Pacific = bad for PNW snow
    if "np" in df.columns and "pna" in df.columns:
        df["np_x_pna"] = df["np"] * df["pna"]
        print("   Added np_x_pna ridge-blocking interaction feature")

    # Build a lookup dict: (year, month) → {col: val}
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    idx = df.set_index(["year","month"])[tele_cols].to_dict("index")

    def get_lagged_val(year, month, col, lag):
        # Shift back `lag` months
        total = (year - 1) * 12 + month - lag
        ly = (total - 1) // 12 + 1
        lm = (total - 1) % 12 + 1
        return idx.get((ly, lm), {}).get(col, np.nan)

    # Add lagged columns
    new_cols = {}
    for col in tele_cols:
        for lag in LAGS:
            cname = f"{col}_lag{lag}"
            new_cols[cname] = df.apply(lambda r: get_lagged_val(r["year"], r["month"], col, lag), axis=1)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    # Also add "nao" lags if present
    if "nao" in df.columns:
        for lag in LAGS:
            cname = f"nao_lag{lag}"
            new_cols2 = df.apply(lambda r: get_lagged_val(r["year"], r["month"], "nao", lag), axis=1)
            df[cname] = new_cols2

    print(f"   Added {len(LAGS) * len(tele_cols)} teleconnection lag features")
    return df


def make_feature_names(tele_cols, include_nao=True, extra_cols=None):
    feats = []
    for col in tele_cols:
        for lag in LAGS:
            feats.append(f"{col}_lag{lag}")
    if include_nao:
        for lag in LAGS:
            feats.append(f"nao_lag{lag}")
    if extra_cols:
        for col in extra_cols:
            for lag in LAGS:
                feats.append(f"{col}_lag{lag}")
    return feats


# ── 3. Model training & CV ─────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, target: str, months: list = WINTER_MONTHS):
    """
    Train an expanded model suite on all available winter-month data.
    Models: Ridge, BayesianRidge, ElasticNet, KNN, SVR,
            RandomForest, ExtraTrees, GBR, XGBoost (if available).
    Returns fitted pipelines + feature metadata.
    """
    tele_cols  = [c for c in CORE_TELE if c in df.columns]
    feat_names = make_feature_names(tele_cols, include_nao="nao" in df.columns)
    feat_names = [f for f in feat_names if f in df.columns]

    # Filter to winter months + non-null target
    mask = df["month"].isin(months) & df[target].notna()
    sub = df[mask].copy()

    # Drop rows with too many missing features.
    # Start at 50% threshold; if too few rows survive (<180), lower to 30%.
    # New predictors with partial coverage (buoy 2007+, synoptic 2003+) are
    # NaN-imputed for older rows — the model still uses them for modern years.
    frac = 0.5
    sub_trial = sub.dropna(subset=feat_names, thresh=int(frac * len(feat_names)))
    if len(sub_trial) < 180 and len(sub) > len(sub_trial):
        frac = 0.3
        sub_trial = sub.dropna(subset=feat_names, thresh=int(frac * len(feat_names)))
        print(f"   Adaptive threshold: lowered to {frac:.0%} ({int(frac * len(feat_names))} feats) "
              f"-> {len(sub_trial)} rows (was {len(sub)})")
    sub = sub_trial

    X_df_raw = sub.reindex(columns=feat_names)
    all_nan_cols = X_df_raw.columns[X_df_raw.isna().all()].tolist()
    if all_nan_cols:
        X_df_raw = X_df_raw.drop(columns=all_nan_cols)
    feat_names = list(X_df_raw.columns)

    X_raw = X_df_raw.values
    y     = sub[target].values

    print(f"\n[5] Training on target='{target}': {len(X_raw)} rows, {len(feat_names)} features")

    # ── Define all model pipelines ────────────────────────────────────────────
    IMP = lambda: SimpleImputer(strategy="mean")
    SCL = lambda: StandardScaler()

    pipes = {
        "Ridge":       Pipeline([("imp", IMP()), ("scl", SCL()), ("m", Ridge(alpha=100.0))]),
        "BayesRidge":  Pipeline([("imp", IMP()), ("scl", SCL()), ("m", BayesianRidge())]),
        "ElasticNet":  Pipeline([("imp", IMP()), ("scl", SCL()), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5))]),
        "KNN":         Pipeline([("imp", IMP()), ("scl", SCL()), ("m", KNeighborsRegressor(n_neighbors=7, weights="distance"))]),
        "SVR":         Pipeline([("imp", IMP()), ("scl", SCL()), ("m", SVR(C=10, epsilon=0.5, kernel="rbf"))]),
        "RF":          Pipeline([("imp", IMP()), ("m",  RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                                                               min_samples_leaf=3, random_state=42))]),
        "ExtraTrees":  Pipeline([("imp", IMP()), ("m",  ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                                                                              min_samples_leaf=3, random_state=42))]),
        "GBR":         Pipeline([("imp", IMP()), ("m",  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                                                   learning_rate=0.05, subsample=0.8,
                                                                                   random_state=42))]),
    }
    if HAS_XGB:
        pipes["XGBoost"] = Pipeline([("imp", IMP()), ("m", XGBRegressor(n_estimators=200, max_depth=4,
                                                                          learning_rate=0.05, subsample=0.8,
                                                                          colsample_bytree=0.7, random_state=42,
                                                                          verbosity=0))])

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    cv_results = {}
    for name, pipe in pipes.items():
        try:
            r2s  = cross_val_score(pipe, X_raw, y, cv=5, scoring="r2")
            rmses = np.sqrt(-cross_val_score(pipe, X_raw, y, cv=5,
                                             scoring="neg_mean_squared_error"))
            cv_results[name] = {"r2_mean": r2s.mean(), "r2_std": r2s.std(),
                                 "rmse_mean": rmses.mean(), "rmse_std": rmses.std()}
            print(f"   {name:<12} CV  R²={r2s.mean():.3f}±{r2s.std():.3f}  "
                  f"RMSE={rmses.mean():.2f}±{rmses.std():.2f}")
        except Exception as e:
            print(f"   {name:<12} FAILED: {e}")
            cv_results[name] = {"r2_mean": np.nan}

    # ── Fit on all data ────────────────────────────────────────────────────────
    fitted = {}
    for name, pipe in pipes.items():
        if not np.isnan(cv_results[name]["r2_mean"]):
            pipe.fit(X_raw, y)
            fitted[name] = pipe

    # Weighted ensemble: weight each model by max(0, CV R²)
    weights = {n: max(0, cv_results[n]["r2_mean"]) for n in fitted}
    total_w = sum(weights.values()) or 1.0
    weights = {n: w / total_w for n, w in weights.items()}

    preds = np.column_stack([pipe.predict(X_raw) for pipe in fitted.values()])
    w_arr = np.array([weights[n] for n in fitted])
    y_pred_ens = preds @ w_arr
    r2_ens = r2_score(y, y_pred_ens)
    print(f"   Weighted ensemble in-sample R²={r2_ens:.3f}")

    # Legacy aliases for backward compatibility
    imputer = SimpleImputer(strategy="mean")
    X_imp   = imputer.fit_transform(X_raw)
    X_df    = pd.DataFrame(X_imp, columns=feat_names)

    # Save CV table
    cv_df = pd.DataFrame(cv_results).T.reset_index().rename(columns={"index": "model"})
    cv_df.to_csv(os.path.join(DATA, f"cv_scores_{target}.csv"), index=False)

    result = {
        **fitted,                    # all fitted pipelines by name
        "ridge":    fitted.get("Ridge"),
        "rf":       fitted.get("RF"),
        "gbr":      fitted.get("GBR"),
        "weights":  weights,
        "features": feat_names,
        "cv":       cv_results,
        "X":        X_df,
        "X_raw":    X_raw,
        "y":        y,
        "sub":      sub,
    }
    return result


def _get_pipelines():
    """Return the same model pipelines used in train_models (for backtest)."""
    IMP = lambda: SimpleImputer(strategy="mean")
    SCL = lambda: StandardScaler()
    pipes = {
        "Ridge":       Pipeline([("imp", IMP()), ("scl", SCL()), ("m", Ridge(alpha=100.0))]),
        "BayesRidge":  Pipeline([("imp", IMP()), ("scl", SCL()), ("m", BayesianRidge())]),
        "ElasticNet":  Pipeline([("imp", IMP()), ("scl", SCL()), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5))]),
        "KNN":         Pipeline([("imp", IMP()), ("scl", SCL()), ("m", KNeighborsRegressor(n_neighbors=7, weights="distance"))]),
        "SVR":         Pipeline([("imp", IMP()), ("scl", SCL()), ("m", SVR(C=10, epsilon=0.5, kernel="rbf"))]),
        "RF":          Pipeline([("imp", IMP()), ("m",  RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                                                               min_samples_leaf=3, random_state=42))]),
        "ExtraTrees":  Pipeline([("imp", IMP()), ("m",  ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                                                                              min_samples_leaf=3, random_state=42))]),
        "GBR":         Pipeline([("imp", IMP()), ("m",  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                                                   learning_rate=0.05, subsample=0.8,
                                                                                   random_state=42))]),
    }
    if HAS_XGB:
        pipes["XGBoost"] = Pipeline([("imp", IMP()), ("m", XGBRegressor(n_estimators=200, max_depth=4,
                                                                          learning_rate=0.05, subsample=0.8,
                                                                          colsample_bytree=0.7, random_state=42,
                                                                          verbosity=0))])
    return pipes


def load_best_tune_config():
    """
    Load the best config from data/tune_backtest_results.csv (written by tune_backtest.py).
    Returns (label, model_names, tele_subset, clim_blend, pipeline_overrides) or None if missing/invalid.
    """
    path = os.path.join(DATA, "tune_backtest_results.csv")
    if not os.path.exists(path):
        return None
    try:
        rdf = pd.read_csv(path)
        if rdf.empty or "skill" not in rdf.columns:
            return None
        rdf = rdf.sort_values("skill", ascending=False).reset_index(drop=True)
        row = rdf.iloc[0]
        label = str(row.get("label", ""))
        mn_str = row.get("model_names", "ensemble")
        model_names = None if mn_str == "ensemble" else [s.strip() for s in str(mn_str).split("|") if s.strip()]
        tele_key = str(row.get("tele_subset", "all")).strip().lower()
        tele_subset = None if tele_key == "all" else TUNE_TELE_SUBSETS.get(tele_key)
        clim_blend = row.get("clim_blend")
        if pd.isna(clim_blend):
            clim_blend = None
        else:
            clim_blend = float(clim_blend)
            if clim_blend == 0:
                clim_blend = None
        ridge_alpha = row.get("ridge_alpha")
        pipeline_overrides = None
        if pd.notna(ridge_alpha) and model_names and "Ridge" in model_names:
            try:
                pipeline_overrides = {"Ridge": {"m__alpha": int(float(ridge_alpha))}}
            except (ValueError, TypeError):
                pass
        return (label, model_names, tele_subset, clim_blend, pipeline_overrides)
    except Exception:
        return None


def run_backtest(
    df: pd.DataFrame,
    target: str,
    months: list = None,
    model_names: list = None,
    tele_subset: list = None,
    min_train_rows: int = 80,
    clim_blend: float = None,
    pipeline_overrides: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Leave-one-year-out backtest: for each year with target data, train on all other years,
    predict that year's winter months, compare to actuals.

    tele_subset: if set, use only these teleconnection columns (e.g. ["ao","roni","pdo","pna"])
                 to reduce overfitting. Default None = use all CORE_TELE in df.
    clim_blend: if set in [0,1], final prediction = (1-clim_blend)*pred + clim_blend*clim_pred
                (e.g. 0.5 = 50% model, 50% climatology) to shrink toward mean and often improve RMSE.
    pipeline_overrides: optional dict, e.g. {"Ridge": {"m__alpha": 200}} to override pipeline params
                        (use sklearn Pipeline param syntax, e.g. m__alpha for Ridge in step "m").

    Returns dict with: rmse, rmse_clim, skill (1 - rmse/rmse_clim), correlation, bias,
    n_points, results (list of {year, month, actual, pred, clim_pred}).
    """
    if months is None:
        months = WINTER_MONTHS

    if tele_subset is not None:
        tele_cols = [c for c in tele_subset if c in df.columns]
    else:
        tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_names = make_feature_names(tele_cols, include_nao="nao" in df.columns)
    feat_names = [f for f in feat_names if f in df.columns]

    winter = df[df["month"].isin(months) & df[target].notna()].copy()
    if winter.empty:
        return {"rmse": np.nan, "rmse_clim": np.nan, "skill": np.nan, "correlation": np.nan,
                "bias": np.nan, "n_points": 0, "results": []}

    test_years = sorted(winter["year"].unique())
    if len(test_years) < 3:
        return {"rmse": np.nan, "rmse_clim": np.nan, "skill": np.nan, "correlation": np.nan,
                "bias": np.nan, "n_points": 0, "results": []}

    pipes = _get_pipelines()
    if model_names:
        pipes = {k: v for k, v in pipes.items() if k in model_names}
    if not pipes:
        pipes = _get_pipelines()

    clim = df[df["month"].isin(months) & df[target].notna()].groupby("month")[target].mean().to_dict()

    results = []
    for test_year in test_years:
        train_mask = (df["year"] != test_year) & df["month"].isin(months) & df[target].notna()
        sub = df.loc[train_mask].copy()
        frac = 0.5
        sub = sub.dropna(subset=feat_names, thresh=int(frac * len(feat_names)))
        if len(sub) < min_train_rows:
            frac = 0.3
            sub = df.loc[train_mask].dropna(subset=feat_names, thresh=int(frac * len(feat_names)))
            if len(sub) < min_train_rows:
                continue
        X_raw = sub.reindex(columns=feat_names).dropna(axis=1, how="all")
        feat_use = list(X_raw.columns)
        X_raw = X_raw.values
        y_train = sub[target].values

        fitted = {}
        for name, pipe in pipes.items():
            try:
                pipe = clone(pipe)
                if pipeline_overrides and name in pipeline_overrides:
                    pipe.set_params(**pipeline_overrides[name])
                pipe.fit(X_raw, y_train)
                fitted[name] = pipe
            except Exception:
                pass
        if not fitted:
            continue
        weights = {n: 1.0 / len(fitted) for n in fitted}

        test_months = winter[winter["year"] == test_year]
        for _, row in test_months.iterrows():
            month = int(row["month"])
            actual = float(row[target])
            row_df = build_current_row(df, tele_cols, test_year, month)
            row_df = row_df.reindex(columns=feat_use)
            if row_df.isna().all(axis=None):
                continue
            X_row = row_df.values
            preds = [fitted[n].predict(X_row)[0] for n in fitted]
            pred = sum(preds[i] * list(weights.values())[i] for i in range(len(fitted)))
            clim_pred = clim.get(month, np.nan)
            if np.isnan(clim_pred):
                clim_pred = winter[target].mean()
            if clim_blend is not None and 0 <= clim_blend <= 1:
                pred = (1.0 - clim_blend) * pred + clim_blend * clim_pred
            results.append({
                "year": test_year, "month": month,
                "actual": actual, "pred": pred, "clim_pred": clim_pred,
            })

    if not results:
        return {"rmse": np.nan, "rmse_clim": np.nan, "skill": np.nan, "correlation": np.nan,
                "bias": np.nan, "n_points": 0, "results": []}

    actuals = np.array([r["actual"] for r in results])
    preds = np.array([r["pred"] for r in results])
    clim_preds = np.array([r["clim_pred"] for r in results])

    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    rmse_clim = float(np.sqrt(np.mean((actuals - clim_preds) ** 2)))
    skill = 1.0 - (rmse / rmse_clim) if rmse_clim > 0 else np.nan
    corr = np.corrcoef(actuals, preds)[0, 1] if len(actuals) > 1 else np.nan
    bias = float(np.mean(preds) - np.mean(actuals))

    out = {
        "rmse": rmse,
        "rmse_clim": rmse_clim,
        "skill": skill,
        "correlation": corr,
        "bias": bias,
        "n_points": len(results),
        "results": results,
    }
    if verbose:
        print(f"   Backtest {target}: n={out['n_points']}  RMSE={rmse:.2f}  RMSE_clim={rmse_clim:.2f}  "
              f"skill={skill:.2%}  corr={corr:.3f}  bias={bias:.2f}")
    return out


# ── 4. Forecast current season ─────────────────────────────────────────────────

def build_current_row(df: pd.DataFrame, tele_cols: list, target_year: int, target_month: int) -> pd.DataFrame:
    """
    Build a single-row feature vector for (target_year, target_month).

    If the row already exists in df (observed data), use it directly.
    If the row is a future/missing month, compute each lag feature from
    whichever past months ARE available in df rather than returning all-NaN.
    This is the fix for all forecast months collapsing to the same prediction.
    """
    include_nao = "nao" in df.columns
    all_tele = [c for c in tele_cols if c in df.columns]
    if include_nao and "nao" not in all_tele:
        all_tele = all_tele + ["nao"]

    feat_names = make_feature_names([c for c in tele_cols if c in df.columns],
                                    include_nao=include_nao)
    feat_names = [f for f in feat_names if f in df.columns]

    # If the row exists, use it as-is
    row = df[(df["year"] == target_year) & (df["month"] == target_month)]
    if len(row) > 0 and not row[feat_names].isnull().all(axis=None):
        return pd.DataFrame([row[feat_names].iloc[0].to_dict()], columns=feat_names)

    # Row is missing or entirely NaN — recompute lags from available history
    # Build a fast lookup: (year, month) → {col: value}
    available_tele = [c for c in all_tele if c in df.columns]
    lookup = (df.set_index(["year", "month"])[available_tele]
                .to_dict("index"))

    row_dict = {}
    for feat in feat_names:
        val = np.nan
        for col in available_tele:
            for lag in LAGS:
                if feat == f"{col}_lag{lag}":
                    # Compute which (year, month) this lag points to
                    total = (target_year - 1) * 12 + target_month - lag
                    ly = (total - 1) // 12 + 1
                    lm = (total - 1) % 12 + 1
                    val = lookup.get((ly, lm), {}).get(col, np.nan)
                    break
            else:
                continue
            break
        row_dict[feat] = val

    return pd.DataFrame([row_dict], columns=feat_names)


def forecast_season(df, models_wteq, models_snow=None, target_year=2026):
    """Forecast SWE (WTEQ) and snowfall for remaining winter months.
    Snowfall is blended with climatology (SNOW_CLIM_BLEND) to reduce variance.
    Layer 2 station telemetry blending adjusts current-month forecasts with actual pace."""
    subset_label = "+".join(WTEQ_MODEL_SUBSET) if WTEQ_MODEL_SUBSET else "full ensemble"
    blend_label = f"{WTEQ_CLIM_BLEND:.0%} clim" if WTEQ_CLIM_BLEND > 0 else "no blend"
    print(f"\n[6] Forecasting {target_year-1}/{target_year} winter season ...")
    print(f"    WTEQ: {subset_label}, {blend_label}")
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_wteq = models_wteq["features"]
    feat_snow = (models_snow or {}).get("features") or []
    forecast_months = [2, 3, 4]

    # Layer 2: load pre-computed station nowcast for current-month blending
    # (run `python nowcast.py` or pace pre-compute before forecasting)
    layer2_pace = {}
    layer2_sounding = {}
    pace_path = os.path.join(DATA, "nowcast_pace.json")
    if os.path.exists(pace_path):
        try:
            with open(pace_path) as f:
                raw_pace = json.load(f)
            for m in forecast_months:
                pace = raw_pace.get(str(m))
                if pace and "error" not in pace and pace.get("days_elapsed", 0) > 3:
                    layer2_pace[m] = pace
                    print(f"   Layer 2 ({pace.get('station','?')}): month {m} -> "
                          f"{pace['actual_snowfall_in']}\" snow in {pace['days_elapsed']}d, "
                          f"pace {pace['pace_snowfall_in']}\"")
            # Load sounding forecast data
            snd = raw_pace.get("sounding", {})
            if snd:
                layer2_sounding = snd
                fzl = snd.get("freezing_level_forecast", {})
                sf = snd.get("snowfall_possible_hours", {})
                print(f"   Layer 2 sounding: freezing level {fzl.get('current_ft','?')}' "
                      f"(48h: {fzl.get('min_48h_ft','?')}-{fzl.get('max_48h_ft','?')}'), "
                      f"snowfall {sf.get('next_48h','?')}/48h, {sf.get('next_120h','?')}/120h")
        except Exception as e:
            print(f"   Layer 2 pace load failed: {e}")
    else:
        print("   Layer 2: no nowcast_pace.json (run nowcast.py first)")

    # Get SNOTEL SWE at start of each month for blending
    snotel_swe = {}
    snotel_path = os.path.join(DATA, "snoqualmie_snotel.csv")
    if os.path.exists(snotel_path):
        snq = pd.read_csv(snotel_path)
        snq["year"] = snq["year"].astype(int)
        snq["month"] = snq["month"].astype(int)
        for m in forecast_months:
            row = snq[(snq["year"] == target_year) & (snq["month"] == m)]
            if len(row) > 0 and "swe_in" in row.columns:
                val = row["swe_in"].iloc[0]
                if pd.notna(val):
                    snotel_swe[m] = float(val)

    records = []
    for month in forecast_months:
        row_w = build_current_row(df, tele_cols, target_year, month)
        row_w = row_w.reindex(columns=feat_wteq)
        X_arr = row_w.values
        model_names_w = [k for k in models_wteq if isinstance(models_wteq[k], Pipeline)]
        preds_all_w = {n: models_wteq[n].predict(X_arr)[0] for n in model_names_w}

        # Use tuned model subset (WTEQ_MODEL_SUBSET) if available, else full ensemble
        subset_w = WTEQ_MODEL_SUBSET
        if subset_w:
            avail = [n for n in subset_w if n in preds_all_w]
            if avail:
                p_ens_w = np.mean([preds_all_w[n] for n in avail])
            else:
                weights_w = models_wteq.get("weights", {})
                p_ens_w = sum(preds_all_w[n] * weights_w.get(n, 0) for n in preds_all_w) if weights_w else np.mean(list(preds_all_w.values()))
        else:
            weights_w = models_wteq.get("weights", {})
            p_ens_w = sum(preds_all_w[n] * weights_w.get(n, 0) for n in preds_all_w) if weights_w else np.mean(list(preds_all_w.values()))

        # Apply WTEQ climatology blend (from tune_backtest best config)
        hist_w = df[(df["month"] == month) & df["WTEQ"].notna()]["WTEQ"]
        if WTEQ_CLIM_BLEND > 0:
            clim_w = float(hist_w.mean()) if len(hist_w) > 0 else p_ens_w
            p_ens_w_raw = p_ens_w
            p_ens_w = (1.0 - WTEQ_CLIM_BLEND) * p_ens_w + WTEQ_CLIM_BLEND * clim_w

        p_spread_w = float(np.std(list(preds_all_w.values()))) if len(preds_all_w) > 1 else 0.0
        pct_w = (hist_w < p_ens_w).mean() * 100

        rec = {
            "year": target_year, "month": month, "month_name": MONTH_NAMES[month],
            "wteq_ridge": round(preds_all_w.get("Ridge", p_ens_w), 2),
            "wteq_rf": round(preds_all_w.get("RF", p_ens_w), 2),
            "wteq_gbr": round(preds_all_w.get("GBR", p_ens_w), 2),
            "wteq_ensemble": round(p_ens_w, 2),
            "wteq_spread": round(p_spread_w, 2),
            "wteq_hist_mean": round(hist_w.mean(), 2), "wteq_hist_std": round(hist_w.std(), 2),
            "wteq_pct": round(pct_w, 1),
            **{f"wteq_{n.lower()}": round(v, 2) for n, v in preds_all_w.items()},
        }

        if models_snow and feat_snow:
            row_s = build_current_row(df, tele_cols, target_year, month).reindex(columns=feat_snow)
            X_s = row_s.values
            model_names_s = [k for k in models_snow if isinstance(models_snow[k], Pipeline)]
            preds_all_s = {n: models_snow[n].predict(X_s)[0] for n in model_names_s}
            weights_s = models_snow.get("weights", {})
            p_ens_s = sum(preds_all_s[n] * weights_s.get(n, 0) for n in preds_all_s) if weights_s else np.mean(list(preds_all_s.values()))
            p_spread_s = float(np.std(list(preds_all_s.values()))) if len(preds_all_s) > 1 else 0.0
            hist_s = df[(df["month"] == month) & df["snow_inches"].notna()]["snow_inches"]
            clim_s = float(hist_s.mean()) if len(hist_s) > 0 else 0.0
            p_ens_s = (1.0 - SNOW_CLIM_BLEND) * p_ens_s + SNOW_CLIM_BLEND * clim_s
            pct_s = (hist_s < p_ens_s).mean() * 100 if len(hist_s) > 0 else 50.0
            rec["snow_ridge"] = round(preds_all_s.get("Ridge", p_ens_s), 2)
            rec["snow_rf"] = round(preds_all_s.get("RF", p_ens_s), 2)
            rec["snow_gbr"] = round(preds_all_s.get("GBR", p_ens_s), 2)
            rec["snow_ensemble"] = round(p_ens_s, 2)
            rec["snow_spread"] = round(p_spread_s, 2)
            rec["snow_hist_mean"] = round(clim_s, 2)
            rec["snow_hist_std"] = round(hist_s.std(), 2) if len(hist_s) > 0 else 0.0
            rec["snow_pct"] = round(pct_s, 1)
            rec.update({f"snow_{n.lower()}": round(v, 2) for n, v in preds_all_s.items()})

        # Layer 2 blending: adjust current-month forecast with actual station pace
        if month in layer2_pace:
            pace = layer2_pace[month]
            swe_start = snotel_swe.get(month)
            try:
                days_e = pace["days_elapsed"]
                days_t = pace["days_in_month"]
                w = min(1.0, days_e / days_t)  # actual data weight
                l1_snow = rec.get("snow_ensemble", 0)
                l1_swe = rec.get("wteq_ensemble", 0)
                pace_snow = pace["pace_snowfall_in"]
                blended_snow = round(w * pace_snow + (1 - w) * l1_snow, 1)
                # SWE: month-start SWE + observed precip gain at pace
                if swe_start is not None and pace.get("swe_gain_est_in") is not None:
                    swe_gain_pace = pace["swe_gain_est_in"] * (days_t / max(1, days_e))
                    blended_swe = round(swe_start + swe_gain_pace, 1)
                else:
                    blended_swe = l1_swe

                rec["snow_layer1"] = l1_snow
                rec["wteq_layer1"] = l1_swe
                rec["snow_ensemble"] = blended_snow
                rec["wteq_ensemble"] = blended_swe
                rec["layer2_weight"] = round(w, 2)
                rec["layer2_pace_snow"] = pace_snow
                rec["layer2_actual_snow"] = pace["actual_snowfall_in"]
                # Recalculate percentiles with blended values
                hist_s2 = df[(df["month"] == month) & df["snow_inches"].notna()]["snow_inches"]
                if len(hist_s2) > 0:
                    rec["snow_pct"] = round((hist_s2 < blended_snow).mean() * 100, 1)
                hist_w2 = df[(df["month"] == month) & df["WTEQ"].notna()]["WTEQ"]
                rec["wteq_pct"] = round((hist_w2 < blended_swe).mean() * 100, 1)
                print(f"   Layer 2 blend for {MONTH_NAMES[month]}: "
                      f"snow {l1_snow:.1f}\" -> {blended_snow:.1f}\" | "
                      f"SWE {l1_swe:.1f}\" -> {blended_swe:.1f}\"")
            except Exception as e:
                print(f"   Layer 2 blend failed for {MONTH_NAMES[month]}: {e}")

        records.append(rec)

    # Attach sounding context to forecast output
    fc_df = pd.DataFrame(records)
    if layer2_sounding:
        fzl = layer2_sounding.get("freezing_level_forecast", {})
        sf = layer2_sounding.get("snowfall_possible_hours", {})
        sm = layer2_sounding.get("snowmaking_windows", {})
        w850 = layer2_sounding.get("wind_850hPa_48h", {})
        fc_df.attrs["sounding"] = {
            "freezing_level_current_ft": fzl.get("current_ft"),
            "freezing_level_48h_min_ft": fzl.get("min_48h_ft"),
            "freezing_level_48h_max_ft": fzl.get("max_48h_ft"),
            "snow_level_ft": layer2_sounding.get("snow_level_ft"),
            "snowfall_hours_48h": sf.get("next_48h"),
            "snowfall_hours_120h": sf.get("next_120h"),
            "snowmaking_good_48h": sm.get("good_hours_48h"),
            "current_wetbulb_f": sm.get("current_wetbulb_f"),
            "wind_850_dir": w850.get("mean_dir_deg"),
            "wind_850_mph": w850.get("mean_speed_mph"),
        }
    return fc_df


def build_forecast_vs_actual_recent(
    df: pd.DataFrame,
    models_wteq: dict,
    models_snow: dict = None,
    n_snow_months: int = 6,
) -> pd.DataFrame:
    """Compare WTEQ and (optionally) snowfall predictions to actuals for the last n winter months."""
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_wteq = models_wteq.get("features", [])
    feat_snow = (models_snow or {}).get("features", [])
    if not feat_wteq:
        return pd.DataFrame()
    winter = df[df["month"].isin(WINTER_MONTHS)].copy()
    winter = winter[winter["WTEQ"].notna()]
    if winter.empty:
        return pd.DataFrame()
    winter = winter.sort_values(["year", "month"], ascending=[False, False]).head(n_snow_months)
    if winter.empty:
        return pd.DataFrame()

    rows = []
    for _, r in winter.iterrows():
        yr, mo = int(r["year"]), int(r["month"])
        row_w = build_current_row(df, tele_cols, yr, mo)
        row_w = row_w.reindex(columns=feat_wteq)
        X_w = row_w.values
        model_names_w = [k for k in models_wteq if isinstance(models_wteq[k], Pipeline)]
        weights_w = models_wteq.get("weights", {})
        pred_w = sum(models_wteq[n].predict(X_w)[0] * weights_w.get(n, 0) for n in model_names_w) if weights_w else np.mean([models_wteq[n].predict(X_w)[0] for n in model_names_w])
        actual_w = float(r["WTEQ"])
        actual_s = float(r["snow_inches"]) if "snow_inches" in r.index and pd.notna(r.get("snow_inches")) else np.nan
        pred_s = np.nan
        if models_snow and feat_snow:
            row_s = build_current_row(df, tele_cols, yr, mo).reindex(columns=feat_snow)
            X_s = row_s.values
            model_names_s = [k for k in models_snow if isinstance(models_snow[k], Pipeline)]
            weights_s = models_snow.get("weights", {})
            pred_s = sum(models_snow[n].predict(X_s)[0] * weights_s.get(n, 0) for n in model_names_s) if weights_s else np.mean([models_snow[n].predict(X_s)[0] for n in model_names_s])
        rows.append({
            "year": yr, "month": mo, "month_name": MONTH_NAMES[mo],
            "actual_wteq": round(actual_w, 2),
            "actual_snow": round(actual_s, 2) if np.isfinite(actual_s) else np.nan,
            "pred_wteq": round(pred_w, 2),
            "pred_snow": round(pred_s, 2) if np.isfinite(pred_s) else np.nan,
            "error_wteq": round(pred_w - actual_w, 2),
            "error_snow": round(pred_s - actual_s, 2) if np.isfinite(pred_s) and np.isfinite(actual_s) else np.nan,
        })
    return pd.DataFrame(rows)


def tune_ensemble_weights_from_recent(
    df: pd.DataFrame,
    models_wteq: dict,
    models_snow: dict = None,
    n_snow_months: int = 6,
) -> tuple[dict, dict]:
    """Recompute ensemble weights from recent WTEQ (and optionally snow) RMSE."""
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_wteq = models_wteq.get("features", [])
    feat_snow = (models_snow or {}).get("features", [])
    winter = df[df["month"].isin(WINTER_MONTHS)].copy()
    winter = winter[winter["WTEQ"].notna()]
    if not feat_snow and "snow_inches" in winter.columns:
        winter = winter  # no snow model to tune
    elif feat_snow:
        winter = winter[winter["snow_inches"].notna()] if "snow_inches" in winter.columns else winter
    winter = winter.sort_values(["year", "month"], ascending=[False, False]).head(n_snow_months)
    if len(winter) < 2:
        return models_wteq, models_snow or {}

    names_w = [k for k in models_wteq if isinstance(models_wteq[k], Pipeline)]
    preds_w = {n: [] for n in names_w}
    actual_w = []
    for _, r in winter.iterrows():
        yr, mo = int(r["year"]), int(r["month"])
        row_w = build_current_row(df, tele_cols, yr, mo).reindex(columns=feat_wteq)
        for n in names_w:
            preds_w[n].append(models_wteq[n].predict(row_w.values)[0])
        actual_w.append(float(r["WTEQ"]))
    actual_w = np.array(actual_w)
    inv_rmse_w = {n: 1.0 / (np.sqrt(np.mean((np.array(preds_w[n]) - actual_w) ** 2)) + 1e-6) for n in names_w}
    total_w = sum(inv_rmse_w.values())
    new_weights_w = {n: inv_rmse_w[n] / total_w for n in names_w}
    models_wteq = dict(models_wteq)
    models_wteq["weights"] = new_weights_w

    if models_snow and feat_snow:
        names_s = [k for k in models_snow if isinstance(models_snow[k], Pipeline)]
        preds_s = {n: [] for n in names_s}
        actual_s = []
        for _, r in winter.iterrows():
            yr, mo = int(r["year"]), int(r["month"])
            row_s = build_current_row(df, tele_cols, yr, mo).reindex(columns=feat_snow)
            for n in names_s:
                preds_s[n].append(models_snow[n].predict(row_s.values)[0])
            actual_s.append(float(r["snow_inches"]))
        if len(actual_s) >= 2:
            actual_s = np.array(actual_s)
            inv_rmse_s = {n: 1.0 / (np.sqrt(np.mean((np.array(preds_s[n]) - actual_s) ** 2)) + 1e-6) for n in names_s}
            total_s = sum(inv_rmse_s.values())
            new_weights_s = {n: inv_rmse_s[n] / total_s for n in names_s}
            models_snow = dict(models_snow)
            models_snow["weights"] = new_weights_s
    return models_wteq, models_snow if models_snow is not None else {}


# ── 5. Analog year analysis ────────────────────────────────────────────────────

def find_analogs(df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    """
    Find the N historical years whose Oct-Jan teleconnection pattern
    is most similar to 2025-26. Uses Euclidean distance in normalized space.
    """
    print("\n[7] Finding analog years ...")
    # Reference period: Oct-Jan for the forecast year (2025-26 season)
    ref_months = [(2025, 10), (2025, 11), (2025, 12), (2026, 1)]
    compare_cols = ["ao", "roni", "pdo", "pna"]  # RONI for ENSO (most data-complete)
    compare_cols = [c for c in compare_cols if c in df.columns]

    # Build the reference vector (mean of Oct-Jan indices)
    ref_rows = df[df.apply(lambda r: (r["year"], r["month"]) in ref_months, axis=1)]
    ref_vec  = ref_rows[compare_cols].mean()

    # For each historical year, compute mean of Oct-Jan
    analog_scores = []
    for yr in sorted(df["year"].unique()):
        if yr >= 2025:
            continue
        hist_months = [(yr-1, 10), (yr-1, 11), (yr-1, 12), (yr, 1)]
        hist_rows = df[df.apply(lambda r: (r["year"], r["month"]) in hist_months, axis=1)]
        if len(hist_rows) < 2:
            continue
        hist_vec = hist_rows[compare_cols].mean()
        if hist_vec.isna().sum() > 1:
            continue
        dist = np.sqrt(((ref_vec - hist_vec) ** 2).sum())
        analog_scores.append({"year": yr, "distance": dist, **hist_vec.to_dict()})

    scores_df = pd.DataFrame(analog_scores).sort_values("distance").head(n)

    # Pull their winter SWE and (if present) snowfall
    snow_data = []
    for _, row in scores_df.iterrows():
        yr = int(row["year"])
        for m in [11, 12, 1, 2, 3, 4]:
            sub = df[(df["year"] == yr) & (df["month"] == m)]
            if len(sub) > 0:
                rec = {
                    "analog_year": yr,
                    "month": m,
                    "month_name": MONTH_NAMES[m],
                    "WTEQ": sub["WTEQ"].values[0],
                    "distance": row["distance"],
                }
                if "snow_inches" in df.columns:
                    rec["snow_inches"] = sub["snow_inches"].values[0]
                snow_data.append(rec)
    analogs_detail = pd.DataFrame(snow_data)
    print(f"   Top {n} analog years: {list(scores_df['year'].astype(int))}")
    return scores_df, analogs_detail


# ── 6. Feature importance ──────────────────────────────────────────────────────

def _get_step(pipe, step_key="m"):
    """Extract the final estimator from a Pipeline regardless of step name."""
    if hasattr(pipe, "named_steps"):
        if step_key in pipe.named_steps:
            return pipe.named_steps[step_key]
        # Fallback: return last step
        return list(pipe.named_steps.values())[-1]
    return pipe


def get_feature_importances(models: dict, target: str) -> pd.DataFrame:
    feats = models["features"]
    imp_series = []

    # Tree-based importances
    for key in ("RF", "ExtraTrees", "GBR", "XGBoost"):
        pipe = models.get(key)
        if pipe is None:
            continue
        est = _get_step(pipe)
        if hasattr(est, "feature_importances_"):
            s = pd.Series(est.feature_importances_, index=feats)
            s /= s.sum() if s.sum() > 0 else 1
            imp_series.append(s)

    # Linear |coef|
    for key in ("Ridge", "BayesRidge", "ElasticNet"):
        pipe = models.get(key)
        if pipe is None:
            continue
        est = _get_step(pipe)
        if hasattr(est, "coef_"):
            s = pd.Series(np.abs(est.coef_), index=feats)
            s /= s.sum() if s.sum() > 0 else 1
            imp_series.append(s)

    if not imp_series:
        return pd.DataFrame()

    combined = pd.concat(imp_series, axis=1).mean(axis=1)

    # Also keep RF, GBR, Ridge separately for backward compat with plots
    def _safe(key, attr):
        pipe = models.get(key)
        if pipe is None:
            return pd.Series(np.zeros(len(feats)), index=feats)
        est = _get_step(pipe)
        arr = getattr(est, attr, np.zeros(len(feats)))
        s = pd.Series(arr if len(arr) == len(feats) else np.zeros(len(feats)), index=feats)
        s /= s.sum() if s.sum() > 0 else 1
        return s

    rf_imp    = _safe("RF",    "feature_importances_")
    gbr_imp   = _safe("GBR",   "feature_importances_")
    ridge_imp = _safe("Ridge", "coef_")

    df_imp = pd.DataFrame({
        "feature":   feats,
        "rf":        rf_imp.values,
        "gbr":       gbr_imp.values,
        "ridge_abs": ridge_imp.values,
        "combined":  combined.values,
    }).sort_values("combined", ascending=False)
    return df_imp


# ── 7. Plotting ────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap: correlation between each teleconnection × month and WTEQ."""
    print("\n[Plot] Correlation heatmap ...")
    tele_short = {
        "ao":             "AO",
        "roni":           "RONI",
        "enso34":         "ENSO34",
        "pdo":            "PDO",
        "pna":            "PNA",
        "qbo":            "QBO",
        "np":             "NP",
        "pmm":            "PMM",
        "nao":            "NAO",
        "solar":          "Solar",
        "wp":             "WP",
        "index4_140e_mjo": "MJO-4",
        "index5_160e_mjo": "MJO-5",
        "index6_120w_mjo": "MJO-6",
        "index7_40w_mjo":  "MJO-7",
    }
    available = {k: v for k, v in tele_short.items() if k in df.columns}

    rows = []
    for m in WINTER_MONTHS:
        sub = df[df["month"] == m].copy()
        for raw_col, label in available.items():
            if "WTEQ" in df.columns:
                paired = sub[[raw_col, "WTEQ"]].dropna()
                r = paired.corr().iloc[0, 1] if len(paired) > 10 else np.nan
                rows.append({"telecon": label, "month": MONTH_NAMES[m], "corr_wteq": r})

    corr_df = pd.DataFrame(rows)
    pivot = corr_df.pivot(index="telecon", columns="month", values="corr_wteq")
    # Reorder columns
    month_order = [MONTH_NAMES[m] for m in WINTER_MONTHS]
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-0.6, vmax=0.6, linewidths=0.4, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Teleconnection ↔ Snoqualmie Pass SWE\n(same-month correlation by calendar month)", fontsize=13)
    ax.set_xlabel("Calendar Month"); ax.set_ylabel("Teleconnection Index")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"   Saved plots/correlation_heatmap.png")


def plot_feature_importance(imp_wteq: pd.DataFrame, imp_snow: pd.DataFrame = None):
    print("[Plot] Feature importance ...")
    if imp_snow is None or (isinstance(imp_snow, pd.DataFrame) and imp_snow.empty):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for _ax, imp, title in [(ax, imp_wteq, "WTEQ (SWE)")]:
            top = imp.head(20)
            if top.empty:
                _ax.text(0.5, 0.5, "No data", transform=_ax.transAxes, ha="center")
            else:
                colors = ["steelblue" if "lag0" in f else "cornflowerblue" if "lag1" in f
                          else "lightsteelblue" for f in top["feature"]]
                _ax.barh(top["feature"][::-1], top["combined"][::-1], color=colors[::-1])
            _ax.set_title(f"Feature Importance: {title}", fontsize=12)
            _ax.set_xlabel("Normalized Combined Importance")
            _ax.axvline(0, color="k", lw=0.5)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, imp, title in zip(axes, [imp_wteq, imp_snow], ["WTEQ (SWE)", "Snowfall"]):
            top = imp.head(20)
            if top.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            else:
                colors = ["steelblue" if "lag0" in f else "cornflowerblue" if "lag1" in f
                          else "lightsteelblue" for f in top["feature"]]
                ax.barh(top["feature"][::-1], top["combined"][::-1], color=colors[::-1])
            ax.set_title(f"Feature Importance: {title}", fontsize=12)
            ax.set_xlabel("Normalized Combined Importance")
            ax.axvline(0, color="k", lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"   Saved plots/feature_importance.png")


def plot_analog_years(analogs_detail: pd.DataFrame, df: pd.DataFrame, fc_df: pd.DataFrame):
    print("[Plot] Analog years ...")
    analog_years = analogs_detail["analog_year"].unique()
    month_order  = [11, 12, 1, 2, 3, 4]
    month_labels = [MONTH_NAMES[m] for m in month_order]
    # Two panels when we have snowfall forecast + snow data
    has_snow = (
        "snow_ensemble" in fc_df.columns
        and "snow_inches" in analogs_detail.columns
        and df["snow_inches"].notna().any()
    )
    panels = [
        ("WTEQ", "SWE (inches)", "wteq_ensemble"),
    ]
    if has_snow:
        panels.append(("snow_inches", "Snowfall (inches)", "snow_ensemble"))
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]
    for ax, (col, unit, forecast_col) in zip(axes, panels):
        hist_mean = []
        hist_lo = []
        hist_hi = []
        for m in month_order:
            vals = df[(df["month"] == m) & df[col].notna()][col]
            hist_mean.append(vals.mean() if len(vals) else np.nan)
            hist_lo.append(vals.quantile(0.25) if len(vals) else np.nan)
            hist_hi.append(vals.quantile(0.75) if len(vals) else np.nan)
        ax.fill_between(month_labels, hist_lo, hist_hi,
                        alpha=0.2, color="grey", label="Historical IQR (all years)")
        ax.plot(month_labels, hist_mean, "k--", lw=1.5, label="Historical mean")
        cmap = plt.cm.tab10
        for i, yr in enumerate(analog_years):
            sub = analogs_detail[analogs_detail["analog_year"] == yr]
            ys = [sub[sub["month"] == m][col].values[0] if len(sub[sub["month"] == m]) > 0 else np.nan for m in month_order]
            ax.plot(month_labels, ys, "o-", color=cmap(i), alpha=0.75, lw=1.5, label=f"{yr}")
        fc_vals = []
        for m in month_order:
            fc_row = fc_df[fc_df["month"] == m]
            if len(fc_row) > 0 and forecast_col in fc_df.columns:
                fc_vals.append(fc_row[forecast_col].values[0])
            else:
                obs = df[(df["year"].isin([2025, 2026])) & (df["month"] == m)]
                fc_vals.append(obs[col].values[0] if len(obs) > 0 and col in obs and not obs[col].isna().all() else np.nan)
        obs_vals = []
        for m in month_order:
            obs = df[((df["year"] == 2025) & (df["month"] == m)) | ((df["year"] == 2026) & (df["month"] == m))]
            obs_vals.append(obs[col].values[0] if len(obs) > 0 and col in obs and not obs[col].isna().all() else np.nan)
        ax.plot(month_labels, obs_vals, "ko-", lw=2.5, ms=7, label="2025-26 Observed")
        ax.plot(month_labels, fc_vals, "r^--", lw=2.5, ms=8, label="2025-26 Forecast")
        ax.set_title(f"Analog Years vs 2025-26: {col}", fontsize=12)
        ax.set_ylabel(unit)
        ax.set_xlabel("Month")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Analog Year Comparison — Snoqualmie Pass Winter 2025-26", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "analog_years.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/analog_years.png")


def plot_forecast_summary(df: pd.DataFrame, fc_df: pd.DataFrame):
    """Time series of historical WTEQ with 2026 forecast highlighted."""
    print("[Plot] Forecast summary time series ...")
    has_snow = "snow_ensemble" in fc_df.columns and "snow_hist_mean" in fc_df.columns
    n_rows = 3 if has_snow else 2
    fig = plt.figure(figsize=(15, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Panel 1: Jan-Mar historical SWE timeseries ---
    ax1 = fig.add_subplot(gs[0, :])
    for m, color in [(1,"steelblue"), (2,"royalblue"), (3,"navy")]:
        sub = df[df["month"] == m][["year","WTEQ"]].dropna().sort_values("year")
        ax1.plot(sub["year"], sub["WTEQ"], "o-", ms=3, lw=1, color=color, alpha=0.6,
                 label=f"{MONTH_NAMES[m]} SWE")
    # Forecast dots
    for _, row in fc_df.iterrows():
        m = int(row["month"])
        if m in [1, 2, 3]:
            color = {"1":"steelblue","2":"royalblue","3":"navy"}[str(m)]
            ax1.axvline(row["year"], color="red", alpha=0.2, lw=1)
            ax1.scatter(row["year"], row["wteq_ensemble"], color="red", s=120,
                        zorder=5, marker="*", label=f"{MONTH_NAMES[m]} 2026 forecast" if m == 2 else "")
    ax1.set_title("Snoqualmie Pass Monthly SWE — Historical + 2026 Forecast", fontsize=12)
    ax1.set_xlabel("Year"); ax1.set_ylabel("SWE (inches)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # --- Panel 2: Forecast bar chart (SWE) ---
    ax2 = fig.add_subplot(gs[1, :])
    months_fc = fc_df["month_name"].tolist()
    ens   = fc_df["wteq_ensemble"].tolist()
    hmean = fc_df["wteq_hist_mean"].tolist()
    hstd  = fc_df["wteq_hist_std"].tolist()
    x = np.arange(len(months_fc))
    ax2.bar(x - 0.2, hmean, width=0.35, color="lightsteelblue", label="Historical mean")
    ax2.errorbar(x - 0.2, hmean, yerr=hstd, fmt="none", color="steelblue", capsize=4)
    ax2.bar(x + 0.2, ens, width=0.35, color="tomato", label="Forecast")
    ax2.set_xticks(x); ax2.set_xticklabels(months_fc)
    ax2.set_title("Forecast vs Historical Mean: SWE", fontsize=11)
    ax2.set_ylabel("SWE (inches)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    # --- Panel 3 (optional): Snowfall bar chart ---
    if has_snow:
        ax3 = fig.add_subplot(gs[2, :])
        snow_ens   = fc_df["snow_ensemble"].tolist()
        snow_hist  = fc_df["snow_hist_mean"].tolist()
        snow_std   = fc_df["snow_hist_std"].tolist() if "snow_hist_std" in fc_df.columns else [0] * len(months_fc)
        ax3.bar(x - 0.2, snow_hist, width=0.35, color="lightsteelblue", label="Historical mean")
        ax3.errorbar(x - 0.2, snow_hist, yerr=snow_std, fmt="none", color="steelblue", capsize=4)
        ax3.bar(x + 0.2, snow_ens, width=0.35, color="tomato", label="Forecast (model + clim blend)")
        ax3.set_xticks(x); ax3.set_xticklabels(months_fc)
        ax3.set_title("Forecast vs Historical Mean: Snowfall (inches)", fontsize=11)
        ax3.set_ylabel("Snowfall (inches)"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Snoqualmie Pass Snowpack Forecast — Winter 2025-26 (SWE)"
                 + (" + Snowfall" if has_snow else "")
                 + f"\nCurrent conditions (Jan 2026): AO={-2.05:.2f}, ONI={-0.55:.2f}, "
                 f"PDO={-0.36:.2f}, PNA={+0.79:.2f}, NAO={-0.36:.2f}",
                 fontsize=11)
    plt.savefig(os.path.join(PLOTS, "forecast_2025_2026.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/forecast_2025_2026.png")


def plot_telecon_current_state(df: pd.DataFrame):
    """Bar chart of current teleconnection indices vs historical range."""
    print("[Plot] Current teleconnection state ...")
    current = {
        "AO":      -2.05,   # Jan 2026 (ao.csv)
        "ONI":     -0.55,   # NDJ 2025 season (oni.csv)
        "PDO":     -0.36,   # Jan 2026 (pdo.csv) — note: was -3.51 in Aug 2025 (summer outlier)
        "PNA":      0.79,   # Jan 2026 (pna.csv)
        "NAO":     -0.36,   # Jan 2026 (nao.csv)
    }
    # Rough mapping to df columns for historical distribution
    col_map = {"AO":"ao","ONI":"enso34","PDO":"pdo","PNA":"pna","NAO":"nao"}

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = []
    means   = []
    stds    = []
    vals    = list(current.values())
    labels  = list(current.keys())

    for label, val in current.items():
        col = col_map.get(label)
        if col and col in df.columns:
            hist = df[df["month"].isin([12,1,2])][col].dropna()
            means.append(hist.mean())
            stds.append(hist.std())
        else:
            means.append(0); stds.append(1)
        # Color based on sign convention vs snowpack
        sign_good = {"AO": -1, "ONI": -1, "PDO": -1, "PNA": -1, "NAO": -1}
        good = (sign_good.get(label, -1) * val) > 0.3
        neutral = abs(val) < 0.3
        colors.append("steelblue" if good else ("gold" if neutral else "tomato"))

    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, alpha=0.85, width=0.5, label="Current value")
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=6, lw=2,
                label="DJF historical mean ± 1σ")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Index Value")
    ax.set_title("Current Teleconnection State vs Historical DJF Distribution\n"
                 "(Blue=favorable for snow, Red=unfavorable, Gold=neutral)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "current_telecon_state.png"), dpi=150)
    plt.close()
    print(f"   Saved plots/current_telecon_state.png")


# ── 8. Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Snoqualmie Pass snowpack forecast")
    ap.add_argument("--tune-recent", action="store_true", help="Refit ensemble weights from last 6 snow months RMSE")
    ap.add_argument("--backtest", action="store_true", help="Run leave-one-year-out backtest and exit (no forecast)")
    ap.add_argument("--no-tune-results", action="store_true", help="With --backtest: ignore tune_backtest_results.csv and use built-in configs")
    args = ap.parse_args()

    print("\n" + "="*60)
    print("  Snoqualmie Pass Snowpack Forecasting Tool")
    print("  Target season: Winter 2025-2026")
    print("="*60 + "\n")

    # --- Load and build dataset ---
    df = load_base()
    df = patch_fresh_telecons(df)
    df = patch_fresh_snotel(df)
    df = patch_historical_snowfall(df)  # Restore 1950-2022 snowfall from transformed_snow.csv
    df = apply_sno_pass_first(df)   # Prefer Stampede/DOT-ALP; 20-yr SNOTEL correction
    df = patch_ndbc_buoy(df)        # NE Pacific buoy: wave height, SLP, wind (2007+)
    df = patch_synoptic_monthly(df) # Synoptic SLP gradient (2003+)
    df = patch_slp_nepac(df)        # Aleutian Low SLP anomaly (1948+)
    df = patch_hgt500_gradient(df)  # 500mb height gradient offshore vs Cascade (1948+)
    df = patch_nino12_tni(df)       # Nino1+2 (EP SST) + TNI (ENSO flavour) (1950+)
    df = build_features(df)

    if getattr(args, "backtest", False):
        print("[Backtest] Leave-one-year-out ...\n")
        use_tune = not getattr(args, "no_tune_results", False)
        best = load_best_tune_config() if use_tune else None
        if best is not None:
            label, model_names, tele_subset, clim_blend, pipeline_overrides = best
            print(f"  Using best tune config: {label[:70]}\n")
            configs = [(label, model_names, tele_subset, clim_blend, pipeline_overrides)]
        else:
            if use_tune:
                print("  (No data/tune_backtest_results.csv — using built-in configs.)\n")
            CORE_ONLY = ["ao", "roni", "pdo", "pna"]
            configs = [
                ("Full ensemble, all features", None, None, None, None),
                ("Ridge only, all features", ["Ridge"], None, None, None),
                ("Ridge only, core telecons (ao,roni,pdo,pna)", ["Ridge"], CORE_ONLY, None, None),
                ("Ridge + core + 50% climatology blend", ["Ridge"], CORE_ONLY, 0.5, None),
            ]
        all_bt = []
        for cfg in configs:
            if len(cfg) == 5:
                label, model_names, tele_subset, clim_blend, pipeline_overrides = cfg
            else:
                label, model_names, tele_subset, clim_blend = cfg[:4]
                pipeline_overrides = None
            bt_wteq = run_backtest(df, "WTEQ", model_names=model_names, tele_subset=tele_subset,
                                   clim_blend=clim_blend, pipeline_overrides=pipeline_overrides, verbose=False)
            all_bt.append((label, bt_wteq))
            print(f"  {label}")
            print(f"    WTEQ: RMSE={bt_wteq['rmse']:.2f}  RMSE_clim={bt_wteq['rmse_clim']:.2f}  skill={bt_wteq['skill']:.1%}  corr={bt_wteq['correlation']:.3f}")
            print()
        print("Summary: skill = 1 - RMSE/RMSE_clim (positive = better than climatology)")
        best_wteq = max(all_bt, key=lambda x: x[1].get("skill") or -999)
        print(f"  Best WTEQ config: {best_wteq[0]} (skill={best_wteq[1]['skill']:.1%})")
        # Snowfall backtest using same config as best WTEQ (or first config if multiple)
        best_label = best_wteq[0]
        best_cfg = next((c for c in configs if (c[0] if len(c)==5 else c[0]) == best_label), configs[0] if configs else None)
        bt_snow = None
        if best_cfg is not None and "snow_inches" in df.columns and df["snow_inches"].notna().any():
            print("\n  Snowfall (snow_inches) — same config as best WTEQ:")
            _, mn, ts, cb, po = best_cfg if len(best_cfg) == 5 else (*best_cfg[:4], None)
            bt_snow = run_backtest(df, "snow_inches", model_names=mn, tele_subset=ts, clim_blend=cb,
                                  pipeline_overrides=po, verbose=False)
            print(f"    RMSE={bt_snow['rmse']:.2f}  RMSE_clim={bt_snow['rmse_clim']:.2f}  skill={bt_snow['skill']:.1%}  corr={bt_snow['correlation']:.3f}")
        bt_wteq = all_bt[0][1]
        for target, bt in [("WTEQ", bt_wteq)]:
            res = bt.get("results", [])
            if res:
                recs = [{"year": r["year"], "month": r["month"], "actual": r["actual"], "pred": r["pred"], "clim_pred": r["clim_pred"]} for r in res]
                pd.DataFrame(recs).to_csv(os.path.join(DATA, f"backtest_{target}.csv"), index=False)
        if bt_snow is not None and bt_snow.get("results"):
            recs = [{"year": r["year"], "month": r["month"], "actual": r["actual"], "pred": r["pred"], "clim_pred": r["clim_pred"]} for r in bt_snow["results"]]
            pd.DataFrame(recs).to_csv(os.path.join(DATA, "backtest_snow_inches.csv"), index=False)
        print("  Saved data/backtest_WTEQ.csv" + (" , data/backtest_snow_inches.csv" if bt_snow is not None else ""))
        return

    # --- Load best tune config (updates WTEQ_CLIM_BLEND & WTEQ_MODEL_SUBSET) ---
    _apply_tune_config()

    # --- Train models (SWE primary; snowfall secondary with climatology blend) ---
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    models_wteq = train_models(df, "WTEQ", months=WINTER_MONTHS)
    models_snow = train_models(df, "snow_inches", months=WINTER_MONTHS)

    if getattr(args, "tune_recent", False):
        try:
            models_wteq, models_snow = tune_ensemble_weights_from_recent(df, models_wteq, models_snow, n_snow_months=6)
            print("\n   Ensemble weights tuned from last 6 months WTEQ + snowfall RMSE.")
        except Exception as e:
            print(f"\n   Tune-recent skipped: {e}")
    print("\n[Saving models for dashboard ...]")
    model_dir = os.path.join(BASE, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(models_wteq, os.path.join(model_dir, "models_wteq.pkl"))
    joblib.dump(models_snow, os.path.join(model_dir, "models_snow.pkl"))
    df.to_parquet(os.path.join(model_dir, "forecast_df.parquet"), index=False)
    print(f"   Saved models/ directory (dashboard will load these)")

    # --- Feature importances ---
    imp_wteq = get_feature_importances(models_wteq, "WTEQ")
    imp_snow = get_feature_importances(models_snow, "snow_inches")
    imp_wteq.to_csv(os.path.join(DATA, "feature_importance_wteq.csv"), index=False)
    imp_snow.to_csv(os.path.join(DATA, "feature_importance_snow.csv"), index=False)
    print(f"\n   Top 10 features for WTEQ:")
    print(imp_wteq.head(10)[["feature", "combined"]].to_string(index=False))

    # --- Forecast (SWE + snowfall with climatology blend) ---
    fc_df = forecast_season(df, models_wteq, models_snow, target_year=2026)
    fc_df.to_csv(os.path.join(DATA, "forecast_results.csv"), index=False, encoding="utf-8")
    print("\n" + "="*60)
    print("  FORECAST: Snoqualmie Pass — 2026")
    print("="*60)
    for _, r in fc_df.iterrows():
        print(f"\n  {r['month_name']} 2026:")
        wteq_blend_str = f" (blend {1-WTEQ_CLIM_BLEND:.0%} model / {WTEQ_CLIM_BLEND:.0%} clim)" if WTEQ_CLIM_BLEND > 0 else ""
        print(f"    SWE:       {r.get('wteq_ensemble','?'):.1f}\"{wteq_blend_str} "
              f"(hist mean {r.get('wteq_hist_mean','?'):.1f}\" +/- {r.get('wteq_hist_std','?'):.1f}\")  "
              f"-> {r.get('wteq_pct','?'):.0f}th percentile")
        if "snow_ensemble" in r:
            print(f"    Snowfall:  {r.get('snow_ensemble','?'):.1f}\" (blend {1-SNOW_CLIM_BLEND:.0%} model / {SNOW_CLIM_BLEND:.0%} clim)  "
              f"hist mean {r.get('snow_hist_mean','?'):.1f}\" -> {r.get('snow_pct','?'):.0f}th pct")

    # Sounding context
    snd_ctx = getattr(fc_df, "attrs", {}).get("sounding", {})
    if snd_ctx:
        print(f"\n  Sounding context:")
        print(f"    Freezing level: {snd_ctx.get('freezing_level_current_ft','?')}' ASL "
              f"(48h: {snd_ctx.get('freezing_level_48h_min_ft','?')}'-{snd_ctx.get('freezing_level_48h_max_ft','?')}')")
        print(f"    Snow level (wet-bulb): {snd_ctx.get('snow_level_ft','?')}' ASL")
        print(f"    Snowfall hours: {snd_ctx.get('snowfall_hours_48h','?')}/48h, "
              f"{snd_ctx.get('snowfall_hours_120h','?')}/120h")
        print(f"    850 hPa wind: {snd_ctx.get('wind_850_dir','?')} deg @ {snd_ctx.get('wind_850_mph','?')} mph")

    # --- Analog years ---
    analog_scores, analogs_detail = find_analogs(df, n=7)
    analog_scores.to_csv(os.path.join(DATA, "analog_years.csv"), index=False)
    analogs_detail.to_csv(os.path.join(DATA, "analog_detail.csv"), index=False)

    # --- Natural-language bottom line (for dashboard + human distillation) ---
    try:
        from bottom_line import build_context, generate_bottom_line, save_bottom_line
        nowcast_data = None
        nc_path = os.path.join(DATA, "nowcast.json")
        if os.path.isfile(nc_path):
            try:
                with open(nc_path, "r", encoding="utf-8") as f:
                    nowcast_data = json.load(f)
            except Exception:
                pass
        ctx = build_context(fc_df, analog_scores, df, target_year=2026,
                            nowcast_data=nowcast_data)
        bl_text = generate_bottom_line(ctx)
        bl_path = os.path.join(DATA, "bottom_line.json")
        human_notes = ""
        if os.path.isfile(bl_path):
            try:
                with open(bl_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    human_notes = existing.get("human_notes") or ""
            except Exception:
                pass
        out_path = save_bottom_line(bl_text, ctx, out_path=bl_path, human_notes=human_notes)
        print(f"\n   Bottom line saved to {out_path}")
    except Exception as e:
        print(f"\n   Bottom line skipped: {e}")

    # --- Forecast vs actual (last 6 snow months) for dashboard and tuning ---
    try:
        fva = build_forecast_vs_actual_recent(df, models_wteq, models_snow, n_snow_months=6)
        if not fva.empty:
            fva.to_csv(os.path.join(DATA, "forecast_vs_actual_recent.csv"), index=False)
            print(f"\n   Forecast vs actual (last 6 snow months): {os.path.join(DATA, 'forecast_vs_actual_recent.csv')}")
    except Exception as e:
        print(f"\n   Forecast vs actual skipped: {e}")

    # --- Plots ---
    print("\n[Generating plots ...]")
    plot_correlation_heatmap(df)
    plot_feature_importance(imp_wteq, imp_snow)
    plot_analog_years(analogs_detail, df, fc_df)
    plot_forecast_summary(df, fc_df)
    plot_telecon_current_state(df)

    print("\n" + "="*60)
    print("  Done. Outputs:")
    print(f"  data/forecast_results.csv")
    print(f"  data/analog_years.csv")
    print(f"  data/bottom_line.json")
    print(f"  data/forecast_vs_actual_recent.csv")
    print(f"  data/feature_importance_wteq.csv")
    print(f"  plots/correlation_heatmap.png")
    print(f"  plots/feature_importance.png")
    print(f"  plots/analog_years.png")
    print(f"  plots/forecast_2025_2026.png")
    print(f"  plots/current_telecon_state.png")
    print("="*60)


if __name__ == "__main__":
    main()
