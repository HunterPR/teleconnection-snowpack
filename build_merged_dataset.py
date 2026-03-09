"""
build_merged_dataset.py
========================
Build Merged_Dataset.csv from data/ teleconnection and snowpack files so forecast.py
can run without a pre-existing Merged_Dataset.

Run order:
  1. python fetch_data.py          # teleconnections + snoqualmie_snotel
  2. python fetch_new_predictors.py  # optional: epo, nino4, z500, amo, stampede, etc.
  3. python organize_data.py       # optional: if using custom_sources (DOT/ALP) for pass snowfall
  4. python build_merged_dataset.py
  5. python forecast.py

Output: Merged_Dataset.csv in project root (used by forecast.py).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT_PATH = BASE / "Merged_Dataset.csv"
WINTER_MONTHS = [10, 11, 12, 1, 2, 3, 4]
START_YEAR = 1950


def _read_csv(path: Path, required_cols: list) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if not all(c in df.columns for c in required_cols):
            return None
        return df
    except Exception:
        return None


def _season_to_month(season: str) -> int:
    m = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
         "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    return m.get((season or "").strip().upper(), np.nan)


def build() -> pd.DataFrame:
    """Build merged monthly dataset from data/ CSVs."""
    # Full year-month grid
    end_year = pd.Timestamp.now().year + 1
    rows = []
    for y in range(START_YEAR, end_year + 1):
        for m in range(1, 13):
            rows.append({"year": y, "month": m})
    df = pd.DataFrame(rows)

    # ONI -> enso34; RONI -> roni (primary ENSO predictor for models)
    oni = _read_csv(DATA / "oni.csv", ["year", "season", "oni_anomaly"])
    if oni is not None:
        oni = oni.copy()
        oni["month"] = oni["season"].map(_season_to_month)
        oni = oni.dropna(subset=["month"])
        oni["month"] = oni["month"].astype(int)
        oni = oni.rename(columns={"oni_anomaly": "enso34"})[["year", "month", "enso34"]]
        df = df.merge(oni, on=["year", "month"], how="left")
    roni_tbl = _read_csv(DATA / "roni.csv", ["year", "month", "roni"])
    if roni_tbl is not None:
        roni_tbl = roni_tbl.copy()
        roni_tbl["year"] = pd.to_numeric(roni_tbl["year"], errors="coerce").astype(int)
        roni_tbl["month"] = pd.to_numeric(roni_tbl["month"], errors="coerce").astype(int)
        df = df.merge(roni_tbl[["year", "month", "roni"]], on=["year", "month"], how="left")

    # PDO, PNA, AO, NAO
    for name, file, col in [("pdo", "pdo.csv", "pdo"), ("pna", "pna.csv", "pna"),
                            ("ao", "ao.csv", "ao"), ("nao", "nao.csv", "nao")]:
        tbl = _read_csv(DATA / file, ["year", "month", col])
        if tbl is not None:
            tbl = tbl.copy()
            tbl["year"] = pd.to_numeric(tbl["year"], errors="coerce").astype("Int64")
            tbl["month"] = pd.to_numeric(tbl["month"], errors="coerce").astype("Int64")
            tbl = tbl.dropna(subset=["year", "month"])
            tbl["year"] = tbl["year"].astype(int)
            tbl["month"] = tbl["month"].astype(int)
            df = df.merge(tbl[["year", "month", col]], on=["year", "month"], how="left")

    # EPO, nino4_anom, z500_nepac_anom, amo (from fetch_new_predictors)
    for file, col in [("epo.csv", "epo"), ("nino4_anom.csv", "nino4_anom"),
                      ("z500_nepac.csv", "z500_nepac_anom"), ("amo.csv", "amo")]:
        tbl = _read_csv(DATA / file, ["year", "month", col])
        if tbl is not None:
            tbl = tbl.copy()
            tbl["year"] = pd.to_numeric(tbl["year"], errors="coerce").astype(int)
            tbl["month"] = pd.to_numeric(tbl["month"], errors="coerce").astype(int)
            df = df.merge(tbl[["year", "month", col]], on=["year", "month"], how="left")

    # MJO -> monthly mean RMM indices (index4_140e_mjo, etc. from phase/amplitude)
    mjo = _read_csv(DATA / "mjo_rmm.csv", ["year", "month", "rmm1", "rmm2", "amplitude"])
    if mjo is not None:
        mjo = mjo.copy()
        mjo["year"] = pd.to_numeric(mjo["year"], errors="coerce").astype(int)
        mjo["month"] = pd.to_numeric(mjo["month"], errors="coerce").astype(int)
        mjo = mjo.groupby(["year", "month"], as_index=False).agg({
            "rmm1": "mean", "rmm2": "mean", "amplitude": "mean"
        })
        # Approximate MJO indices used in forecast (simplified: use rmm1/rmm2 as proxies)
        mjo["index4_140e_mjo"] = mjo["rmm1"]
        mjo["index5_160e_mjo"] = mjo["rmm1"]  # placeholder
        mjo["index6_120w_mjo"] = mjo["rmm2"]
        mjo["index7_40w_mjo"] = mjo["rmm2"]   # placeholder
        df = df.merge(mjo[["year", "month", "index4_140e_mjo", "index5_160e_mjo",
                           "index6_120w_mjo", "index7_40w_mjo"]], on=["year", "month"], how="left")

    # PSL extras (qbo, np, pmm, wp, solar) - optional
    psl = DATA / "PSL CSV Files"
    for name, fname in [("qbo", "transformed_qbo.csv"), ("np", "transformed_np.csv"),
                        ("pmm", "transformed_pmm.csv"), ("solar", "transformed_solar.csv")]:
        path = psl / fname if psl.exists() else None
        if path and path.exists():
            try:
                tbl = pd.read_csv(path)
                if "year" in tbl.columns:
                    tbl = tbl.copy()
                    year_col = "year"
                    month_cols = [c for c in tbl.columns if c != year_col and pd.api.types.is_numeric_dtype(tbl[c])]
                    if len(month_cols) >= 12:
                        long = []
                        for _, r in tbl.iterrows():
                            y = r[year_col]
                            for i, c in enumerate(month_cols[:12], start=1):
                                long.append({"year": int(y), "month": i, name: r[c]})
                        tbl = pd.DataFrame(long)
                        df = df.merge(tbl, on=["year", "month"], how="left")
            except Exception:
                pass
    if (psl / "wp.csv").exists():
        try:
            tbl = pd.read_csv(psl / "wp.csv")
            if "year" in tbl.columns and "month" in tbl.columns and "wp" in tbl.columns:
                tbl = tbl.copy()
                tbl["year"] = pd.to_numeric(tbl["year"], errors="coerce").astype(int)
                tbl["month"] = pd.to_numeric(tbl["month"], errors="coerce").astype(int)
                df = df.merge(tbl[["year", "month", "wp"]], on=["year", "month"], how="left")
        except Exception:
            pass

    # slp_nepac, hgt500_gradient, nino12, tni
    for file, col in [("slp_nepac.csv", "slp_nepac_anom"), ("hgt500_gradient.csv", "hgt500_gradient"),
                      ("nino12_anom.csv", "nino12_anom"), ("tni.csv", "tni")]:
        tbl = _read_csv(DATA / file, ["year", "month", col])
        if tbl is not None:
            tbl = tbl.copy()
            tbl["year"] = pd.to_numeric(tbl["year"], errors="coerce").astype(int)
            tbl["month"] = pd.to_numeric(tbl["month"], errors="coerce").astype(int)
            df = df.merge(tbl[["year", "month", col]], on=["year", "month"], how="left")

    # Snoqualmie SNOTEL -> WTEQ, snow_inches proxy
    snq = _read_csv(DATA / "snoqualmie_snotel.csv", ["year", "month"])
    if snq is not None:
        snq = snq.copy()
        snq["year"] = pd.to_numeric(snq["year"], errors="coerce").astype(int)
        snq["month"] = pd.to_numeric(snq["month"], errors="coerce").astype(int)
        swe_col = "swe_in" if "swe_in" in snq.columns else next(
            (c for c in snq.columns if "wteq" in c.lower() or "snow water" in str(c).lower()), None
        )
        if swe_col:
            snq = snq[["year", "month", swe_col]].rename(columns={swe_col: "WTEQ"})
            df = df.merge(snq, on=["year", "month"], how="left")
        else:
            df["WTEQ"] = np.nan
    else:
        df["WTEQ"] = np.nan
    # snow_inches: from pass monthly if available (DOT/ALP stations)
    pass_monthly = DATA / "processed" / "pass_monthly_snowfall.csv"
    if pass_monthly.exists():
        try:
            pm = pd.read_csv(pass_monthly)
            if "year" in pm.columns and "month" in pm.columns:
                snow_col = "snow_inches_pass" if "snow_inches_pass" in pm.columns else next(
                    (c for c in pm.columns if "snow" in c.lower()), None
                )
                if snow_col:
                    pm = pm[["year", "month", snow_col]].rename(columns={snow_col: "snow_inches"})
                    pm["year"] = pd.to_numeric(pm["year"], errors="coerce").astype(int)
                    pm["month"] = pd.to_numeric(pm["month"], errors="coerce").astype(int)
                    df = df.drop(columns=["snow_inches"], errors="ignore")
                    df = df.merge(pm, on=["year", "month"], how="left")
        except Exception:
            pass
    if "snow_inches" not in df.columns:
        df["snow_inches"] = np.nan

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df


def main():
    print("Building Merged_Dataset.csv from data/ ...")
    df = build()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} ({len(df)} rows, {len(df.columns)} cols)")
    print("Run: python forecast.py")


if __name__ == "__main__":
    main()
