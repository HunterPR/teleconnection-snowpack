"""
Sno-Pass–first targets: prefer DOT/ALP / Stampede Pass stations; correct SNOTEL to pass space.

Users often discredit raw SNOTEL. This module:
  - Prefers Stampede Pass SNOTEL (nearest SNOTEL to Snoqualmie Pass) for SWE when available.
  - Applies a 20-year correction from Snoqualmie SNOTEL #908 to pass-equivalent space
    (fit over overlapping period with Stampede, then apply to 908 when Stampede is missing).
  - **Weighted SWE blend**: when both Stampede and corrected 908 exist, combines them using
    inverse-variance (or per-month minimum-variance) weights so the more reliable source
    gets higher weight; when only one exists, uses that one. Result is pass-representative SWE.
  - Prefers monthly snowfall from pass stations (SNO38, ALP31, etc.) when available;
    otherwise keeps existing snow_inches (snow is deprecated as a forecast target).
Correction and blend weights stored in data/snotel_to_pass_correction.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
CORRECTION_FILE = DATA / "snotel_to_pass_correction.json"
WINTER_MONTHS = [10, 11, 12, 1, 2, 3, 4]  # Oct–Apr

# Overlap period for fitting 908 → Stampede (prefer ~20 years when available)
CORRECTION_START_YEAR = 2000
CORRECTION_END_YEAR = 2020


def _load_snotel_908() -> pd.DataFrame:
    path = DATA / "snoqualmie_snotel.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "year" not in df.columns or "month" not in df.columns:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    swe_col = "swe_in" if "swe_in" in df.columns else next(
        (c for c in df.columns if "snow water" in c.lower() or "wteq" in c.lower()), None
    )
    if swe_col is None:
        return pd.DataFrame()
    df = df.rename(columns={swe_col: "WTEQ_908"})
    df["WTEQ_908"] = pd.to_numeric(df["WTEQ_908"], errors="coerce")
    return df[["year", "month", "WTEQ_908"]].dropna(subset=["year", "month", "WTEQ_908"]).copy()


def _load_stampede() -> pd.DataFrame:
    path = DATA / "snotel_stampede.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    wteq_col = next(
        (c for c in df.columns if "WTEQ" in c.upper() or "swe" in c.lower()), None
    )
    if wteq_col is None:
        return pd.DataFrame()
    df = df.rename(columns={wteq_col: "WTEQ_stampede"})
    df["WTEQ_stampede"] = pd.to_numeric(df["WTEQ_stampede"], errors="coerce")
    return df[["year", "month", "WTEQ_stampede"]].dropna(
        subset=["year", "month", "WTEQ_stampede"]
    ).copy()


def compute_snotel_to_pass_correction(
    start_year: int = CORRECTION_START_YEAR,
    end_year: int = CORRECTION_END_YEAR,
    save: bool = True,
) -> Tuple[float, float, int]:
    """
    Fit linear correction WTEQ_pass = intercept + slope * WTEQ_908 over overlapping
    Stampede vs Snoqualmie 908 period. Returns (slope, intercept, n_obs).
    """
    df908 = _load_snotel_908()
    df_stampede = _load_stampede()
    if df908.empty or df_stampede.empty:
        # No correction: identity (pass = 908)
        if save:
            DATA.mkdir(parents=True, exist_ok=True)
            with open(CORRECTION_FILE, "w") as f:
                json.dump({
                    "slope": 1.0,
                    "intercept": 0.0,
                    "n_obs": 0,
                    "period": f"{start_year}-{end_year}",
                    "note": "No Stampede data; correction disabled (use 908 as-is).",
                }, f, indent=2)
        return 1.0, 0.0, 0

    merged = df908.merge(
        df_stampede,
        on=["year", "month"],
        how="inner",
    )
    merged = merged[
        (merged["year"] >= start_year)
        & (merged["year"] <= end_year)
        & merged["WTEQ_908"].notna()
        & merged["WTEQ_stampede"].notna()
        & (merged["month"].isin(WINTER_MONTHS))
    ]
    if len(merged) < 24:  # at least 2 winter years of months
        if save:
            DATA.mkdir(parents=True, exist_ok=True)
            with open(CORRECTION_FILE, "w") as f:
                json.dump({
                    "slope": 1.0,
                    "intercept": 0.0,
                    "n_obs": len(merged),
                    "period": f"{start_year}-{end_year}",
                    "note": "Insufficient overlap; correction disabled.",
                }, f, indent=2)
        return 1.0, 0.0, len(merged)

    x = merged["WTEQ_908"].values.astype(float)
    y = merged["WTEQ_stampede"].values.astype(float)
    # Linear regression y = intercept + slope * x
    slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
    intercept = float(np.mean(y) - slope * np.mean(x))
    n_obs = len(merged)

    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(CORRECTION_FILE, "w") as f:
            json.dump({
                "slope": round(float(slope), 6),
                "intercept": round(intercept, 6),
                "n_obs": n_obs,
                "period": f"{start_year}-{end_year}",
                "note": "SNOTEL 908 -> Stampede Pass (pass-equivalent) correction.",
            }, f, indent=2)

    return float(slope), intercept, n_obs


def load_correction() -> Tuple[float, float]:
    """Load slope and intercept from disk; if missing, compute and save."""
    if CORRECTION_FILE.exists():
        try:
            with open(CORRECTION_FILE) as f:
                d = json.load(f)
            return float(d["slope"]), float(d["intercept"])
        except Exception:
            pass
    slope, intercept, _ = compute_snotel_to_pass_correction(save=True)
    return slope, intercept


def compute_blend_weights(
    start_year: int = CORRECTION_START_YEAR,
    end_year: int = CORRECTION_END_YEAR,
    per_month: bool = True,
    save: bool = True,
) -> Tuple[float, Optional[dict]]:
    """
    Compute optimal weights to blend Stampede (pass) and corrected-908 SWE when both exist.
    Uses inverse-variance weighting: w_stampede = var_908 / (var_908 + var_stampede) so the
    lower-variance source gets more weight. Optionally computes per-month weights for
    seasonal representativeness (e.g. peak winter may trust Stampede more).
    Returns (global_w_stampede, per_month_dict or None). per_month_dict: month -> w_stampede.
    """
    df908 = _load_snotel_908()
    df_stampede = _load_stampede()
    if df908.empty or df_stampede.empty:
        if save and CORRECTION_FILE.exists():
            try:
                with open(CORRECTION_FILE) as f:
                    d = json.load(f)
                d["blend_w_stampede"] = 1.0
                d["blend_per_month"] = None
                with open(CORRECTION_FILE, "w") as f:
                    json.dump(d, f, indent=2)
            except Exception:
                pass
        return 1.0, None

    slope, intercept = load_correction()
    merged = df908.merge(df_stampede, on=["year", "month"], how="inner")
    merged = merged[
        (merged["year"] >= start_year)
        & (merged["year"] <= end_year)
        & merged["WTEQ_908"].notna()
        & merged["WTEQ_stampede"].notna()
        & (merged["month"].isin(WINTER_MONTHS))
    ]
    if len(merged) < 12:
        return 1.0, None

    merged["WTEQ_908_corrected"] = merged["WTEQ_908"] * slope + intercept
    var_s = float(merged["WTEQ_stampede"].var())
    var_908 = float(merged["WTEQ_908_corrected"].var())
    if var_s <= 0 or var_908 <= 0:
        w_global = 0.5
    else:
        # Inverse-variance: w_stampede = (1/var_s) / (1/var_s + 1/var_908) = var_908 / (var_908 + var_s)
        w_global = var_908 / (var_908 + var_s)

    per_month_weights = None
    if per_month:
        per_month_weights = {}
        for month in WINTER_MONTHS:
            sub = merged[merged["month"] == month]
            if len(sub) < 3:
                per_month_weights[month] = w_global
                continue
            vs = sub["WTEQ_stampede"].var()
            v9 = sub["WTEQ_908_corrected"].var()
            if vs <= 0 or v9 <= 0:
                per_month_weights[month] = w_global
            else:
                per_month_weights[month] = float(v9 / (v9 + vs))

    if save:
        DATA.mkdir(parents=True, exist_ok=True)
        try:
            with open(CORRECTION_FILE) as f:
                d = json.load(f)
        except Exception:
            d = {}
        d["blend_w_stampede"] = round(w_global, 4)
        d["blend_per_month"] = {int(k): round(v, 4) for k, v in (per_month_weights or {}).items()}
        d["blend_note"] = "Inverse-variance blend: Stampede (pass) vs corrected 908; per-month weights when both series exist."
        with open(CORRECTION_FILE, "w") as f:
            json.dump(d, f, indent=2)

    return w_global, per_month_weights


def load_blend_weights() -> Tuple[float, Optional[dict]]:
    """Load blend weights from disk; if missing, compute and save."""
    if CORRECTION_FILE.exists():
        try:
            with open(CORRECTION_FILE) as f:
                d = json.load(f)
            w = float(d.get("blend_w_stampede", 0.5))
            pm = d.get("blend_per_month")
            if pm is not None:
                pm = {int(k): float(v) for k, v in pm.items()}
            return w, pm
        except Exception:
            pass
    return compute_blend_weights(save=True)


def wteq_pass_equivalent(wteq_908: float, slope: Optional[float] = None, intercept: Optional[float] = None) -> float:
    """Convert Snoqualmie SNOTEL 908 WTEQ to pass-equivalent (Stampede space)."""
    if slope is None or intercept is None:
        slope, intercept = load_correction()
    if np.isnan(wteq_908):
        return np.nan
    return float(intercept + slope * float(wteq_908))


def build_pass_first_wteq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set WTEQ to a pass-representative SWE using a weighted blend of Stampede (pass) and
    Snoqualmie 908 (corrected to pass space). When both exist: WTEQ = w*Stampede + (1-w)*corrected_908
    with w from inverse-variance (per-month when available). When only one exists, use that one.
    Keeps WTEQ_snotel908 and WTEQ_stampede in df for transparency.
    """
    df = df.copy()
    has_908 = "WTEQ_snotel908" in df.columns
    if not has_908:
        snq = _load_snotel_908()
        if not snq.empty:
            df = df.merge(snq, on=["year", "month"], how="left")

    stampede = _load_stampede()
    if not stampede.empty:
        df = df.merge(stampede, on=["year", "month"], how="left")

    slope, intercept = load_correction()
    w_stampede_global, w_per_month = load_blend_weights()

    wteq_908_col = "WTEQ_908" if "WTEQ_908" in df.columns else "WTEQ_snotel908"
    if wteq_908_col not in df.columns:
        wteq_908_col = None
    if wteq_908_col:
        df["WTEQ_908_corrected"] = df[wteq_908_col].apply(
            lambda x: wteq_pass_equivalent(x, slope, intercept)
        )

    # Build WTEQ: blend when both exist, else Stampede, else corrected 908, else raw 908
    has_stampede = "WTEQ_stampede" in df.columns
    has_corrected = "WTEQ_908_corrected" in df.columns

    if has_stampede and has_corrected:
        def blended(row):
            s = row.get("WTEQ_stampede")
            c = row.get("WTEQ_908_corrected")
            if pd.isna(s) and pd.isna(c):
                return np.nan
            if pd.isna(s):
                return c
            if pd.isna(c):
                return s
            month = int(row.get("month", 1))
            w = (w_per_month.get(month, w_stampede_global) if w_per_month else w_stampede_global)
            return float(w * s + (1.0 - w) * c)
        df["WTEQ"] = df.apply(blended, axis=1)
    elif has_stampede:
        df["WTEQ"] = df["WTEQ_stampede"].copy()
    elif has_corrected:
        df["WTEQ"] = df["WTEQ_908_corrected"].copy()
    else:
        df["WTEQ"] = np.nan

    if has_corrected:
        df["WTEQ"] = df["WTEQ"].fillna(df["WTEQ_908_corrected"])
    if "WTEQ_snotel908" in df.columns and df["WTEQ"].isna().any():
        df["WTEQ"] = df["WTEQ"].fillna(df["WTEQ_snotel908"])

    return df


def load_pass_monthly_snowfall() -> pd.DataFrame:
    """
    Load monthly pass snowfall (year, month, snow_inches_pass) from processed pipeline.
    organize_data.py writes daily pass snowfall; we aggregate to monthly sum here,
    or read from a pre-aggregated file if present.
    """
    # Prefer pre-aggregated file if it exists
    monthly_path = DATA / "processed" / "pass_monthly_snowfall.csv"
    if monthly_path.exists():
        out = pd.read_csv(monthly_path)
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
        out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
        return out

    daily_path = DATA / "processed" / "snoqualmie_daily_targets.csv"
    if not daily_path.exists():
        return pd.DataFrame()
    daily = pd.read_csv(daily_path)
    if "date" not in daily.columns or "target_snowfall_24h_in" not in daily.columns:
        return pd.DataFrame()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date", "target_snowfall_24h_in"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly = (
        daily.groupby(["year", "month"], as_index=False)["target_snowfall_24h_in"]
        .sum()
        .rename(columns={"target_snowfall_24h_in": "snow_inches_pass"})
    )
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    return monthly


def build_pass_first_snow_inches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer monthly snowfall from pass stations (DOT/ALP) when available;
    otherwise leave snow_inches unchanged (base Merged_Dataset or existing).
    """
    df = df.copy()
    pass_snow = load_pass_monthly_snowfall()
    if pass_snow.empty:
        return df
    df = df.merge(
        pass_snow[["year", "month", "snow_inches_pass"]],
        on=["year", "month"],
        how="left",
    )
    if "snow_inches_pass" in df.columns:
        # Prefer pass snowfall where we have it
        if "snow_inches" in df.columns:
            df["snow_inches"] = df["snow_inches_pass"].fillna(df["snow_inches"])
        else:
            df["snow_inches"] = df["snow_inches_pass"]
        df["snow_inches_source"] = df["snow_inches_pass"].notna().map(
            {True: "pass", False: "other"}
        )
    return df
