# Storms & conditions data loader for dashboard.
# Load freezing level, wind, and related series from pipeline or synoptic when available.

from pathlib import Path
from typing import Any

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def load_storms_conditions() -> dict[str, Any]:
    out = {"pipeline_forecast": None, "synoptic_daily": None}
    p1 = DATA / "pipeline" / "model_forecast_daily.csv"
    if p1.exists():
        try:
            df = pd.read_csv(p1)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            out["pipeline_forecast"] = df
        except Exception:
            pass
    for p2 in [DATA / "synoptic_daily_features.csv", DATA / "processed" / "synoptic_daily_features.csv"]:
        if p2.exists():
            try:
                df = pd.read_csv(p2)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                out["synoptic_daily"] = df
                break
            except Exception:
                pass
    return out
