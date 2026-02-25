"""
forecast.py
===========
Snoqualmie Pass snowpack forecasting tool.

Pipeline:
  1. Load historical training data (Merged_Dataset.csv: 1950-2024)
  2. Patch with fresh teleconnection indices and SNOTEL data
  3. Engineer lagged features (0, 1, 2, 3 month lags)
  4. Train Ridge + Random Forest models (leave-one-year-out CV)
  5. Identify analog years (nearest-neighbor in teleconnection space)
  6. Forecast Feb-Apr 2026 SWE and snowfall
  7. Save results + generate plots

Targets:
  - WTEQ        : Snow water equivalent at Snoqualmie Pass (inches)
  - snow_inches : Monthly snowfall at Snoqualmie Pass (inches)

Key teleconnections used (from prior correlation analysis):
  ao, enso34, pdo, pna, qbo, np, pmm, wp, solar
  + MJO indices: index4_140e, index5_160e, index6_120w, index7_40w
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
PLOTS  = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
WINTER_MONTHS   = [10, 11, 12, 1, 2, 3, 4]  # Oct-Apr
CORE_TELE = ["ao", "enso34", "pdo", "pna", "qbo", "np", "pmm", "wp", "solar",
             "index4_140e_mjo", "index5_160e_mjo", "index6_120w_mjo", "index7_40w_mjo"]
# Lags to build features for (months before the target month)
LAGS = [0, 1, 2, 3]

MONTH_NAMES = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

# ── 1. Load & extend base dataset ─────────────────────────────────────────────

def load_base() -> pd.DataFrame:
    print("[1] Loading Merged_Dataset.csv ...")
    df = pd.read_csv(os.path.join(BASE, "Merged_Dataset.csv"))
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


# ── 2. Build feature matrix ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, add lagged teleconnection values.
    Lag N means the teleconnection value from N months before this row's month.
    """
    print("[4] Engineering lagged features ...")
    df = df.sort_values(["year","month"]).reset_index(drop=True)

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

    print(f"   Added {len(LAGS) * len(tele_cols)} lag features")
    return df


def make_feature_names(tele_cols, include_nao=True):
    feats = []
    for col in tele_cols:
        for lag in LAGS:
            feats.append(f"{col}_lag{lag}")
    if include_nao:
        for lag in LAGS:
            feats.append(f"nao_lag{lag}")
    return feats


# ── 3. Model training & CV ─────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, target: str, months: list = WINTER_MONTHS):
    """
    Train Ridge and Random Forest on all available winter-month data
    where target is not NaN. Returns fitted models + scaler + feature names.
    """
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_names = make_feature_names(tele_cols, include_nao="nao" in df.columns)
    feat_names = [f for f in feat_names if f in df.columns]

    # Filter to winter months + non-null target
    mask = df["month"].isin(months) & df[target].notna()
    sub = df[mask].copy()

    # Drop rows with too many missing features (keep rows that have ≥70% of features)
    sub = sub.dropna(subset=feat_names, thresh=int(0.7 * len(feat_names)))

    # Use reindex so every feat_name maps to a column (missing ones get NaN)
    # then drop any columns that are entirely NaN (no signal at all)
    X_df_raw = sub.reindex(columns=feat_names)
    all_nan_cols = X_df_raw.columns[X_df_raw.isna().all()].tolist()
    if all_nan_cols:
        X_df_raw = X_df_raw.drop(columns=all_nan_cols)
    feat_names = list(X_df_raw.columns)

    X_raw = X_df_raw.values
    y     = sub[target].values

    print(f"\n[5] Training on target='{target}': {len(X_raw)} rows, {len(feat_names)} features")

    # Pipelines handle NaN imputation within every CV fold
    imputer = SimpleImputer(strategy="mean")

    pipe_ridge = Pipeline([
        ("imp",    SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=1.0)),
    ])
    pipe_rf = Pipeline([
        ("imp",   SimpleImputer(strategy="mean")),
        ("model", RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                        min_samples_leaf=3, random_state=42)),
    ])
    pipe_gbr = Pipeline([
        ("imp",   SimpleImputer(strategy="mean")),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=42)),
    ])

    # 5-fold CV scores
    for name, pipe in [("Ridge", pipe_ridge), ("RF", pipe_rf), ("GBR", pipe_gbr)]:
        scores = cross_val_score(pipe, X_raw, y, cv=5, scoring="r2")
        rmse_s = np.sqrt(-cross_val_score(pipe, X_raw, y, cv=5,
                                          scoring="neg_mean_squared_error"))
        print(f"   {name:<6} 5-fold CV  R²={scores.mean():.3f}±{scores.std():.3f}  "
              f"RMSE={rmse_s.mean():.2f}±{rmse_s.std():.2f}")

    # Fit on all data
    pipe_ridge.fit(X_raw, y)
    pipe_rf.fit(X_raw, y)
    pipe_gbr.fit(X_raw, y)

    # In-sample R²
    y_pred_ens = (pipe_ridge.predict(X_raw) + pipe_rf.predict(X_raw) + pipe_gbr.predict(X_raw)) / 3
    r2_ens = r2_score(y, y_pred_ens)
    print(f"   Ensemble in-sample R²={r2_ens:.3f}")

    # Keep a clean imputed+scaled X for feature importance extraction
    X_imp = imputer.fit_transform(X_raw)
    X_df  = pd.DataFrame(X_imp, columns=feat_names)

    return {
        "ridge": pipe_ridge,
        "rf":    pipe_rf,
        "gbr":   pipe_gbr,
        "scaler": None,          # scaler is inside the pipeline
        "features": feat_names,
        "X":     X_df,
        "X_raw": X_raw,
        "y":     y,
        "sub":   sub,
    }


# ── 4. Forecast current season ─────────────────────────────────────────────────

def build_current_row(df: pd.DataFrame, tele_cols: list, target_year: int, target_month: int) -> pd.DataFrame:
    """
    Build a single-row feature vector for (target_year, target_month)
    using the most recently available teleconnection values.
    """
    feat_names = make_feature_names(tele_cols, include_nao="nao" in df.columns)
    feat_names = [f for f in feat_names if f in df.columns]
    # Look for the row in df first
    row = df[(df["year"] == target_year) & (df["month"] == target_month)]
    if len(row) == 0:
        # Create a blank row
        row_dict = {f: np.nan for f in feat_names}
    else:
        row_dict = row[feat_names].iloc[0].to_dict()
    return pd.DataFrame([row_dict], columns=feat_names)


def forecast_season(df, models_wteq, models_snow, target_year=2026):
    """Forecast WTEQ and snowfall for remaining winter months of target_year."""
    print(f"\n[6] Forecasting {target_year-1}/{target_year} winter season ...")
    tele_cols = [c for c in CORE_TELE if c in df.columns]
    feat_wteq = models_wteq["features"]
    feat_snow = models_snow["features"]

    # Months to forecast: those that haven't been observed yet
    # Feb 2026 is current month (we have 15.9" SWE already observed)
    # Remaining: Mar, Apr
    forecast_months = [2, 3, 4]  # Feb still running, Mar, Apr upcoming

    records = []
    for month in forecast_months:
        row_w = build_current_row(df, tele_cols, target_year, month)
        row_s = build_current_row(df, tele_cols, target_year, month)

        # Fill NaN with column means from training data
        for feat_df, feat_list, mods, target_name in [
            (row_w, feat_wteq, models_wteq, "WTEQ"),
            (row_s, feat_snow, models_snow, "snow_inches"),
        ]:
            feat_df = feat_df.reindex(columns=feat_list)
            # Imputation is handled inside the pipelines — just pass raw array
            X_arr = feat_df[feat_list].values  # may contain NaN; pipelines handle it

            p_ridge = mods["ridge"].predict(X_arr)[0]
            p_rf    = mods["rf"].predict(X_arr)[0]
            p_gbr   = mods["gbr"].predict(X_arr)[0]
            p_ens   = (p_ridge + p_rf + p_gbr) / 3

            # Historical distribution for this month (for percentile context)
            hist = df[(df["month"] == month) & df[target_name].notna()][target_name]
            pct  = (hist < p_ens).mean() * 100

            if target_name == "WTEQ":
                records.append({
                    "year": target_year, "month": month, "month_name": MONTH_NAMES[month],
                    "wteq_ridge": round(p_ridge, 2), "wteq_rf": round(p_rf, 2),
                    "wteq_gbr": round(p_gbr, 2), "wteq_ensemble": round(p_ens, 2),
                    "wteq_hist_mean": round(hist.mean(), 2), "wteq_hist_std": round(hist.std(), 2),
                    "wteq_pct": round(pct, 1),
                })
            else:
                for r in records:
                    if r["month"] == month:
                        r.update({
                            "snow_ridge": round(p_ridge, 2), "snow_rf": round(p_rf, 2),
                            "snow_gbr": round(p_gbr, 2), "snow_ensemble": round(p_ens, 2),
                            "snow_hist_mean": round(hist.mean(), 2), "snow_hist_std": round(hist.std(), 2),
                            "snow_pct": round(pct, 1),
                        })

    fc_df = pd.DataFrame(records)
    return fc_df


# ── 5. Analog year analysis ────────────────────────────────────────────────────

def find_analogs(df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    """
    Find the N historical years whose Oct-Jan teleconnection pattern
    is most similar to 2025-26. Uses Euclidean distance in normalized space.
    """
    print("\n[7] Finding analog years ...")
    # Reference period: Oct-Jan for the forecast year (2025-26 season)
    ref_months = [(2025, 10), (2025, 11), (2025, 12), (2026, 1)]
    compare_cols = ["ao","enso34","pdo","pna"]  # most data-complete indices
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

    # Pull their winter snowfall
    snow_data = []
    for _, row in scores_df.iterrows():
        yr = int(row["year"])
        for m in [11, 12, 1, 2, 3, 4]:
            sub = df[(df["year"] == yr) & (df["month"] == m)]
            if len(sub) > 0:
                snow_data.append({
                    "analog_year": yr,
                    "month": m,
                    "month_name": MONTH_NAMES[m],
                    "WTEQ": sub["WTEQ"].values[0],
                    "snow_inches": sub["snow_inches"].values[0],
                    "distance": row["distance"],
                })
    analogs_detail = pd.DataFrame(snow_data)
    print(f"   Top {n} analog years: {list(scores_df['year'].astype(int))}")
    return scores_df, analogs_detail


# ── 6. Feature importance ──────────────────────────────────────────────────────

def get_feature_importances(models: dict, target: str) -> pd.DataFrame:
    feats = models["features"]
    rf_imp    = pd.Series(models["rf"]["model"].feature_importances_,  index=feats)
    gbr_imp   = pd.Series(models["gbr"]["model"].feature_importances_, index=feats)
    ridge_imp = pd.Series(np.abs(models["ridge"]["model"].coef_),       index=feats)
    # Normalize each
    rf_imp    /= rf_imp.sum()  if rf_imp.sum()    > 0 else 1
    gbr_imp   /= gbr_imp.sum() if gbr_imp.sum()   > 0 else 1
    ridge_imp /= ridge_imp.sum() if ridge_imp.sum() > 0 else 1
    combined   = (rf_imp + gbr_imp + ridge_imp) / 3
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


def plot_feature_importance(imp_wteq: pd.DataFrame, imp_snow: pd.DataFrame):
    print("[Plot] Feature importance ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, imp, title in zip(axes, [imp_wteq, imp_snow], ["WTEQ (SWE)", "Snow Inches"]):
        top = imp.head(20)
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, unit, forecast_col in [
        (axes[0], "WTEQ",        "SWE (inches)",      "wteq_ensemble"),
        (axes[1], "snow_inches", "Snowfall (inches)", "snow_ensemble"),
    ]:
        # Historical mean
        hist_mean = []
        hist_lo   = []
        hist_hi   = []
        for m in month_order:
            vals = df[(df["month"] == m) & df[col].notna()][col]
            hist_mean.append(vals.mean())
            hist_lo.append(vals.quantile(0.25))
            hist_hi.append(vals.quantile(0.75))

        ax.fill_between(month_labels, hist_lo, hist_hi,
                        alpha=0.2, color="grey", label="Historical IQR (all years)")
        ax.plot(month_labels, hist_mean, "k--", lw=1.5, label="Historical mean")

        # Analog year traces
        cmap = plt.cm.tab10
        for i, yr in enumerate(analog_years):
            sub = analogs_detail[analogs_detail["analog_year"] == yr]
            ys  = [sub[sub["month"] == m][col].values[0] if len(sub[sub["month"]==m]) > 0 else np.nan
                   for m in month_order]
            ax.plot(month_labels, ys, "o-", color=cmap(i), alpha=0.75, lw=1.5, label=f"{yr}")

        # Forecast (2025-26)
        fc_vals = []
        for m in month_order:
            fc_row = fc_df[fc_df["month"] == m]
            if len(fc_row) > 0:
                fc_vals.append(fc_row[forecast_col].values[0])
            else:
                # Observed
                obs = df[(df["year"].isin([2025,2026])) & (df["month"] == m)]
                fc_vals.append(obs[col].values[0] if len(obs) > 0 and not obs[col].isna().all() else np.nan)

        # Observed 2025-26
        obs_vals = []
        for m in month_order:
            obs = df[((df["year"] == 2025) & (df["month"] == m)) |
                     ((df["year"] == 2026) & (df["month"] == m))]
            obs_vals.append(obs[col].values[0] if len(obs) > 0 and not obs[col].isna().all() else np.nan)

        ax.plot(month_labels, obs_vals, "ko-", lw=2.5, ms=7, label="2025-26 Observed")
        ax.plot(month_labels, fc_vals,  "r^--", lw=2.5, ms=8, label="2025-26 Forecast")

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
    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

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

    # --- Panel 2: Forecast bar chart ---
    ax2 = fig.add_subplot(gs[1, 0])
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

    # --- Panel 3: Forecast bar chart for snowfall ---
    ax3 = fig.add_subplot(gs[1, 1])
    ens_s   = fc_df["snow_ensemble"].tolist() if "snow_ensemble" in fc_df.columns else []
    hmean_s = fc_df["snow_hist_mean"].tolist() if "snow_hist_mean" in fc_df.columns else []
    hstd_s  = fc_df["snow_hist_std"].tolist()  if "snow_hist_std" in fc_df.columns else []
    if ens_s:
        ax3.bar(x - 0.2, hmean_s, width=0.35, color="lightsteelblue", label="Historical mean")
        ax3.errorbar(x - 0.2, hmean_s, yerr=hstd_s, fmt="none", color="steelblue", capsize=4)
        ax3.bar(x + 0.2, ens_s, width=0.35, color="tomato", label="Forecast")
        ax3.set_xticks(x); ax3.set_xticklabels(months_fc)
    ax3.set_title("Forecast vs Historical Mean: Snowfall", fontsize=11)
    ax3.set_ylabel("Snowfall (inches)"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Snoqualmie Pass Snowpack Forecast — Winter 2025-26\n"
                 f"Current conditions: AO={-2.05:.2f}, ONI={-0.55:.2f}, PDO≈{-3.51:.2f}, PNA={0.79:.2f}",
                 fontsize=11)
    plt.savefig(os.path.join(PLOTS, "forecast_2025_2026.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/forecast_2025_2026.png")


def plot_telecon_current_state(df: pd.DataFrame):
    """Bar chart of current teleconnection indices vs historical range."""
    print("[Plot] Current teleconnection state ...")
    current = {
        "AO":      -2.05,
        "ONI":     -0.55,
        "PDO":     -3.51,  # Aug 2025, most recent
        "PNA":      0.79,
        "NAO":     -0.36,
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
    print("\n" + "="*60)
    print("  Snoqualmie Pass Snowpack Forecasting Tool")
    print("  Target season: Winter 2025-2026")
    print("="*60 + "\n")

    # --- Load and build dataset ---
    df = load_base()
    df = patch_fresh_telecons(df)
    df = patch_fresh_snotel(df)
    df = build_features(df)

    # --- Train models ---
    tele_cols = [c for c in CORE_TELE if c in df.columns]

    models_wteq = train_models(df, "WTEQ", months=WINTER_MONTHS)
    models_snow = train_models(df, "snow_inches", months=WINTER_MONTHS)

    # --- Feature importances ---
    imp_wteq = get_feature_importances(models_wteq, "WTEQ")
    imp_snow = get_feature_importances(models_snow, "snow_inches")
    imp_wteq.to_csv(os.path.join(DATA, "feature_importance_wteq.csv"), index=False)
    imp_snow.to_csv(os.path.join(DATA, "feature_importance_snow.csv"), index=False)
    print(f"\n   Top 10 features for WTEQ:")
    print(imp_wteq.head(10)[["feature","combined"]].to_string(index=False))

    # --- Forecast ---
    fc_df = forecast_season(df, models_wteq, models_snow, target_year=2026)
    fc_df.to_csv(os.path.join(DATA, "forecast_results.csv"), index=False, encoding="utf-8")
    print("\n" + "="*60)
    print("  FORECAST: Snoqualmie Pass — 2026")
    print("="*60)
    for _, r in fc_df.iterrows():
        print(f"\n  {r['month_name']} 2026:")
        if "wteq_ensemble" in r:
            print(f"    SWE:       {r.get('wteq_ensemble','?'):.1f}\" "
                  f"(hist mean {r.get('wteq_hist_mean','?'):.1f}\" +/- {r.get('wteq_hist_std','?'):.1f}\")  "
                  f"-> {r.get('wteq_pct','?'):.0f}th percentile")
        if "snow_ensemble" in r:
            print(f"    Snowfall:  {r.get('snow_ensemble','?'):.1f}\" "
                  f"(hist mean {r.get('snow_hist_mean','?'):.1f}\" +/- {r.get('snow_hist_std','?'):.1f}\")  "
                  f"-> {r.get('snow_pct','?'):.0f}th percentile")

    # --- Analog years ---
    analog_scores, analogs_detail = find_analogs(df, n=7)
    analog_scores.to_csv(os.path.join(DATA, "analog_years.csv"), index=False)
    analogs_detail.to_csv(os.path.join(DATA, "analog_detail.csv"), index=False)

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
    print(f"  data/feature_importance_wteq.csv")
    print(f"  plots/correlation_heatmap.png")
    print(f"  plots/feature_importance.png")
    print(f"  plots/analog_years.png")
    print(f"  plots/forecast_2025_2026.png")
    print(f"  plots/current_telecon_state.png")
    print("="*60)


if __name__ == "__main__":
    main()
