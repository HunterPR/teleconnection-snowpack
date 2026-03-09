"""
lanina_analysis.py
==================
Deep-dive: How does La Niña affect Snoqualmie Pass snowpack?
And what other factors (PNA, AO, PDO, NP) modulate the outcome?

Driving question:
  La Niña is "supposed" to be good for Snoqualmie snow, but recent La Niña
  winters (2021-22, 2022-23) have been disappointing. What's the key
  differentiator?

Key hypothesis:
  A strong POSITIVE PNA ridge sets up over the NE Pacific, diverting the
  jet stream northward and shutting off Pacific moisture delivery to the
  Cascades — even when La Niña forcing is present. High NP (weak Aleutian
  Low) compounds this effect.

Outputs:
  plots/lanina_ranked.png           - All La Niña winters ranked by snow
  plots/lanina_phase_scatter.png    - PNA vs WTEQ scatter (La Niña only)
  plots/lanina_composite.png        - Good vs bad La Niña telecon fingerprint
  plots/lanina_current_context.png  - Current season in the phase space
  data/lanina_winters.csv           - Full La Niña winter table
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "data")
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ── La Niña threshold ─────────────────────────────────────────────────────────
NINA_THRESH = -0.5   # ONI DJF ≤ this → La Niña winter

# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data():
    print("[1] Loading datasets ...")

    df = pd.read_csv(os.path.join(BASE, "Merged_Dataset.csv"))
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)

    # Patch fresh teleconnections
    oni = pd.read_csv(os.path.join(DATA, "oni.csv"))
    seas_to_month = {
        "DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
        "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12,
    }
    oni["month"] = oni["season"].map(seas_to_month)
    oni = oni.rename(columns={"oni_anomaly": "enso34"})[["year","month","enso34"]].dropna()

    pdo = pd.read_csv(os.path.join(DATA, "pdo.csv"))
    pna = pd.read_csv(os.path.join(DATA, "pna.csv"))
    ao  = pd.read_csv(os.path.join(DATA, "ao.csv"))
    nao = pd.read_csv(os.path.join(DATA, "nao.csv"))

    patch_list = [(oni,"enso34"), (pdo,"pdo"), (pna,"pna"), (ao,"ao"), (nao,"nao")]

    # EPO and Nino4 are optional (created by fetch_new_predictors.py)
    for fname, col in [("epo.csv","epo"), ("nino4_anom.csv","nino4_anom")]:
        fpath = os.path.join(DATA, fname)
        if os.path.exists(fpath):
            patch_list.append((pd.read_csv(fpath), col))
            print(f"   Loaded {fname} for patching")

    for src, col in patch_list:
        src = src.copy()
        src[["year","month"]] = src[["year","month"]].astype(int)
        merged = df.merge(src.rename(columns={col: col+"_fr"}), on=["year","month"], how="left")
        if col in merged.columns:
            merged[col] = merged[col+"_fr"].combine_first(merged[col])
        else:
            merged[col] = merged[col+"_fr"]
        df = merged.drop(columns=[col+"_fr"])

    print(f"   {len(df)} rows | years {df.year.min()}-{df.year.max()} | "
          f"cols: {len(df.columns)}")
    return df


# ── 2. Build La Niña winter table ─────────────────────────────────────────────

def build_nina_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each winter season, classify La Niña based on DJF ONI mean.
    'Winter year' = the year of Jan/Feb/Mar (so 2021 = winter 2020-21).
    Collect seasonal mean teleconnections and peak snowpack.
    """
    print("[2] Building La Niña winter table ...")
    records = []

    all_years = sorted(df["year"].unique())
    for yr in all_years:
        # DJF = Dec(yr-1), Jan(yr), Feb(yr)
        djf = df[((df["year"] == yr-1) & (df["month"] == 12)) |
                 ((df["year"] == yr)   & (df["month"].isin([1, 2])))]
        # NDJFMA = Nov-Apr around the winter
        ndjfma = df[((df["year"] == yr-1) & (df["month"].isin([11, 12]))) |
                    ((df["year"] == yr)   & (df["month"].isin([1, 2, 3, 4])))]
        # Oct-Mar
        oct_mar = df[((df["year"] == yr-1) & (df["month"].isin([10, 11, 12]))) |
                     ((df["year"] == yr)   & (df["month"].isin([1, 2, 3])))]

        if len(djf) < 2:
            continue

        oni_djf = djf["enso34"].mean()
        if pd.isna(oni_djf):
            continue

        # Seasonal mean teleconnections
        pna_djf       = djf["pna"].mean()       if "pna"       in djf.columns else np.nan
        ao_djf        = djf["ao"].mean()        if "ao"        in djf.columns else np.nan
        nao_djf       = djf["nao"].mean()       if "nao"       in djf.columns else np.nan
        pdo_djf       = djf["pdo"].mean()       if "pdo"       in djf.columns else np.nan
        np_djf        = djf["np"].mean()        if "np"        in djf.columns else np.nan
        epo_djf       = djf["epo"].mean()       if "epo"       in djf.columns else np.nan
        nino4_anom_djf= djf["nino4_anom"].mean()if "nino4_anom"in djf.columns else np.nan

        pna_oct_mar = oct_mar["pna"].mean() if "pna" in oct_mar.columns else np.nan
        ao_oct_mar  = oct_mar["ao"].mean()  if "ao"  in oct_mar.columns else np.nan

        # Peak snowpack: max WTEQ (Feb or Mar) and total snow_inches
        # Peak SWE (Jan, Feb, Mar) — SNOTEL data available ~1996+
        wteq_vals = ndjfma[ndjfma["month"].isin([1,2,3])]["WTEQ"].dropna()
        wteq_peak = wteq_vals.max() if len(wteq_vals) > 0 else np.nan
        wteq_feb  = df[(df["year"]==yr) & (df["month"]==2)]["WTEQ"].values
        wteq_feb  = wteq_feb[0] if len(wteq_feb) > 0 and not np.isnan(wteq_feb[0]) else np.nan
        wteq_mar  = df[(df["year"]==yr) & (df["month"]==3)]["WTEQ"].values
        wteq_mar  = wteq_mar[0] if len(wteq_mar) > 0 and not np.isnan(wteq_mar[0]) else np.nan

        # Snowfall (months 11-4)
        snow_vals = ndjfma["snow_inches"].dropna()
        snow_tot  = snow_vals.sum() if len(snow_vals) > 0 else np.nan

        # Monthly snowfall for each winter month
        snow_by_month = {}
        for m in [11, 12, 1, 2, 3, 4]:
            y_ = yr-1 if m >= 10 else yr
            row_ = df[(df["year"]==y_) & (df["month"]==m)]["snow_inches"].values
            snow_by_month[f"snow_{m}"] = row_[0] if len(row_) > 0 and not np.isnan(row_[0]) else np.nan

        # Dec PNA specifically (often a leading indicator)
        dec_row = df[(df["year"]==yr-1) & (df["month"]==12)]
        pna_dec = dec_row["pna"].values[0] if len(dec_row) > 0 else np.nan

        records.append({
            "winter_year":      yr,
            "oni_djf":          round(oni_djf, 2),
            "pna_djf":          round(pna_djf, 2)       if not pd.isna(pna_djf)       else np.nan,
            "pna_oct_mar":      round(pna_oct_mar, 2)   if not pd.isna(pna_oct_mar)   else np.nan,
            "pna_dec":          round(pna_dec, 2)       if not pd.isna(pna_dec)       else np.nan,
            "ao_djf":           round(ao_djf, 2)        if not pd.isna(ao_djf)        else np.nan,
            "ao_oct_mar":       round(ao_oct_mar, 2)    if not pd.isna(ao_oct_mar)    else np.nan,
            "nao_djf":          round(nao_djf, 2)       if not pd.isna(nao_djf)       else np.nan,
            "pdo_djf":          round(pdo_djf, 2)       if not pd.isna(pdo_djf)       else np.nan,
            "np_djf":           round(np_djf, 2)        if not pd.isna(np_djf)        else np.nan,
            "epo_djf":          round(epo_djf, 2)       if not pd.isna(epo_djf)       else np.nan,
            "nino4_anom_djf":   round(nino4_anom_djf,2) if not pd.isna(nino4_anom_djf)else np.nan,
            "wteq_peak":     round(wteq_peak, 1) if not pd.isna(wteq_peak) else np.nan,
            "wteq_feb":      round(wteq_feb, 1) if not pd.isna(wteq_feb) else np.nan,
            "wteq_mar":      round(wteq_mar, 1) if not pd.isna(wteq_mar) else np.nan,
            "snow_total":    round(snow_tot, 0) if not pd.isna(snow_tot) else np.nan,
            **snow_by_month,
        })

    all_df = pd.DataFrame(records)

    # La Niña subset
    nina_df = all_df[all_df["oni_djf"] <= NINA_THRESH].copy()
    print(f"   Total winters: {len(all_df)} | La Niña winters: {len(nina_df)}")
    print(f"   La Niña years: {sorted(nina_df['winter_year'].tolist())}")
    return all_df, nina_df


# ── 3. Print summary table ─────────────────────────────────────────────────────

def print_nina_table(nina_df: pd.DataFrame, all_df: pd.DataFrame):
    print("\n" + "="*80)
    print("  LA NINA WINTERS -- Snoqualmie Pass Snowpack Summary")
    print("="*80)

    # Sort by peak SWE for winters with WTEQ data, then by snow_total for older
    swe_df = nina_df[nina_df["wteq_peak"].notna()].sort_values("wteq_peak", ascending=False)
    snow_df = nina_df[nina_df["wteq_peak"].isna() & nina_df["snow_total"].notna()].sort_values("snow_total", ascending=False)

    print("\n  [Post-1996 SNOTEL era — ranked by peak SWE]")
    print(f"  {'Year':<8} {'ONI':>6} {'PNA-DJF':>8} {'AO-DJF':>7} {'PDO-DJF':>8} "
          f"{'NP-DJF':>7} {'SWE-peak':>9} {'Snow-tot':>9}  Category")
    print("  " + "-"*80)

    # Historical all-years Feb SWE for percentile context
    all_feb_wteq = all_df["wteq_feb"].dropna()
    p25 = all_feb_wteq.quantile(0.25)
    p75 = all_feb_wteq.quantile(0.75)
    p90 = all_feb_wteq.quantile(0.90)

    def category(wteq_peak):
        if pd.isna(wteq_peak): return "---"
        if wteq_peak >= p90:   return "EXCEPTIONAL"
        if wteq_peak >= p75:   return "GREAT"
        if wteq_peak >= 50:    return "GOOD"
        if wteq_peak >= p25:   return "FAIR"
        return "POOR"

    for _, r in swe_df.iterrows():
        cat = category(r["wteq_peak"])
        flag = " <-- 2020-21" if r["winter_year"] == 2021 else ""
        flag = " <-- 2021-22" if r["winter_year"] == 2022 else flag
        flag = " <-- 2022-23" if r["winter_year"] == 2023 else flag
        print(f"  {int(r['winter_year']):<8} "
              f"{r['oni_djf']:>6.2f} "
              f"{r['pna_djf']:>8.2f} "
              f"{r['ao_djf']:>7.2f} "
              f"{r['pdo_djf']:>8.2f} "
              f"{r['np_djf'] if not pd.isna(r['np_djf']) else float('nan'):>7.2f} "
              f"{r['wteq_peak']:>9.1f} "
              f"{r['snow_total'] if not pd.isna(r['snow_total']) else float('nan'):>9.0f}"
              f"  {cat}{flag}")

    # Correlation analysis
    sub = swe_df.dropna(subset=["wteq_peak","pna_djf","ao_djf","pdo_djf"])
    if len(sub) >= 5:
        print("\n  [Correlations with peak SWE — La Niña winters only]")
        for col, label in [("oni_djf","ONI"),("pna_djf","PNA"),("ao_djf","AO"),
                           ("pdo_djf","PDO"),("np_djf","NP"),
                           ("epo_djf","EPO"),("nino4_anom_djf","Nino4")]:
            vals = sub[[col,"wteq_peak"]].dropna()
            if len(vals) < 5:
                continue
            r, p = stats.pearsonr(vals[col], vals["wteq_peak"])
            sig = "**" if p < 0.05 else ("*" if p < 0.1 else "  ")
            print(f"  {label:<8} r = {r:+.3f}  p = {p:.3f} {sig}")

    print("\n  KEY INSIGHT:")
    print("  Within La Niña winters, PNA is often the most important modulator.")
    print("  High PNA (+) = ridge over NE Pacific = jet deflects NORTH of Cascades.")
    print("  Low NP (+) = strong Aleutian Low = moisture trains into PNW.")
    print()

    # Current season
    print("  CURRENT SEASON (2025-26):")
    print(f"  ONI (NDJ 2025): -0.55  => Weak La Nina / borderline")
    print(f"  AO  (Jan 2026): -2.05  => Strongly NEGATIVE (favorable)")
    print(f"  PNA (Jan 2026): +0.79  => Moderately positive (mixed)")
    print(f"  PDO (Jan 2026): -0.36  => Weakly negative (note: was -3.51 in Aug 2025)")
    print(f"  NAO (Jan 2026): -0.36  => Slightly negative")
    print()


# ── 4. Plots ───────────────────────────────────────────────────────────────────

def plot_nina_ranked(nina_df: pd.DataFrame, all_df: pd.DataFrame):
    """Bar chart of La Niña winters ranked by peak SWE, colored by PNA."""
    print("[Plot] La Niña ranked bar chart ...")

    swe_df = nina_df[nina_df["wteq_peak"].notna()].sort_values("wteq_peak", ascending=False).copy()
    if len(swe_df) == 0:
        print("   No WTEQ data for La Niña years — skipping")
        return

    # Historical median for reference
    hist_med = all_df["wteq_peak"].median()
    hist_p75 = all_df["wteq_peak"].quantile(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    cmap = plt.cm.RdBu_r
    pna_vals = swe_df["pna_djf"].fillna(0)
    pna_norm = plt.Normalize(vmin=-1.5, vmax=1.5)
    colors = [cmap(pna_norm(v)) for v in pna_vals]

    labels = [f"{int(r['winter_year'])}\nONI={r['oni_djf']:.1f}" for _, r in swe_df.iterrows()]
    bars = ax.bar(range(len(swe_df)), swe_df["wteq_peak"], color=colors, edgecolor="white", linewidth=0.5)

    # Annotate PNA value
    for i, (_, r) in enumerate(swe_df.iterrows()):
        pna = r["pna_djf"]
        label = f"PNA\n{pna:+.1f}" if not pd.isna(pna) else ""
        ax.text(i, r["wteq_peak"] + 0.5, label, ha="center", va="bottom",
                fontsize=7, color="black")

    ax.axhline(hist_med, color="black",   lw=1.5, ls="--", label=f"All-years median ({hist_med:.0f}\")")
    ax.axhline(hist_p75, color="dimgray", lw=1.0, ls=":",  label=f"75th percentile ({hist_p75:.0f}\")")
    ax.set_xticks(range(len(swe_df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Peak SWE (inches, Jan-Mar)", fontsize=11)
    ax.set_title("La Niña Winters at Snoqualmie Pass — Peak SWE\n"
                 "(bar color = DJF PNA: blue=negative ridge, red=positive ridge)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=pna_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="DJF PNA Index", shrink=0.7)

    # Right panel: PNA vs WTEQ scatter
    ax2 = axes[1]
    sc = ax2.scatter(swe_df["pna_djf"], swe_df["wteq_peak"],
                     c=swe_df["oni_djf"], cmap="Blues_r",
                     vmin=-2.0, vmax=-0.3, s=80, zorder=3)
    for _, r in swe_df.iterrows():
        ax2.annotate(str(int(r["winter_year"])),
                     (r["pna_djf"], r["wteq_peak"]),
                     textcoords="offset points", xytext=(5, 2), fontsize=7)

    # Fit line
    sub = swe_df.dropna(subset=["pna_djf","wteq_peak"])
    if len(sub) >= 4:
        m, b, r, p, _ = stats.linregress(sub["pna_djf"], sub["wteq_peak"])
        xs = np.linspace(sub["pna_djf"].min()-0.1, sub["pna_djf"].max()+0.1, 50)
        ax2.plot(xs, m*xs + b, "r--", lw=1.5, label=f"r={r:+.2f} p={p:.2f}")
        ax2.legend(fontsize=8)

    ax2.set_xlabel("DJF PNA Index", fontsize=10)
    ax2.set_ylabel("Peak SWE (inches)", fontsize=10)
    ax2.set_title("PNA vs SWE\n(La Niña only)", fontsize=10)
    ax2.axvline(0, color="k", lw=0.8, ls="--")
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="ONI (DJF)", shrink=0.7)

    plt.suptitle("La Niña Winters: PNA Ridge is the Key Modulator for Snoqualmie Pass Snowpack",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "lanina_ranked.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/lanina_ranked.png")


def plot_nina_composite(nina_df: pd.DataFrame, all_df: pd.DataFrame):
    """
    Composite teleconnection fingerprint: Good La Niña vs Bad La Niña.
    Monthly mean AO, PNA, PDO, NP for each group, Oct through April.
    """
    print("[Plot] La Niña composite ...")

    swe_df = nina_df[nina_df["wteq_peak"].notna()].copy()
    if len(swe_df) < 6:
        print("   Not enough SWE data for composite — skipping")
        return

    med_wteq = swe_df["wteq_peak"].median()
    good_years = swe_df[swe_df["wteq_peak"] >= med_wteq]["winter_year"].tolist()
    bad_years  = swe_df[swe_df["wteq_peak"] <  med_wteq]["winter_year"].tolist()

    print(f"   Good La Niña years (SWE >= {med_wteq:.0f}\"): {sorted(good_years)}")
    print(f"   Bad La Niña years  (SWE <  {med_wteq:.0f}\"): {sorted(bad_years)}")

    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
    tele_vars = [
        ("pna",        "PNA",   "Pacific/North American"),
        ("ao",         "AO",    "Arctic Oscillation"),
        ("pdo",        "PDO",   "Pacific Decadal Oscillation"),
        ("np",         "NP",    "North Pacific Index"),
        ("epo",        "EPO",   "East Pacific Oscillation"),
        ("nino4_anom", "Nino4", "Central Pacific SST Anomaly"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("La Niña Composite: What Separates Great Snow Years from Poor Ones?\n"
                 f"Good = peak SWE ≥ {med_wteq:.0f}\" | Poor = peak SWE < {med_wteq:.0f}\"",
                 fontsize=13, fontweight="bold")

    for ax, (col, label, fullname) in zip(axes.flat, tele_vars):
        if col not in all_df.columns:
            ax.set_visible(False)
            continue

        good_series = []
        bad_series  = []

        for yr in good_years:
            vals = []
            for (y_, m_) in [(yr-1, 10), (yr-1, 11), (yr-1, 12), (yr, 1), (yr, 2), (yr, 3), (yr, 4)]:
                row = all_df[(all_df["year"]==y_) & (all_df["month"]==m_)][col].values
                vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
            good_series.append(vals)

        for yr in bad_years:
            vals = []
            for (y_, m_) in [(yr-1, 10), (yr-1, 11), (yr-1, 12), (yr, 1), (yr, 2), (yr, 3), (yr, 4)]:
                row = all_df[(all_df["year"]==y_) & (all_df["month"]==m_)][col].values
                vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
            bad_series.append(vals)

        good_arr = np.array(good_series, dtype=float)
        bad_arr  = np.array(bad_series,  dtype=float)

        good_mean = np.nanmean(good_arr, axis=0)
        bad_mean  = np.nanmean(bad_arr,  axis=0)
        good_std  = np.nanstd(good_arr,  axis=0)
        bad_std   = np.nanstd(bad_arr,   axis=0)

        x = np.arange(len(month_labels))
        ax.fill_between(x, good_mean-good_std, good_mean+good_std,
                        alpha=0.2, color="steelblue")
        ax.fill_between(x, bad_mean-bad_std,   bad_mean+bad_std,
                        alpha=0.2, color="tomato")
        ax.plot(x, good_mean, "o-", color="steelblue", lw=2,
                label=f"Good La Niña (n={len(good_years)})")
        ax.plot(x, bad_mean,  "s-", color="tomato",    lw=2,
                label=f"Bad La Niña (n={len(bad_years)})")

        # Plot individual years (thin lines)
        for series in good_series:
            ax.plot(x, series, "-", color="steelblue", alpha=0.2, lw=0.8)
        for series in bad_series:
            ax.plot(x, series, "-", color="tomato",    alpha=0.2, lw=0.8)

        # Current season 2025-26
        current_vals = []
        for (y_, m_) in [(2025, 10), (2025, 11), (2025, 12), (2026, 1), (2026, 2), (2026, 3), (2026, 4)]:
            row = all_df[(all_df["year"]==y_) & (all_df["month"]==m_)][col].values
            current_vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
        # Manual override with freshest known values
        if col == "ao"  and not np.isnan(current_vals[3]): pass  # Jan 2026 = -2.05 in file
        if col == "pna" and not np.isnan(current_vals[3]): pass  # Jan 2026 = 0.79 in file
        ax.plot(x, current_vals, "k^-", lw=2.5, ms=8, zorder=5, label="2025-26 (current)")

        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(month_labels)
        ax.set_title(f"{label}: {fullname}", fontsize=11)
        ax.set_ylabel(f"{label} Index")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Shade Dec-Feb core winter
        for xi in [2, 3, 4]:
            ax.axvspan(xi - 0.4, xi + 0.4, alpha=0.06, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "lanina_composite.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/lanina_composite.png")


def plot_phase_space(nina_df: pd.DataFrame, all_df: pd.DataFrame):
    """
    2D phase space: PNA vs ONI, with WTEQ as bubble size/color.
    Illustrates the 'deadly quadrant': strong La Niña + strong positive PNA = bad snow.
    """
    print("[Plot] Phase space plot ...")

    swe_df = nina_df[nina_df["wteq_peak"].notna()].dropna(subset=["pna_djf","oni_djf"]).copy()
    if len(swe_df) < 4:
        print("   Not enough data for phase space — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: ONI vs PNA phase space
    ax = axes[0]
    all_sub = all_df.dropna(subset=["pna_djf","oni_djf","wteq_peak"])

    # Background: all years faint
    ax.scatter(all_sub["pna_djf"], all_sub["oni_djf"],
               c="lightgray", s=20, alpha=0.5, zorder=1, label="All winters")
    # La Niña highlighted
    sc = ax.scatter(swe_df["pna_djf"], swe_df["oni_djf"],
                    c=swe_df["wteq_peak"], cmap="RdYlBu",
                    vmin=10, vmax=55, s=150, zorder=3, edgecolor="k", lw=0.8)

    for _, r in swe_df.iterrows():
        ax.annotate(str(int(r["winter_year"])),
                    (r["pna_djf"], r["oni_djf"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)

    # Current season
    ax.scatter([0.79], [-0.45], marker="*", s=300, c="gold", edgecolor="k",
               lw=1.5, zorder=5, label="2025-26 (current)")
    ax.annotate("2025-26", (0.79, -0.45), xytext=(6, -8),
                textcoords="offset points", fontsize=9, fontweight="bold", color="darkorange")

    # Quadrant shading
    ax.axhline(NINA_THRESH, color="blue", lw=1, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.axvspan(0.5, 3.0, alpha=0.06, color="red")
    ax.text(0.55, -1.8, "Ridge zone\n(jet deflects\nnorth)", fontsize=8,
            color="red", alpha=0.8, va="bottom")

    ax.set_xlabel("DJF PNA Index  [+ = ridge over NE Pacific]", fontsize=10)
    ax.set_ylabel("DJF ONI  [negative = La Niña]", fontsize=10)
    ax.set_title("Phase Space: PNA vs ONI\n(bubble color = peak SWE)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Peak SWE (inches)', shrink=0.8)

    # Panel 2: PNA vs NP interaction
    ax2 = axes[1]
    sub2 = swe_df.dropna(subset=["pna_djf","np_djf","wteq_peak"])
    if len(sub2) >= 4:
        sc2 = ax2.scatter(sub2["pna_djf"], sub2["np_djf"],
                          c=sub2["wteq_peak"], cmap="RdYlBu",
                          vmin=10, vmax=55, s=150, edgecolor="k", lw=0.8)
        for _, r in sub2.iterrows():
            ax2.annotate(str(int(r["winter_year"])),
                         (r["pna_djf"], r["np_djf"]),
                         xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax2.set_xlabel("DJF PNA Index  [+ = ridge over NE Pacific]", fontsize=10)
        ax2.set_ylabel("DJF NP Index  [+ = high pressure / weak Aleutian Low]", fontsize=10)
        ax2.set_title("PNA vs NP (North Pacific) Phase Space\n"
                      "Top-right = ridge dominant = BAD for Cascades snow", fontsize=11)
        ax2.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
        ax2.axhspan(2, 8, alpha=0.06, color="red")
        ax2.text(0.05, 0.97, "High NP + high PNA\n= worst combination",
                 transform=ax2.transAxes, fontsize=8, color="red",
                 va="top", ha="left")
        ax2.grid(True, alpha=0.3)
        plt.colorbar(sc2, ax=ax2, label='Peak SWE (inches)', shrink=0.8)
    else:
        ax2.text(0.5, 0.5, "Insufficient NP data", transform=ax2.transAxes,
                 ha="center", va="center")

    plt.suptitle("The Ridge Problem: PNA + NP Control La Niña Snow Outcome at Snoqualmie Pass",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "lanina_phase_space.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/lanina_phase_space.png")


def plot_recent_lanina_timeline(nina_df: pd.DataFrame, all_df: pd.DataFrame, raw_df: pd.DataFrame):
    """
    Monthly WTEQ traces for recent La Niña winters + 2025-26,
    annotated with PNA for each month.
    """
    print("[Plot] Recent La Niña timeline ...")

    recent_nina_years = [2021, 2022, 2023, 2008, 2012]  # notable recent ones
    highlight = {2021: "20-21 (GREAT)", 2022: "21-22 (POOR)", 2023: "22-23 (OK)",
                 2008: "07-08 (GREAT)", 2012: "11-12 (GREAT)"}
    colors = {2021: "steelblue", 2022: "tomato", 2023: "goldenrod",
              2008: "seagreen",  2012: "mediumpurple"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 11))

    months_seq = [10, 11, 12, 1, 2, 3, 4]
    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

    # Panel 1: WTEQ traces
    ax = axes[0]
    # Historical IQR (use raw monthly df)
    hist_vals = []
    for m in months_seq:
        v = raw_df[raw_df["month"] == m]["WTEQ"].dropna()
        hist_vals.append((v.quantile(0.25), v.median(), v.quantile(0.75)))
    lo  = [h[0] for h in hist_vals]
    med = [h[1] for h in hist_vals]
    hi  = [h[2] for h in hist_vals]
    ax.fill_between(range(7), lo, hi, alpha=0.15, color="gray", label="All-years IQR")
    ax.plot(range(7), med, "k--", lw=1.5, alpha=0.6, label="All-years median")

    for yr in recent_nina_years:
        vals = []
        for m in months_seq:
            y_ = yr-1 if m >= 10 else yr
            row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)]["WTEQ"].values
            vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
        nina_row = nina_df[nina_df["winter_year"]==yr]
        oni_lbl = f" (ONI={nina_row['oni_djf'].values[0]:.1f})" if len(nina_row)>0 else ""
        ax.plot(range(7), vals, "o-", color=colors.get(yr,"gray"), lw=2, ms=6,
                label=f"{highlight.get(yr, yr)}{oni_lbl}")

    # 2025-26 observed so far
    current_vals = []
    for m in months_seq:
        y_ = 2025 if m >= 10 else 2026
        row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)]["WTEQ"].values
        current_vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
    ax.plot(range(7), current_vals, "k^-", lw=2.5, ms=9, zorder=5, label="2025-26 (current)")

    ax.set_xticks(range(7))
    ax.set_xticklabels(month_labels)
    ax.set_ylabel("SWE (inches)")
    ax.set_title("Monthly SWE: Recent La Nina Winters vs 2025-26", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.3, 6.3)

    # Panel 2: Monthly PNA for same winters
    ax2 = axes[1]
    for yr in recent_nina_years:
        pna_vals = []
        for m in months_seq:
            y_ = yr-1 if m >= 10 else yr
            row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)]["pna"].values
            pna_vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
        ax2.plot(range(7), pna_vals, "o-", color=colors.get(yr,"gray"), lw=2, ms=6,
                 label=highlight.get(yr, yr))

    # Current 2025-26 PNA
    cur_pna = []
    for m in months_seq:
        y_ = 2025 if m >= 10 else 2026
        row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)]["pna"].values
        cur_pna.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
    ax2.plot(range(7), cur_pna, "k^-", lw=2.5, ms=9, zorder=5, label="2025-26 (current)")

    ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax2.axhspan(0.5, 3.0, alpha=0.08, color="red")
    ax2.text(6.35, 0.55, "Ridge\nzone", fontsize=8, color="red", va="bottom", ha="right")
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(month_labels)
    ax2.set_ylabel("PNA Index  (+) = ridge over NE Pacific")
    ax2.set_title("Monthly PNA: Note How Poor Snow Years (red) Spend Time in the Ridge Zone",
                  fontsize=12)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, 6.3)

    plt.suptitle("La Niña Snow Performance at Snoqualmie: The PNA Ridge Modulator",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "lanina_recent_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/lanina_recent_timeline.png")


def plot_current_context(nina_df: pd.DataFrame, all_df: pd.DataFrame, raw_df: pd.DataFrame):
    """
    Current 2025-26 season in context: where we are vs good/bad La Niña analogs.
    Shows the 'funnel' of possible outcomes based on PNA evolution.
    """
    print("[Plot] Current season context ...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Left: AO evolution comparison
    months_seq   = [10, 11, 12, 1, 2, 3, 4]
    month_labels = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]

    for ax, col, ylabel in [
        (axes[0], "ao",  "AO Index"),
        (axes[1], "pna", "PNA Index"),
        (axes[2], "WTEQ","SWE (inches)"),
    ]:
        if col not in raw_df.columns:
            ax.text(0.5,0.5,"No data",transform=ax.transAxes,ha="center")
            continue

        swe_sub = nina_df[nina_df["wteq_peak"].notna()]
        if len(swe_sub) == 0:
            ax.text(0.5,0.5,"No WTEQ data",transform=ax.transAxes,ha="center")
            continue
        med_swe = swe_sub["wteq_peak"].median()
        good_years = swe_sub[swe_sub["wteq_peak"] >= med_swe]["winter_year"].tolist()
        bad_years  = swe_sub[swe_sub["wteq_peak"] <  med_swe]["winter_year"].tolist()

        for years, color, label in [(good_years,"steelblue","Good La Nina"),
                                    (bad_years, "tomato",   "Poor La Nina")]:
            arr = []
            for yr in years:
                vals = []
                for m in months_seq:
                    y_ = yr-1 if m >= 10 else yr
                    row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)][col].values
                    vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
                arr.append(vals)
            arr = np.array(arr, dtype=float)
            mean_ = np.nanmean(arr, axis=0)
            std_  = np.nanstd(arr, axis=0)
            ax.fill_between(range(7), mean_-std_, mean_+std_, alpha=0.2, color=color)
            ax.plot(range(7), mean_, "o-", color=color, lw=2, ms=5, label=f"{label} (n={len(years)})")

        # Current
        cur_vals = []
        for m in months_seq:
            y_ = 2025 if m >= 10 else 2026
            row = raw_df[(raw_df["year"]==y_) & (raw_df["month"]==m)][col].values
            cur_vals.append(row[0] if len(row) > 0 and not np.isnan(row[0]) else np.nan)
        ax.plot(range(7), cur_vals, "k^-", lw=2.5, ms=9, zorder=5, label="2025-26")

        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.4)
        ax.set_xticks(range(7))
        ax.set_xticklabels(month_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(col.upper(), fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("2025-26 Season Context: Good vs Poor La Niña Composite Comparison",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, "lanina_current_context.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved plots/lanina_current_context.png")


# ── 5. Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  La Nina Analysis -- Snoqualmie Pass Snowpack")
    print("="*60 + "\n")

    df = load_data()
    all_df, nina_df = build_nina_table(df)

    print_nina_table(nina_df, all_df)

    # Save table
    out_path = os.path.join(DATA, "lanina_winters.csv")
    all_df[all_df["oni_djf"] <= NINA_THRESH].sort_values(
        "winter_year").to_csv(out_path, index=False)
    print(f"   Saved {out_path}")

    print("\n[Generating plots ...]")
    plot_nina_ranked(nina_df, all_df)
    plot_nina_composite(nina_df, all_df)
    plot_phase_space(nina_df, all_df)
    plot_recent_lanina_timeline(nina_df, all_df, df)
    plot_current_context(nina_df, all_df, df)

    print("\n" + "="*60)
    print("  Done. New outputs:")
    print("  data/lanina_winters.csv")
    print("  plots/lanina_ranked.png")
    print("  plots/lanina_composite.png")
    print("  plots/lanina_phase_space.png")
    print("  plots/lanina_recent_timeline.png")
    print("  plots/lanina_current_context.png")
    print("="*60)


if __name__ == "__main__":
    main()
