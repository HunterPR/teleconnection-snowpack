"""
dashboard.py — Snoqualmie Pass Snowpack Forecast Dashboard
==========================================================
Streamlit dashboard for the two-layer forecasting system.

Tabs:
  1. Forecast Overview — SWE + snowfall predictions, bottom line
  2. Sounding & Freezing Level — Vertical profile, wet-bulb, snow level
  3. NWS / NBM — National Weather Service gridpoint forecast + NBM viewer
  4. Nowcast (Layer 2) — Station telemetry, pacing, pressure
  5. Model Performance — CV scores, feature importance, forecast vs actual
  6. Analog Years — Historical analogs and their snowfall/SWE tracks
  7. Teleconnection State — Current indices, trends
  8. Correlation Heatmap — Predictor correlations
  9. Data Explorer — Raw data tables
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
PLOTS = os.path.join(BASE, "plots")

_COMPASS_LABELS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

def _compass(deg: float) -> str:
    """Convert degrees (0-360) to 16-point compass label."""
    return _COMPASS_LABELS[int((deg % 360 + 11.25) / 22.5) % 16]


def build_wind_rose(snd_df, hours=48, mode="recent", title_prefix=""):
    """Build a Plotly Barpolar wind rose from 850 hPa sounding data.

    mode: "recent" = last N hours, "forecast" = next N hours from now.
    """
    t850 = snd_df[snd_df["level_hPa"] == 850].copy()
    if t850.empty:
        return None

    now_utc = pd.Timestamp.now(tz="UTC")
    if mode == "forecast":
        t850 = t850[t850["time"] > now_utc]
        cutoff = now_utc + pd.Timedelta(hours=hours)
        t850 = t850[t850["time"] <= cutoff]
        label_suffix = f"Next {hours}h"
    else:
        cutoff = t850["time"].max() - pd.Timedelta(hours=hours)
        t850 = t850[t850["time"] >= cutoff]
        label_suffix = f"Last {hours}h"

    if t850.empty:
        return None

    t850["speed_mph"] = t850["wind_speed_kph"] * 0.621371
    t850["compass"] = t850["wind_dir"].apply(_compass)

    speed_bins = [(0, 5, "0-5"), (5, 15, "5-15"), (15, 25, "15-25"),
                  (25, 35, "25-35"), (35, 999, "35+")]
    colors = ["#2196F3", "#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    fig = go.Figure()
    for (lo, hi, label), color in zip(speed_bins, colors):
        mask = (t850["speed_mph"] >= lo) & (t850["speed_mph"] < hi)
        subset = t850[mask]
        if subset.empty:
            continue
        counts = subset.groupby("compass").size().reindex(_COMPASS_LABELS, fill_value=0)
        fig.add_trace(go.Barpolar(
            r=counts.values,
            theta=_COMPASS_LABELS,
            name=f"{label} mph",
            marker_color=color,
            opacity=0.85,
        ))

    title = f"{title_prefix}850 hPa Wind Rose ({label_suffix})"
    fig.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(showticklabels=True, tickfont_size=9),
            angularaxis=dict(direction="clockwise", rotation=90),
        ),
        title=title,
        height=420,
        legend=dict(title="Speed (mph)"),
        margin=dict(t=50, b=30),
    )
    return fig


def interpret_storm_track(snd_df, hours=48) -> str:
    """Interpret 850 hPa mean wind for Snoqualmie Pass storm track."""
    t850 = snd_df[snd_df["level_hPa"] == 850].copy()
    if t850.empty:
        return "No 850 hPa wind data available."
    cutoff = t850["time"].max() - pd.Timedelta(hours=hours)
    t850 = t850[t850["time"] >= cutoff]
    if t850.empty:
        return "No recent 850 hPa wind data."

    speed_mph = (t850["wind_speed_kph"] * 0.621371).mean()
    # Circular mean for wind direction
    rads = np.deg2rad(t850["wind_dir"].dropna())
    mean_dir = np.rad2deg(np.arctan2(np.sin(rads).mean(), np.cos(rads).mean())) % 360
    compass = _compass(mean_dir)

    # Regime classification
    if 180 <= mean_dir < 315:
        regime = "Pacific moisture fetch"
        outlook = "Favorable for orographic snowfall at Snoqualmie Pass (if temps are cold enough)."
    elif mean_dir >= 315 or mean_dir < 45:
        regime = "Post-frontal / northwesterly flow"
        outlook = "Colder and drier air mass; scattered snow showers possible, especially in convergence zones."
    else:
        regime = "Easterly / continental"
        outlook = "Rain shadow pattern at Snoqualmie Pass; generally dry conditions expected."

    wind_note = ""
    if speed_mph > 30:
        wind_note = " Strong winds may cause blowing/drifting snow and wind loading on avalanche terrain."
    elif speed_mph < 10:
        wind_note = " Light winds suggest weak forcing."

    return (
        f"Mean 850 hPa flow ({hours}h): {compass} ({mean_dir:.0f} deg) at {speed_mph:.0f} mph. "
        f"Regime: {regime}. {outlook}{wind_note}"
    )

st.set_page_config(
    page_title="Snoqualmie Pass Snowpack Forecast",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Data loading helpers ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_forecast():
    p = os.path.join(DATA, "forecast_results.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_bottom_line():
    p = os.path.join(DATA, "bottom_line.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_nowcast():
    p = os.path.join(DATA, "nowcast.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=300)
def load_sounding():
    p = os.path.join(DATA, "sounding_forecast.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_analogs():
    a = os.path.join(DATA, "analog_years.csv")
    d = os.path.join(DATA, "analog_detail.csv")
    analogs = pd.read_csv(a) if os.path.exists(a) else pd.DataFrame()
    detail = pd.read_csv(d) if os.path.exists(d) else pd.DataFrame()
    return analogs, detail

@st.cache_data(ttl=300)
def load_cv_scores():
    p = os.path.join(DATA, "cv_scores_WTEQ.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_feature_importance(target="wteq"):
    p = os.path.join(DATA, f"feature_importance_{target}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_forecast_vs_actual():
    p = os.path.join(DATA, "forecast_vs_actual_recent.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()


NWS_GRID_URL = "https://api.weather.gov/gridpoints/SEW/152,54"
NWS_HEADERS = {"User-Agent": "(SnoqualmieSnowpackForecast, snoqualmie-forecast@example.com)"}

NBM_VIEWER_URL = (
    "https://apps.gsl.noaa.gov/nbmviewer/?"
    "col=2&hgt=1.1&obs=false&fontsize=1&location=Snoqualmie+Pass"
    "&selectedgroup=Default&darkmode=on&graph=fa-chart-bar"
    "&probfield=Tmax&proboperator=%3E%3D&probvalue=40"
    "&colorfriendly=false&whiskers=false&boxes=true&median=false"
    "&det=true&tz=local"
)


@st.cache_data(ttl=1800)
def load_nws_gridpoint():
    """Fetch NWS gridpoint forecast data for Snoqualmie Pass."""
    import requests
    try:
        r = requests.get(NWS_GRID_URL, headers=NWS_HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("properties", {})
    except Exception as e:
        return {"error": str(e)}


def _parse_nws_ts(prop_data, c_to_f=False, m_to_ft=False, mm_to_in=False,
                  kph_to_mph=False):
    """Parse an NWS gridpoint time series property into a DataFrame."""
    if not prop_data or "values" not in prop_data:
        return pd.DataFrame(columns=["time", "value"])
    records = []
    for v in prop_data["values"]:
        t = v["validTime"].split("/")[0]
        val = v["value"]
        if val is None:
            continue
        if c_to_f:
            val = val * 9 / 5 + 32
        if m_to_ft:
            val = val * 3.28084
        if mm_to_in:
            val = val / 25.4
        if kph_to_mph:
            val = val * 0.621371
        records.append({"time": pd.to_datetime(t, utc=True), "value": val})
    if not records:
        return pd.DataFrame(columns=["time", "value"])
    return pd.DataFrame(records)


# ── Natural-language summary generators ──────────────────────────────────────

MONTH_NAMES = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
               6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
               11: "Nov", 12: "Dec"}
PASS_ELEV = 3022  # ft ASL


def nl_near_term() -> str:
    """Generate near-term (0-48h) NL summary from sounding + NWS data."""
    nc = load_nowcast()
    snd = nc.get("sounding", {})
    parts = []
    if snd and "error" not in snd:
        fzl = snd.get("freezing_level_forecast", {})
        curr_ft = fzl.get("current_ft")
        min_ft = fzl.get("min_48h_ft")
        if curr_ft is not None:
            above = "above" if curr_ft > PASS_ELEV else "below"
            parts.append(f"Freezing level currently at {curr_ft:,}' ASL ({above} pass level).")
            if min_ft is not None and min_ft < PASS_ELEV:
                parts.append(f"Expected to drop to {min_ft:,}' within 48h (below pass).")
        sl = snd.get("snow_level_ft")
        if sl is not None:
            parts.append(f"Snow level (wet-bulb 32F) at ~{sl:,}'.")
        sf = snd.get("snowfall_possible_hours", {})
        h48 = sf.get("next_48h", 0)
        if h48 > 0:
            parts.append(f"{h48} hours of potential snowfall in the next 48h.")
        elif h48 == 0:
            parts.append("No snowfall expected at pass level in the next 48h.")
        w850 = snd.get("wind_850hPa_48h", {})
        if w850:
            d = w850.get("mean_dir_deg", 0)
            s = w850.get("mean_speed_mph", 0)
            if 180 <= d < 315:
                regime = "Pacific moisture fetch (favorable for orographic snow)"
            elif d >= 315 or d < 45:
                regime = "post-frontal northwesterly (colder, scattered showers)"
            else:
                regime = "easterly/continental (rain shadow, generally dry)"
            parts.append(f"850 hPa flow: {d:.0f} deg at {s:.0f} mph -> {regime}.")
    if not parts:
        parts.append("Near-term sounding data unavailable. Run nowcast.py to refresh.")
    return " ".join(parts)


def nl_mid_range() -> str:
    """Generate mid-range (3-7 day) NL summary from NWS gridpoint data."""
    nws = load_nws_gridpoint()
    if not nws or "error" in nws:
        return "NWS forecast data unavailable."
    parts = []
    temp = _parse_nws_ts(nws.get("temperature"), c_to_f=True)
    snow = _parse_nws_ts(nws.get("snowfallAmount"), mm_to_in=True)
    qpf = _parse_nws_ts(nws.get("quantitativePrecipitation"), mm_to_in=True)
    sl = _parse_nws_ts(nws.get("snowLevel"), m_to_ft=True)

    if not temp.empty:
        lo, hi = temp["value"].min(), temp["value"].max()
        parts.append(f"NWS 7-day temps: {lo:.0f}-{hi:.0f}F.")
        if hi < 32:
            parts.append("Remaining below freezing throughout.")
        elif lo > 32:
            parts.append("Staying above freezing; expect rain at pass level.")
    if not snow.empty:
        total = snow["value"].sum()
        if total > 0.5:
            parts.append(f"NWS total snowfall: {total:.1f}\" over the forecast period.")
        else:
            parts.append("Minimal snowfall expected in the NWS 7-day window.")
    if not qpf.empty:
        total_liq = qpf["value"].sum()
        if total_liq > 0.1:
            parts.append(f"Total liquid precip: {total_liq:.2f}\".")
    if not sl.empty:
        sl_min = sl["value"].min()
        sl_max = sl["value"].max()
        below_pass = sl_min < PASS_ELEV
        parts.append(
            f"Snow level range: {sl_min:,.0f}'-{sl_max:,.0f}'"
            f"{' (drops below pass)' if below_pass else ' (stays above pass)'}."
        )
    if not parts:
        parts.append("NWS mid-range data could not be parsed.")
    return " ".join(parts)


def nl_seasonal(fc_df) -> str:
    """Generate seasonal outlook NL summary from teleconnection ML forecasts."""
    if fc_df.empty:
        return "Seasonal forecast data unavailable."
    now = datetime.now()
    current_month = now.month
    parts = []
    # Season total
    total_swe = fc_df["wteq_ensemble"].sum()
    total_snow = fc_df["snow_ensemble"].sum()
    parts.append(
        f"Teleconnection ML ensemble projects {total_swe:.1f}\" SWE and "
        f"{total_snow:.1f}\" snowfall across the forecast window."
    )
    # Current + remaining months
    remaining = fc_df[fc_df["month"] >= current_month]
    if not remaining.empty:
        rem_swe = remaining["wteq_ensemble"].sum()
        rem_snow = remaining["snow_ensemble"].sum()
        months_left = remaining["month_name"].tolist()
        parts.append(
            f"Remaining ({', '.join(months_left)}): {rem_swe:.1f}\" SWE, "
            f"{rem_snow:.1f}\" snowfall projected."
        )
    # Percentile context
    pcts = fc_df[fc_df["month"] >= current_month]["wteq_pct"]
    if not pcts.empty:
        avg_pct = pcts.mean()
        if avg_pct < 25:
            parts.append("Well below historical norms.")
        elif avg_pct < 40:
            parts.append("Below historical norms.")
        elif avg_pct < 60:
            parts.append("Near historical norms.")
        else:
            parts.append("Above historical norms.")
    return " ".join(parts)


# ── Tab definitions ──────────────────────────────────────────────────────────

tabs = st.tabs([
    "Forecast Overview",
    "Sounding & Freezing Level",
    "NWS / NBM",
    "Nowcast (Layer 2)",
    "Model Performance",
    "Analog Years",
    "Teleconnection State",
    "Correlation Heatmap",
    "Data Explorer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Forecast Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    # ── Water year convention ────────────────────────────────────────────
    now = datetime.now()
    current_month = now.month
    # Water year: Oct-Sep. WY26 = Oct 2025 - Sep 2026
    wy = now.year if now.month >= 10 else now.year
    st.header(f"Snoqualmie Pass Snowpack Forecast - WY{wy % 100:02d}")
    st.caption(f"Water Year {wy-1}-{wy} (Oct {wy-1} - Sep {wy})  |  "
               f"Updated {now.strftime('%b %d, %Y %I:%M %p')}")

    fc = load_forecast()
    bl = load_bottom_line()
    fva = load_forecast_vs_actual()

    SNOTEL_URL = "https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=908"
    SNOTEL_STATION = "SNOTEL #908 (Snoqualmie Pass)"

    # ── Forecast Horizons ─────────────────────────────────────────────────
    st.subheader("Forecast Horizons")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("**Near-term (0-48h)**")
        st.markdown(f"_{nl_near_term()}_")
    with h2:
        st.markdown("**Mid-range (3-7 days)**")
        st.markdown(f"_{nl_mid_range()}_")
    with h3:
        st.markdown("**Seasonal Outlook**")
        st.markdown(f"_{nl_seasonal(fc)}_")

    st.divider()

    # ── Bottom line synopsis ──────────────────────────────────────────────
    if bl:
        st.info(bl.get("bottom_line", ""))
        human = bl.get("human_notes", "")
        if human:
            st.caption(f"Forecaster notes: {human}")

    # ── Monthly forecast cards (auto-detect past/current/future) ──────────
    if not fc.empty:
        st.subheader("Monthly Breakdown")
        st.caption(f"SWE actuals from [{SNOTEL_STATION}]({SNOTEL_URL})  |  "
                   f"Snowfall actuals from station depth-gain telemetry  |  "
                   f"Forecast from teleconnection ML ensemble (Layer 1)"
                   + ("  + station blending (Layer 2)" if fc["layer2_weight"].notna().any() else ""))

        cols = st.columns(len(fc))
        for i, (_, row) in enumerate(fc.iterrows()):
            with cols[i]:
                mo_num = int(row.get("month", 0))
                mo_name = row.get("month_name", "?")
                # Year for display: months Oct-Dec are previous calendar year
                mo_year = wy if mo_num < 10 else wy - 1
                swe_fc = row.get("wteq_ensemble", 0)
                snow_fc = row.get("snow_ensemble", 0)
                swe_hist = row.get("wteq_hist_mean", 0)
                snow_hist = row.get("snow_hist_mean", 0)
                swe_pct = row.get("wteq_pct", 50)
                snow_pct = row.get("snow_pct", 50)

                is_past = (mo_num < current_month)
                is_current = (mo_num == current_month)

                # Header with status badge
                if is_past:
                    st.subheader(f"{mo_name} {mo_year}  :white_check_mark:")
                    st.caption("VERIFIED")
                elif is_current:
                    st.subheader(f"{mo_name} {mo_year}  :hourglass_flowing_sand:")
                    st.caption(f"IN PROGRESS (day {now.day})")
                else:
                    st.subheader(f"{mo_name} {mo_year}")
                    st.caption("FORECAST")

                # ── For PAST months: show actual vs forecast ──
                if is_past:
                    actual_row = None
                    if not fva.empty:
                        match = fva[(fva["year"] == mo_year) & (fva["month"] == mo_num)]
                        if not match.empty:
                            actual_row = match.iloc[0]

                    actual_swe = actual_row["actual_wteq"] if actual_row is not None and pd.notna(actual_row.get("actual_wteq")) else None
                    actual_snow = row.get("layer2_actual_snow") if pd.notna(row.get("layer2_actual_snow")) else None

                    if actual_swe is not None:
                        delta_swe = actual_swe - swe_fc
                        st.metric("SWE Actual vs Forecast",
                                  f'{actual_swe:.1f}" (fc: {swe_fc:.1f}")',
                                  delta=f"{delta_swe:+.1f}\"",
                                  delta_color="normal")
                    else:
                        st.metric("SWE Forecast", f'{swe_fc:.1f}"',
                                  delta=f"{swe_pct:.0f}th pct")

                    if actual_snow is not None:
                        delta_snow = actual_snow - snow_fc
                        st.metric("Snow Actual vs Forecast",
                                  f'{actual_snow:.1f}" (fc: {snow_fc:.1f}")',
                                  delta=f"{delta_snow:+.1f}\"",
                                  delta_color="normal")
                    else:
                        st.metric("Snow Forecast", f'{snow_fc:.1f}"',
                                  delta=f"{snow_pct:.0f}th pct")

                    st.caption(f"Hist avg: SWE {swe_hist:.1f}\", Snow {snow_hist:.1f}\"")
                    if actual_swe is not None:
                        st.caption(f"[SNOTEL #908]({SNOTEL_URL})")

                # ── For CURRENT month: show forecast + Layer 2 ──
                elif is_current:
                    st.metric("SWE (inches)", f'{swe_fc:.1f}"',
                              delta=f"{swe_pct:.0f}th pct (avg {swe_hist:.1f}\")")
                    st.metric("Snowfall (inches)", f'{snow_fc:.1f}"',
                              delta=f"{snow_pct:.0f}th pct (avg {snow_hist:.1f}\")")
                    if pd.notna(row.get("layer2_weight")):
                        w = row["layer2_weight"]
                        st.caption(f"Layer 2 blended ({w:.0%} actual)")
                    else:
                        st.caption("Layer 1 only (run nowcast.py)")

                # ── For FUTURE months: show Layer 1 forecast ──
                else:
                    st.metric("SWE (inches)", f'{swe_fc:.1f}"',
                              delta=f"{swe_pct:.0f}th pct (avg {swe_hist:.1f}\")")
                    st.metric("Snowfall (inches)", f'{snow_fc:.1f}"',
                              delta=f"{snow_pct:.0f}th pct (avg {snow_hist:.1f}\")")
                    st.caption("Layer 1 teleconnection forecast")

        # ── Forecast vs Actual bar chart ──────────────────────────────────
        st.subheader("Monthly: Forecast vs Historical vs Actual")

        months = fc["month_name"].tolist()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["SWE (inches)", "Snowfall (inches)"])

        # SWE bars
        fig.add_trace(go.Bar(x=months, y=fc["wteq_ensemble"], name="Forecast SWE",
                             marker_color="#1E88E5"), row=1, col=1)
        fig.add_trace(go.Bar(x=months, y=fc["wteq_hist_mean"],
                             name="Historical Mean SWE",
                             marker_color="#546E7A", opacity=0.6), row=1, col=1)
        # Actual SWE for completed months
        if not fva.empty:
            act_swe = []
            for _, row in fc.iterrows():
                mo = int(row["month"])
                mo_yr = wy if mo < 10 else wy - 1
                if mo < current_month:
                    match = fva[(fva["year"] == mo_yr) & (fva["month"] == mo)]
                    if not match.empty and pd.notna(match.iloc[0].get("actual_wteq")):
                        act_swe.append(match.iloc[0]["actual_wteq"])
                    else:
                        act_swe.append(None)
                else:
                    act_swe.append(None)
            if any(v is not None for v in act_swe):
                fig.add_trace(go.Bar(x=months, y=act_swe, name="Actual SWE",
                                     marker_color="#FF7043"), row=1, col=1)

        # Snowfall bars
        fig.add_trace(go.Bar(x=months, y=fc["snow_ensemble"], name="Forecast Snow",
                             marker_color="#43A047"), row=1, col=2)
        fig.add_trace(go.Bar(x=months, y=fc["snow_hist_mean"],
                             name="Historical Mean Snow",
                             marker_color="#546E7A", opacity=0.6), row=1, col=2)
        # Actual snow for completed months (from Layer 2 telemetry)
        act_snow = []
        for _, row in fc.iterrows():
            mo = int(row["month"])
            if mo < current_month and pd.notna(row.get("layer2_actual_snow")):
                act_snow.append(row["layer2_actual_snow"])
            else:
                act_snow.append(None)
        if any(v is not None for v in act_snow):
            fig.add_trace(go.Bar(x=months, y=act_snow, name="Actual Snow",
                                 marker_color="#FF7043"), row=1, col=2)

        fig.update_layout(barmode="group", template="plotly_dark", height=400,
                          margin=dict(t=40, b=30))
        st.plotly_chart(fig, width="stretch")

        # ── WY Season Tracking ────────────────────────────────────────────
        st.subheader(f"WY{wy % 100:02d} Season Tracking")
        season_col1, season_col2 = st.columns(2)
        with season_col1:
            past_months = fc[fc["month"] < current_month]
            if not past_months.empty and not fva.empty:
                cum_fc_swe = past_months["wteq_ensemble"].sum()
                cum_act_swe = 0
                for _, r in past_months.iterrows():
                    mo = int(r["month"])
                    mo_yr = wy if mo < 10 else wy - 1
                    match = fva[(fva["year"] == mo_yr) & (fva["month"] == mo)]
                    if not match.empty and pd.notna(match.iloc[0].get("actual_wteq")):
                        cum_act_swe += match.iloc[0]["actual_wteq"]
                if cum_act_swe > 0:
                    st.metric("Completed - Actual SWE",
                              f'{cum_act_swe:.1f}"',
                              delta=f"{cum_act_swe - cum_fc_swe:+.1f}\" vs forecast ({cum_fc_swe:.1f}\")")

                cum_fc_snow = past_months["snow_ensemble"].sum()
                cum_act_snow = past_months["layer2_actual_snow"].sum()
                if cum_act_snow > 0:
                    st.metric("Completed - Actual Snow",
                              f'{cum_act_snow:.1f}"',
                              delta=f"{cum_act_snow - cum_fc_snow:+.1f}\" vs forecast ({cum_fc_snow:.1f}\")")

        with season_col2:
            remaining = fc[fc["month"] >= current_month]
            if not remaining.empty:
                rem_swe = remaining["wteq_ensemble"].sum()
                rem_snow = remaining["snow_ensemble"].sum()
                st.metric("Remaining Forecast - SWE", f'{rem_swe:.1f}"')
                st.metric("Remaining Forecast - Snow", f'{rem_snow:.1f}"')

        # ── Model spread ─────────────────────────────────────────────────
        st.subheader("Model Spread")
        model_cols_w = [c for c in fc.columns if c.startswith("wteq_") and c not in
                        ["wteq_ensemble", "wteq_spread", "wteq_hist_mean", "wteq_hist_std",
                         "wteq_pct", "wteq_layer1"]]
        if model_cols_w:
            spread_data = []
            for _, row in fc.iterrows():
                mo = row["month_name"]
                for c in model_cols_w:
                    model_name = c.replace("wteq_", "").upper()
                    spread_data.append({"Month": mo, "Model": model_name, "SWE": row[c]})
            spread_df = pd.DataFrame(spread_data)
            fig2 = px.strip(spread_df, x="Month", y="SWE", color="Model",
                            title="Individual Model SWE Predictions")
            fig2.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig2, width="stretch")
    else:
        st.warning("No forecast data found. Run `python forecast.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Sounding & Freezing Level
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Forecast Sounding & Freezing Level")
    st.caption("Open-Meteo ECMWF + GFS pressure-level data for Snoqualmie Pass. "
               "Shows vertical atmosphere structure, freezing/snow levels, and 850 hPa storm flow.")

    nc = load_nowcast()
    snd_data = nc.get("sounding", {})
    snd_df = load_sounding()

    if "error" in snd_data:
        st.warning(f"Sounding error: {snd_data['error']}")
    elif snd_data:
        # Freezing level summary
        fzl = snd_data.get("freezing_level_forecast", {})
        sl = snd_data.get("snow_level_ft")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Freezing Level Now", f"{fzl.get('current_ft', '?')}' ASL")
        c2.metric("48h Min", f"{fzl.get('min_48h_ft', '?')}'")
        c3.metric("48h Max", f"{fzl.get('max_48h_ft', '?')}'")
        if sl is not None:
            c4.metric("Snow Level (WB=32F)", f"{sl}' ASL")

        # Snowfall / snowmaking windows
        sf = snd_data.get("snowfall_possible_hours", {})
        sm = snd_data.get("snowmaking_windows", {})
        w850 = snd_data.get("wind_850hPa_48h", {})

        c5, c6, c7 = st.columns(3)
        c5.metric("Snowfall Hours (48h/120h)",
                  f"{sf.get('next_48h', '?')} / {sf.get('next_120h', '?')}")
        if sm:
            c6.metric("Surface Wet-Bulb", f"{sm.get('current_wetbulb_f', '?')}F")
            c7.metric("Snowmaking Good/Marginal (48h)",
                      f"{sm.get('good_hours_48h', 0)} / {sm.get('marginal_hours_48h', 0)}h")

        # Vertical profile table
        vp = snd_data.get("vertical_profile", [])
        if vp:
            st.subheader("Current Vertical Profile")
            vp_df = pd.DataFrame(vp)
            vp_df = vp_df.rename(columns={
                "level_hPa": "Level (hPa)", "altitude_ft": "Altitude (ft)",
                "temp_f": "Temp (F)", "wetbulb_f": "Wet-Bulb (F)",
                "dewpoint_f": "Dewpoint (F)", "rh_pct": "RH (%)",
                "wind_kph": "Wind (kph)", "wind_dir": "Wind Dir",
            })
            st.dataframe(vp_df, width="stretch", hide_index=True)

        # Freezing level time series
        if not snd_df.empty:
            st.subheader("Freezing Level Forecast (5-day)")
            fzl_ts = snd_df.groupby(["time", "model"])["freezing_level_ft"].first().reset_index()
            fzl_ts = fzl_ts.dropna(subset=["freezing_level_ft"])
            if not fzl_ts.empty:
                fig_fzl = px.line(fzl_ts, x="time", y="freezing_level_ft", color="model",
                                 labels={"freezing_level_ft": "Freezing Level (ft ASL)",
                                         "time": ""},
                                 title="Freezing Level Forecast")
                fig_fzl.add_hline(y=3022, line_dash="dash", line_color="red",
                                  annotation_text="Snoqualmie Pass (3,022')")
                fig_fzl.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_fzl, width="stretch")

            # Wet-bulb profile over time at 850 hPa
            st.subheader("850 hPa Temperature & Wet-Bulb Forecast")
            t850 = snd_df[snd_df["level_hPa"] == 850].copy()
            if not t850.empty:
                fig_t = go.Figure()
                for model in t850["model"].unique():
                    m = t850[t850["model"] == model]
                    fig_t.add_trace(go.Scatter(x=m["time"], y=m["temp_f"],
                                               name=f"{model} Temp", mode="lines"))
                    fig_t.add_trace(go.Scatter(x=m["time"], y=m["wetbulb_f"],
                                               name=f"{model} WetBulb", mode="lines",
                                               line=dict(dash="dot")))
                fig_t.add_hline(y=32, line_dash="dash", line_color="cyan",
                                annotation_text="32F (Freezing)")
                fig_t.update_layout(template="plotly_dark", height=350,
                                    yaxis_title="Temperature (F)",
                                    title="850 hPa (~5,000' ASL)")
                st.plotly_chart(fig_t, width="stretch")

            # Wind direction at 850 hPa
            st.subheader("850 hPa Wind Direction & Speed")
            if not t850.empty:
                fig_w = make_subplots(specs=[[{"secondary_y": True}]])
                for model in t850["model"].unique():
                    m = t850[t850["model"] == model]
                    fig_w.add_trace(go.Scatter(x=m["time"], y=m["wind_dir"],
                                               name=f"{model} Dir", mode="markers",
                                               marker=dict(size=4)),
                                   secondary_y=False)
                    fig_w.add_trace(go.Scatter(x=m["time"],
                                               y=m["wind_speed_kph"] * 0.621371,
                                               name=f"{model} Speed", mode="lines"),
                                   secondary_y=True)
                fig_w.update_yaxes(title_text="Wind Dir (deg)", secondary_y=False)
                fig_w.update_yaxes(title_text="Wind Speed (mph)", secondary_y=True)
                fig_w.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig_w, width="stretch")

        if w850:
            st.caption(f"850 hPa 48h avg wind: {w850.get('mean_dir_deg')} deg "
                       f"@ {w850.get('mean_speed_mph')} mph")

        # ── Wind Rose & Storm Track ──────────────────────────────────
        if not snd_df.empty:
            st.divider()
            st.subheader("850 hPa Wind Rose & Storm Track")
            wr_hours = st.selectbox("Time window", [48, 120], index=0,
                                    format_func=lambda h: f"{h}h")

            wr_col1, wr_col2 = st.columns(2)
            with wr_col1:
                fig_recent = build_wind_rose(snd_df, hours=wr_hours,
                                             mode="recent", title_prefix="Recent: ")
                if fig_recent:
                    st.plotly_chart(fig_recent, width="stretch")
                else:
                    st.info("Not enough recent 850 hPa data.")
            with wr_col2:
                fig_fcst = build_wind_rose(snd_df, hours=wr_hours,
                                           mode="forecast", title_prefix="Forecast: ")
                if fig_fcst:
                    st.plotly_chart(fig_fcst, width="stretch")
                else:
                    st.info("Not enough forecast 850 hPa data.")

            track_text = interpret_storm_track(snd_df, hours=wr_hours)
            st.info(track_text)
            st.caption(
                "Storm track classification for Snoqualmie Pass: "
                "SW-W-NW = Pacific moisture (favorable), "
                "NW-N = post-frontal (colder/drier showers), "
                "E-SE = rain shadow (dry)."
            )
    else:
        st.info("No sounding data. Run `python nowcast.py` to fetch.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: NWS / NBM
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("NWS Gridpoint Forecast & NBM Viewer")
    st.caption("National Weather Service forecast for Snoqualmie Pass (SEW grid 152,54)")

    nws = load_nws_gridpoint()

    if "error" in nws:
        st.warning(f"Could not fetch NWS data: {nws['error']}")
    elif nws:
        # ── Key metrics row ──
        temp_ts = _parse_nws_ts(nws.get("temperature"), c_to_f=True)
        snow_ts = _parse_nws_ts(nws.get("snowfallAmount"), mm_to_in=True)
        sl_ts = _parse_nws_ts(nws.get("snowLevel"), m_to_ft=True)
        pop_ts = _parse_nws_ts(nws.get("probabilityOfPrecipitation"))
        qpf_ts = _parse_nws_ts(nws.get("quantitativePrecipitation"), mm_to_in=True)
        wind_ts = _parse_nws_ts(nws.get("windSpeed"), kph_to_mph=True)
        gust_ts = _parse_nws_ts(nws.get("windGust"), kph_to_mph=True)
        wdir_ts = _parse_nws_ts(nws.get("windDirection"))

        n1, n2, n3, n4 = st.columns(4)
        if not temp_ts.empty:
            n1.metric("Temp Now", f"{temp_ts['value'].iloc[0]:.0f} F")
            n1.metric("7-day Range",
                       f"{temp_ts['value'].min():.0f} - {temp_ts['value'].max():.0f} F")
        if not snow_ts.empty:
            total_snow = snow_ts["value"].sum()
            n2.metric("NWS Total Snow (7d)", f'{total_snow:.1f}"')
        if not sl_ts.empty:
            n3.metric("Snow Level Now", f"{sl_ts['value'].iloc[0]:.0f}' ASL")
            n3.metric("7-day Snow Level Range",
                       f"{sl_ts['value'].min():.0f}' - {sl_ts['value'].max():.0f}'")
        if not qpf_ts.empty:
            n4.metric("Total QPF (7d)", f'{qpf_ts["value"].sum():.2f}"')

        # ── Temperature + Snow Level chart ──
        st.subheader("Temperature & Snow Level Forecast")
        fig_nws1 = make_subplots(specs=[[{"secondary_y": True}]])
        if not temp_ts.empty:
            fig_nws1.add_trace(go.Scatter(
                x=temp_ts["time"], y=temp_ts["value"],
                name="Temperature (F)", mode="lines",
                line=dict(color="#FF7043")), secondary_y=False)
            fig_nws1.add_hline(y=32, line_dash="dash", line_color="cyan",
                               annotation_text="32F")
        if not sl_ts.empty:
            fig_nws1.add_trace(go.Scatter(
                x=sl_ts["time"], y=sl_ts["value"],
                name="Snow Level (ft)", mode="lines",
                line=dict(color="#42A5F5")), secondary_y=True)
            fig_nws1.add_shape(type="line", x0=0, x1=1, xref="paper",
                               y0=3022, y1=3022, yref="y2",
                               line=dict(color="red", dash="dash"))
            fig_nws1.add_annotation(x=0.01, xref="paper", y=3022, yref="y2",
                                    text="Sno Pass 3,022'", showarrow=False,
                                    font=dict(color="red", size=10),
                                    xanchor="left")
        fig_nws1.update_yaxes(title_text="Temperature (F)", secondary_y=False)
        fig_nws1.update_yaxes(title_text="Snow Level (ft ASL)", secondary_y=True)
        fig_nws1.update_layout(template="plotly_dark", height=400,
                               margin=dict(t=30, b=30))
        st.plotly_chart(fig_nws1, width="stretch")

        # ── Snowfall + QPF chart ──
        st.subheader("Snowfall & Precipitation Forecast")
        fig_nws2 = go.Figure()
        if not snow_ts.empty:
            fig_nws2.add_trace(go.Bar(
                x=snow_ts["time"], y=snow_ts["value"],
                name="Snowfall (in)", marker_color="#90CAF9"))
        if not qpf_ts.empty:
            fig_nws2.add_trace(go.Bar(
                x=qpf_ts["time"], y=qpf_ts["value"],
                name="QPF (in liquid)", marker_color="#4CAF50", opacity=0.6))
        fig_nws2.update_layout(template="plotly_dark", height=350, barmode="overlay",
                               yaxis_title="Inches", margin=dict(t=30, b=30))
        st.plotly_chart(fig_nws2, width="stretch")

        # ── Probability of Precip ──
        if not pop_ts.empty:
            st.subheader("Probability of Precipitation")
            fig_pop = go.Figure()
            fig_pop.add_trace(go.Scatter(
                x=pop_ts["time"], y=pop_ts["value"],
                fill="tozeroy", name="PoP %",
                line=dict(color="#66BB6A")))
            fig_pop.update_layout(template="plotly_dark", height=250,
                                  yaxis_title="Probability (%)",
                                  yaxis_range=[0, 100],
                                  margin=dict(t=20, b=30))
            st.plotly_chart(fig_pop, width="stretch")

        # ── Wind ──
        st.subheader("Wind Forecast")
        fig_wind = make_subplots(specs=[[{"secondary_y": True}]])
        if not wind_ts.empty:
            fig_wind.add_trace(go.Scatter(
                x=wind_ts["time"], y=wind_ts["value"],
                name="Sustained (mph)", mode="lines",
                line=dict(color="#FFA726")), secondary_y=False)
        if not gust_ts.empty:
            fig_wind.add_trace(go.Scatter(
                x=gust_ts["time"], y=gust_ts["value"],
                name="Gusts (mph)", mode="lines",
                line=dict(color="#EF5350", dash="dot")), secondary_y=False)
        if not wdir_ts.empty:
            fig_wind.add_trace(go.Scatter(
                x=wdir_ts["time"], y=wdir_ts["value"],
                name="Direction (deg)", mode="markers",
                marker=dict(size=3, color="#78909C")), secondary_y=True)
        fig_wind.update_yaxes(title_text="Speed (mph)", secondary_y=False)
        fig_wind.update_yaxes(title_text="Direction (deg)", secondary_y=True,
                              range=[0, 360])
        fig_wind.update_layout(template="plotly_dark", height=350,
                               margin=dict(t=30, b=30))
        st.plotly_chart(fig_wind, width="stretch")

        # ── Weather narrative from NWS ──
        weather_data = nws.get("weather", {})
        if weather_data and "values" in weather_data:
            st.subheader("NWS Weather Narrative")
            wx_rows = []
            for w in weather_data["values"]:
                t = w["validTime"].split("/")[0]
                descs = []
                for cond in (w.get("value") or []):
                    cov = (cond.get("coverage") or "").replace("_", " ")
                    wx = cond.get("weather", "")
                    intensity = cond.get("intensity") or ""
                    if wx:
                        descs.append(f"{cov} {intensity} {wx}".strip())
                if descs:
                    wx_rows.append({"Time": t, "Conditions": "; ".join(descs)})
            if wx_rows:
                st.dataframe(pd.DataFrame(wx_rows), width="stretch", hide_index=True)

    else:
        st.info("No NWS data available.")

    # ── NBM Viewer embed ──
    st.divider()
    st.subheader("NBM Probabilistic Viewer (NOAA)")
    st.caption("Interactive National Blend of Models forecast for Snoqualmie Pass")
    try:
        components.iframe(NBM_VIEWER_URL, height=700, scrolling=True)
    except Exception:
        st.markdown(f"[Open NBM Viewer in new tab]({NBM_VIEWER_URL})")
    st.markdown(f"[Open NBM Viewer in new tab]({NBM_VIEWER_URL})")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Nowcast (Layer 2)
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Nowcast - Station Telemetry (Layer 2)")
    st.caption("Real-time station data: snowfall pacing (depth-gain), freezing level from "
               "multi-elevation temps, and barometric pressure trends. Blends with Layer 1 "
               "for the current month's forecast.")

    nc = load_nowcast()

    if not nc:
        st.warning("No nowcast data. Run `python nowcast.py` first.")
    else:
        st.caption(f"Last updated: {nc.get('timestamp', '?')}")

        # Pacing
        pace = nc.get("pace", {})
        if "error" not in pace:
            st.subheader(f"Current Month Pacing ({pace.get('station', '?')})")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Days", f"{pace['days_elapsed']} / {pace['days_in_month']}")
            p2.metric("Snowfall to Date", f"{pace['actual_snowfall_in']}\"")
            p3.metric("On Pace For", f"{pace['pace_snowfall_in']}\"")
            p4.metric("SWE Gain Est", f"{pace.get('swe_gain_est_in', '?')}\"")

            d1, d2 = st.columns(2)
            d1.metric("Snow Depth Start", f"{pace.get('depth_start_in', '?')}\"")
            d2.metric("Snow Depth Latest", f"{pace.get('depth_latest_in', '?')}\"")

            # Pace gauge
            pct_done = pace["days_elapsed"] / pace["days_in_month"]
            fig_pace = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pace["actual_snowfall_in"],
                title={"text": "Snowfall This Month (inches)"},
                delta={"reference": pace["pace_snowfall_in"], "relative": False,
                       "suffix": "\" vs pace"},
                gauge={
                    "axis": {"range": [0, max(pace["pace_snowfall_in"] * 1.5, 60)]},
                    "bar": {"color": "#1E88E5"},
                    "steps": [
                        {"range": [0, pace["pace_snowfall_in"]], "color": "#37474F"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": pace["pace_snowfall_in"],
                    },
                },
            ))
            fig_pace.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_pace, width="stretch")
        else:
            st.info(f"Pacing: {pace.get('error', 'No data')}")

        # Freezing level from stations
        fl = nc.get("freezing_level", {})
        if "error" not in fl and fl:
            st.subheader("Freezing Level (Station Temps)")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Current", f"{fl.get('current_ft', '?')}' ASL")
            f2.metric("48h Avg", f"{fl.get('avg_48h_ft', '?')}' ASL")
            f3.metric("Inversions (48h)", f"{fl.get('inversions_48h', '?')} hrs")
            f4.metric("Lapse Rate", f"{fl.get('lapse_rate_avg', '?')} F/1000ft")

        # Pressure
        pres = nc.get("pressure", {})
        if "error" not in pres and pres:
            st.subheader("Pressure Trends (SNO30)")
            pr1, pr2, pr3 = st.columns(3)
            pr1.metric("SLP", f"{pres.get('slp_hpa', '?')} hPa")
            pr2.metric("12h Change", f"{pres.get('change_12h', '?')} hPa",
                       delta=pres.get("pattern", ""))
            pr3.metric("24h Change", f"{pres.get('change_24h', '?')} hPa")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Model Performance")
    st.caption("Leave-one-year-out cross-validation scores, feature importance rankings, "
               "and recent forecast verification against observed SNOTEL data.")

    cv = load_cv_scores()
    if not cv.empty:
        st.subheader("Cross-Validation Scores (SWE)")
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=cv["model"], y=cv["r2_mean"],
            error_y=dict(type="data", array=cv["r2_std"]),
            marker_color="#1E88E5", name="R^2",
        ))
        fig_cv.update_layout(template="plotly_dark", height=350,
                             yaxis_title="R^2 (CV mean +/- std)",
                             title="Leave-One-Year-Out CV: SWE R^2")
        st.plotly_chart(fig_cv, width="stretch")

    # Feature importance
    imp_target = st.selectbox("Feature importance target", ["wteq", "snow"], index=0)
    imp = load_feature_importance(imp_target)
    if not imp.empty:
        st.subheader(f"Top 20 Features ({imp_target.upper()})")
        top = imp.head(20)
        fig_imp = px.bar(top, x="combined", y="feature", orientation="h",
                         color="combined", color_continuous_scale="Blues")
        fig_imp.update_layout(template="plotly_dark", height=500,
                              yaxis=dict(autorange="reversed"),
                              xaxis_title="Combined Importance")
        st.plotly_chart(fig_imp, width="stretch")

    # Forecast vs actual
    fva = load_forecast_vs_actual()
    if not fva.empty:
        st.subheader("Recent Forecast vs Actual")
        fva_display = fva.copy()
        fva_display["label"] = fva_display["month_name"] + " " + fva_display["year"].astype(str)
        fig_fva = go.Figure()
        fig_fva.add_trace(go.Bar(x=fva_display["label"], y=fva_display["actual_wteq"],
                                  name="Actual SWE", marker_color="#43A047"))
        fig_fva.add_trace(go.Bar(x=fva_display["label"], y=fva_display["pred_wteq"],
                                  name="Predicted SWE", marker_color="#1E88E5"))
        fig_fva.update_layout(barmode="group", template="plotly_dark", height=350,
                              yaxis_title="SWE (inches)")
        st.plotly_chart(fig_fva, width="stretch")

        st.dataframe(fva, width="stretch", hide_index=True)

    # ── Backtest Results ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Leave-One-Year-Out Backtest")
    st.caption("Backtest skill = 1 - RMSE/RMSE_clim. Positive = model beats climatology "
               "(predicting the historical mean). Negative = model is worse than climatology.")

    bt_path = os.path.join(DATA, "backtest_metrics.json")
    tune_path = os.path.join(DATA, "tune_backtest_results.csv")

    if os.path.exists(bt_path):
        with open(bt_path, "r") as f:
            bt_metrics = json.load(f)

        bt_rows = []
        for config_name, targets in bt_metrics.items():
            wteq = targets.get("WTEQ", {})
            bt_rows.append({
                "Config": config_name,
                "Skill": wteq.get("skill"),
                "RMSE": wteq.get("rmse"),
                "RMSE Clim": wteq.get("rmse_clim"),
                "Correlation": wteq.get("correlation"),
                "Bias": wteq.get("bias"),
                "N": wteq.get("n_points"),
            })
        bt_df = pd.DataFrame(bt_rows)

        # Skill bar chart
        fig_bt = go.Figure()
        colors = ["#F44336" if s < 0 else "#4CAF50" for s in bt_df["Skill"]]
        fig_bt.add_trace(go.Bar(
            x=bt_df["Config"], y=bt_df["Skill"],
            marker_color=colors,
            text=[f"{s:.1%}" for s in bt_df["Skill"]],
            textposition="outside",
        ))
        fig_bt.add_hline(y=0, line_dash="dash", line_color="white",
                         annotation_text="Climatology baseline")
        fig_bt.update_layout(template="plotly_dark", height=350,
                             yaxis_title="Skill Score",
                             yaxis_tickformat=".0%",
                             margin=dict(t=30, b=30))
        st.plotly_chart(fig_bt, width="stretch")

        # Metrics table
        st.dataframe(bt_df.style.format({
            "Skill": "{:.1%}", "RMSE": "{:.2f}", "RMSE Clim": "{:.2f}",
            "Correlation": "{:.3f}", "Bias": "{:.2f}",
        }), width="stretch", hide_index=True)

        # Interpretation
        best_skill = bt_df["Skill"].max()
        best_corr = bt_df["Correlation"].max()
        if best_skill < 0:
            st.info(
                f"All configs show negative skill (best: {best_skill:.1%}), meaning the model's "
                f"RMSE exceeds climatology. However, correlation is strong ({best_corr:.3f}), "
                f"indicating the model captures relative patterns. This is a common overfitting "
                f"signature with {bt_df['N'].iloc[0]} training points vs ~104 features. "
                f"Layer 2 station telemetry provides the operational near-term correction."
            )
        else:
            st.success(f"Best config beats climatology with skill = {best_skill:.1%}.")

    # Tune results (if available)
    if os.path.exists(tune_path):
        tune_df = pd.read_csv(tune_path)
        if not tune_df.empty:
            st.subheader("Tune Grid Search Results")
            top_n = min(15, len(tune_df))
            top = tune_df.nlargest(top_n, "skill")
            st.caption(f"Top {top_n} of {len(tune_df)} configs tested (ranked by skill)")
            st.dataframe(
                top[["label", "skill", "rmse", "rmse_clim", "corr", "bias"]].style.format({
                    "skill": "{:.1%}", "rmse": "{:.2f}", "rmse_clim": "{:.2f}",
                    "corr": "{:.3f}", "bias": "{:.2f}",
                }),
                width="stretch", hide_index=True,
            )

    if not os.path.exists(bt_path) and not os.path.exists(tune_path):
        st.info("No backtest results found. Run `python backtest.py` or `python tune_backtest.py`.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: Analog Years
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Analog Years")
    st.caption(f"Winters with the most similar teleconnection patterns to WY{wy % 100:02d}. "
               "Historical SWE and snowfall tracks provide context for how the season may evolve.")

    analogs, detail = load_analogs()

    if not analogs.empty:
        st.subheader("Closest Analog Winters")
        st.dataframe(analogs, width="stretch", hide_index=True)

    if not detail.empty:
        st.subheader("Analog Year SWE Tracks")
        fig_a = px.line(detail[detail["WTEQ"].notna()],
                        x="month_name", y="WTEQ",
                        color="analog_year",
                        markers=True,
                        labels={"WTEQ": "SWE (inches)", "month_name": "Month"},
                        category_orders={"month_name": ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]})

        # Add forecast line
        fc = load_forecast()
        if not fc.empty:
            fig_a.add_trace(go.Scatter(
                x=fc["month_name"], y=fc["wteq_ensemble"],
                name=f"WY{wy % 100:02d} Forecast", mode="lines+markers",
                line=dict(color="white", width=3, dash="dash"),
                marker=dict(size=10, symbol="star"),
            ))
        fig_a.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_a, width="stretch")

        # Snowfall comparison
        st.subheader("Analog Year Snowfall")
        snow_detail = detail[detail["snow_inches"].notna()].copy()
        if not snow_detail.empty:
            fig_s = px.line(snow_detail, x="month_name", y="snow_inches",
                            color="analog_year", markers=True,
                            labels={"snow_inches": "Snowfall (inches)"},
                            category_orders={"month_name": ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]})
            if not fc.empty:
                fig_s.add_trace(go.Scatter(
                    x=fc["month_name"], y=fc["snow_ensemble"],
                    name=f"WY{wy % 100:02d} Forecast", mode="lines+markers",
                    line=dict(color="white", width=3, dash="dash"),
                    marker=dict(size=10, symbol="star"),
                ))
            fig_s.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_s, width="stretch")

    # Static plot from forecast.py
    analog_plot = os.path.join(PLOTS, "analog_years.png")
    if os.path.exists(analog_plot):
        with st.expander("Static Analog Plot"):
            st.image(analog_plot, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: Teleconnection State
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("Current Teleconnection State")
    st.caption("Large-scale ocean-atmosphere circulation indices that drive Snoqualmie Pass snowpack. "
               "These upstream signals are the predictors in the Layer 1 ML models.")

    bl = load_bottom_line()
    ctx = bl.get("context", {})
    tele = ctx.get("teleconnection", {})
    trend = ctx.get("teleconnection_trend", {})

    if tele:
        st.subheader("Current Index Values")
        # Display as bar chart
        tele_df = pd.DataFrame([
            {"Index": k, "Value": v, "Category": "ENSO" if k in ["roni", "enso34", "nino4_anom"] else
             "Circulation" if k in ["ao", "pna", "nao", "epo", "pdo"] else "Other"}
            for k, v in tele.items()
        ])
        fig_tele = px.bar(tele_df, x="Index", y="Value", color="Category",
                          color_discrete_map={"ENSO": "#E53935", "Circulation": "#1E88E5",
                                              "Other": "#43A047"})
        fig_tele.add_hline(y=0, line_color="gray")
        fig_tele.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_tele, width="stretch")

        # Trends
        if trend:
            st.subheader("1-Month Trends")
            t1, t2, t3, t4, t5 = st.columns(5)
            cols = [t1, t2, t3, t4, t5]
            for i, (k, v) in enumerate(trend.items()):
                if i < 5:
                    cols[i].metric(k.upper(), f"{tele.get(k, '?'):.2f}",
                                   delta=f"{v:+.2f}")

        # MJO note
        mjo = ctx.get("mjo_note", "")
        if mjo:
            st.info(mjo)

    # Static teleconnection plot
    tele_plot = os.path.join(PLOTS, "current_telecon_state.png")
    if os.path.exists(tele_plot):
        with st.expander("Static Teleconnection State Plot"):
            st.image(tele_plot, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("Predictor Correlations")

    corr_plot = os.path.join(PLOTS, "correlation_heatmap.png")
    if os.path.exists(corr_plot):
        st.image(corr_plot, width="stretch")
    else:
        st.info("No correlation heatmap. Run `python forecast.py` to generate.")

    # Feature importance plot
    fi_plot = os.path.join(PLOTS, "feature_importance.png")
    if os.path.exists(fi_plot):
        with st.expander("Feature Importance Plot"):
            st.image(fi_plot, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: Data Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("Data Explorer")

    data_choice = st.selectbox("Select dataset", [
        "Forecast Results",
        "Sounding Forecast",
        "Nowcast JSON",
        "Analog Detail",
        "Feature Importance (SWE)",
        "Feature Importance (Snow)",
        "CV Scores",
        "Forecast vs Actual",
    ])

    if data_choice == "Forecast Results":
        fc = load_forecast()
        if not fc.empty:
            st.dataframe(fc, width="stretch", hide_index=True)
    elif data_choice == "Sounding Forecast":
        snd = load_sounding()
        if not snd.empty:
            model_filter = st.multiselect("Model", snd["model"].unique().tolist(),
                                          default=snd["model"].unique().tolist())
            level_filter = st.multiselect("Level (hPa)", sorted(snd["level_hPa"].unique().tolist()),
                                          default=sorted(snd["level_hPa"].unique().tolist()))
            filtered = snd[(snd["model"].isin(model_filter)) & (snd["level_hPa"].isin(level_filter))]
            st.dataframe(filtered, width="stretch", hide_index=True)
            st.caption(f"{len(filtered)} rows")
    elif data_choice == "Nowcast JSON":
        nc = load_nowcast()
        st.json(nc)
    elif data_choice == "Analog Detail":
        _, detail = load_analogs()
        if not detail.empty:
            st.dataframe(detail, width="stretch", hide_index=True)
    elif data_choice == "Feature Importance (SWE)":
        imp = load_feature_importance("wteq")
        if not imp.empty:
            st.dataframe(imp, width="stretch", hide_index=True)
    elif data_choice == "Feature Importance (Snow)":
        imp = load_feature_importance("snow")
        if not imp.empty:
            st.dataframe(imp, width="stretch", hide_index=True)
    elif data_choice == "CV Scores":
        cv = load_cv_scores()
        if not cv.empty:
            st.dataframe(cv, width="stretch", hide_index=True)
    elif data_choice == "Forecast vs Actual":
        fva = load_forecast_vs_actual()
        if not fva.empty:
            st.dataframe(fva, width="stretch", hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Snoqualmie Pass Snowpack Forecast | Layer 1: Teleconnection ML | Layer 2: Station Telemetry + Open-Meteo Soundings")
