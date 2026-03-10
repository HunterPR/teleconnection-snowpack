"""
dashboard.py -- Snoqualmie Pass Snowpack Forecast Dashboard
==========================================================
Streamlit dashboard for the two-layer forecasting system.

Navigation (sidebar):
  Overview   -> Forecast Overview (includes Nowcast / Station Telemetry)
  Weather    -> NWS / NBM | Sounding & Freezing Level | NWAC Telemetry | Inversion & Cross-Section
  Model/Data -> Model Performance | Analog Years | Teleconnection State | Correlation | Data Explorer

Sidebar also contains Pipeline Controls (run scripts) and a data chatbot.
"""

import os
import sys
import json
import subprocess
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

    if mode == "forecast":
        now_utc = pd.Timestamp.now(tz="UTC")
        t850 = t850[t850["time"] >= now_utc]
        cutoff = now_utc + pd.Timedelta(hours=hours)
        t850 = t850[t850["time"] <= cutoff]
    else:
        cutoff = t850["time"].max() - pd.Timedelta(hours=hours)
        t850 = t850[t850["time"] >= cutoff]

    if len(t850) < 3:
        return None

    t850["compass"] = t850["wind_dir"].apply(_compass)
    t850["speed_mph"] = t850["wind_speed_kph"] * 0.621371

    speed_bins = [(0, 5, "0-5"), (5, 15, "5-15"), (15, 25, "15-25"),
                  (25, 35, "25-35"), (35, 200, "35+")]
    colors = ["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]

    fig = go.Figure()
    for (lo, hi, label), color in zip(speed_bins, colors):
        mask = (t850["speed_mph"] >= lo) & (t850["speed_mph"] < hi)
        sub = t850[mask]
        counts = sub.groupby("compass").size().reindex(_COMPASS_LABELS, fill_value=0)
        fig.add_trace(go.Barpolar(
            r=counts.values, theta=_COMPASS_LABELS,
            name=f"{label} mph", marker_color=color, opacity=0.8,
        ))

    fig.update_layout(
        template="plotly_dark", height=350,
        title=f"{title_prefix}850 hPa Wind Rose ({hours}h)",
        polar=dict(radialaxis=dict(showticklabels=True, ticksuffix="h")),
        legend=dict(x=1.1, y=1),
        margin=dict(t=60, b=30, l=30, r=30),
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
    rads = np.deg2rad(t850["wind_dir"].dropna())
    mean_dir = np.rad2deg(np.arctan2(np.sin(rads).mean(), np.cos(rads).mean())) % 360
    compass = _compass(mean_dir)

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


# -- Page config --------------------------------------------------------------

st.set_page_config(
    page_title="Snoqualmie Pass Snowpack Forecast",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Data loading helpers -----------------------------------------------------

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

@st.cache_data(ttl=300)
def load_nwac_current():
    p = os.path.join(DATA, "nwac_current.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


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


# -- Natural-language summary generators --------------------------------------

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
    total_swe = fc_df["wteq_ensemble"].sum()
    total_snow = fc_df["snow_ensemble"].sum()
    parts.append(
        f"Teleconnection ML ensemble projects {total_swe:.1f}\" SWE and "
        f"{total_snow:.1f}\" snowfall across the forecast window."
    )
    remaining = fc_df[fc_df["month"] >= current_month]
    if not remaining.empty:
        rem_swe = remaining["wteq_ensemble"].sum()
        rem_snow = remaining["snow_ensemble"].sum()
        months_left = remaining["month_name"].tolist()
        parts.append(
            f"Remaining ({', '.join(months_left)}): {rem_swe:.1f}\" SWE, "
            f"{rem_snow:.1f}\" snowfall projected."
        )
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


# -- Pipeline runner ----------------------------------------------------------

def _run_pipeline_script(script_name: str):
    """Run a pipeline script via subprocess and show output in Streamlit."""
    cmd = [sys.executable, os.path.join(BASE, script_name)]
    with st.spinner(f"Running {script_name}..."):
        try:
            result = subprocess.run(
                cmd, cwd=BASE, capture_output=True, text=True,
                timeout=600, encoding="utf-8", errors="replace",
            )
            if result.returncode == 0:
                st.success(f"{script_name} completed successfully.")
                if result.stdout:
                    with st.expander("Output", expanded=False):
                        st.code(result.stdout[-3000:], language="text")
                st.cache_data.clear()
            else:
                st.error(f"{script_name} failed (exit code {result.returncode}).")
                if result.stderr:
                    st.code(result.stderr[-2000:], language="text")
        except subprocess.TimeoutExpired:
            st.error(f"{script_name} timed out after 10 minutes.")
        except Exception as e:
            st.error(f"Error running {script_name}: {e}")


# -- Chatbot ------------------------------------------------------------------

@st.cache_data(ttl=300)
def _build_chat_context() -> str:
    """Build a context string from current data for the LLM."""
    parts = []
    bl = load_bottom_line()
    if bl:
        parts.append(f"Bottom line: {bl.get('bottom_line', '')}")
        ctx = bl.get("context", {})
        tele = ctx.get("teleconnection", {})
        if tele:
            parts.append(f"Teleconnection state: {json.dumps(tele)}")
    fc = load_forecast()
    if not fc.empty:
        parts.append(f"Forecast:\n{fc.to_string(index=False)}")
    nc = load_nowcast()
    if nc:
        # Trim to key fields
        for key in ["pace", "freezing_level", "pressure"]:
            if key in nc:
                parts.append(f"Nowcast {key}: {json.dumps(nc[key], default=str)}")
        snd = nc.get("sounding", {})
        if snd:
            for k in ["freezing_level_forecast", "snow_level_ft",
                       "snowfall_possible_hours", "wind_850hPa_48h"]:
                if k in snd:
                    parts.append(f"Sounding {k}: {json.dumps(snd[k])}")
    analogs, _ = load_analogs()
    if not analogs.empty:
        parts.append(f"Analog years:\n{analogs.to_string(index=False)}")
    return "\n\n".join(parts)[:6000]


def _chat_respond(user_question: str) -> str:
    """Generate a chat response using available LLM, or return a fallback."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    context = _build_chat_context()
    system_prompt = (
        "You are a knowledgeable assistant for a Snoqualmie Pass snowpack forecasting tool. "
        "Answer questions about the current forecast, teleconnection indices, analog years, "
        "sounding data, and model performance. Be concise and specific with numbers. "
        "Use the provided context data to inform your answers."
    )
    full_prompt = f"Context data:\n{context}\n\nUser question: {user_question}"

    # Try OpenAI first
    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=600,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"LLM error: {e}"

    # Try Anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
                max_tokens=600,
                system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return (msg.content[0].text if msg.content else "").strip()
        except Exception as e:
            return f"LLM error: {e}"

    return (
        "No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "in your .env file to enable the chat assistant."
    )


# ==============================================================================
# RENDER FUNCTIONS -- one per sub-tab
# ==============================================================================

def render_forecast_overview():
    now = datetime.now()
    current_month = now.month
    wy = now.year if now.month >= 10 else now.year
    st.header(f"Snoqualmie Pass Snowpack Forecast - WY{wy % 100:02d}")
    st.caption(f"Water Year {wy-1}-{wy} (Oct {wy-1} - Sep {wy})  |  "
               f"Updated {now.strftime('%b %d, %Y %I:%M %p')}")

    fc = load_forecast()
    bl = load_bottom_line()
    fva = load_forecast_vs_actual()

    SNOTEL_URL = "https://wcc.sc.egov.usda.gov/nwcc/site?sitenum=908"
    SNOTEL_STATION = "SNOTEL #908 (Snoqualmie Pass)"

    # -- Forecast Horizons -------------------------------------------------
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

    # -- Bottom line synopsis ----------------------------------------------
    if bl:
        st.info(bl.get("bottom_line", ""))
        human = bl.get("human_notes", "")
        if human:
            st.caption(f"Forecaster notes: {human}")

    # -- Monthly forecast cards (auto-detect past/current/future) ----------
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
                mo_year = wy if mo_num < 10 else wy - 1
                swe_fc = row.get("wteq_ensemble", 0)
                snow_fc = row.get("snow_ensemble", 0)
                swe_hist = row.get("wteq_hist_mean", 0)
                snow_hist = row.get("snow_hist_mean", 0)
                swe_pct = row.get("wteq_pct", 50)
                snow_pct = row.get("snow_pct", 50)

                is_past = (mo_num < current_month)
                is_current = (mo_num == current_month)

                if is_past:
                    st.subheader(f"{mo_name} {mo_year}  :white_check_mark:")
                    st.caption("VERIFIED")
                elif is_current:
                    st.subheader(f"{mo_name} {mo_year}  :hourglass_flowing_sand:")
                    st.caption(f"IN PROGRESS (day {now.day})")
                else:
                    st.subheader(f"{mo_name} {mo_year}")
                    st.caption("FORECAST")

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

                else:
                    st.metric("SWE (inches)", f'{swe_fc:.1f}"',
                              delta=f"{swe_pct:.0f}th pct (avg {swe_hist:.1f}\")")
                    st.metric("Snowfall (inches)", f'{snow_fc:.1f}"',
                              delta=f"{snow_pct:.0f}th pct (avg {snow_hist:.1f}\")")
                    st.caption("Layer 1 teleconnection forecast")

        # -- Forecast vs Actual bar chart ----------------------------------
        st.subheader("Monthly: Forecast vs Historical vs Actual")

        months = fc["month_name"].tolist()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["SWE (inches)", "Snowfall (inches)"])

        fig.add_trace(go.Bar(x=months, y=fc["wteq_ensemble"], name="Forecast SWE",
                             marker_color="#1E88E5"), row=1, col=1)
        fig.add_trace(go.Bar(x=months, y=fc["wteq_hist_mean"],
                             name="Historical Mean SWE",
                             marker_color="#546E7A", opacity=0.6), row=1, col=1)
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

        fig.add_trace(go.Bar(x=months, y=fc["snow_ensemble"], name="Forecast Snow",
                             marker_color="#43A047"), row=1, col=2)
        fig.add_trace(go.Bar(x=months, y=fc["snow_hist_mean"],
                             name="Historical Mean Snow",
                             marker_color="#546E7A", opacity=0.6), row=1, col=2)
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

        # -- WY Season Tracking --------------------------------------------
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

        # -- Model spread -------------------------------------------------
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

    # -- Station Telemetry (Layer 2) -- merged from former Nowcast tab ------
    st.divider()
    with st.expander("Station Telemetry (Layer 2)", expanded=True):
        nc = load_nowcast()
        if not nc:
            st.warning("No nowcast data. Run `python nowcast.py` first.")
        else:
            st.caption(f"Last updated: {nc.get('timestamp', '?')}")

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

            fl = nc.get("freezing_level", {})
            if "error" not in fl and fl:
                st.subheader("Freezing Level (Station Temps)")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Current", f"{fl.get('current_ft', '?')}' ASL")
                f2.metric("48h Avg", f"{fl.get('avg_48h_ft', '?')}' ASL")
                f3.metric("Inversions (48h)", f"{fl.get('inversions_48h', '?')} hrs")
                f4.metric("Lapse Rate", f"{fl.get('lapse_rate_avg', '?')} F/1000ft")

            pres = nc.get("pressure", {})
            if "error" not in pres and pres:
                st.subheader("Pressure Trends (SNO30)")
                pr1, pr2, pr3 = st.columns(3)
                pr1.metric("SLP", f"{pres.get('slp_hpa', '?')} hPa")
                pr2.metric("12h Change", f"{pres.get('change_12h', '?')} hPa",
                           delta=pres.get("pattern", ""))
                pr3.metric("24h Change", f"{pres.get('change_24h', '?')} hPa")


def render_nws_nbm():
    st.header("NWS Gridpoint Forecast & NBM Viewer")
    st.caption("National Weather Service forecast for Snoqualmie Pass (SEW grid 152,54)")

    nws = load_nws_gridpoint()

    if "error" in nws:
        st.warning(f"Could not fetch NWS data: {nws['error']}")
    elif nws:
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

    st.divider()
    st.subheader("NBM Probabilistic Viewer (NOAA)")
    st.caption("Interactive National Blend of Models forecast for Snoqualmie Pass")
    try:
        components.iframe(NBM_VIEWER_URL, height=700, scrolling=True)
    except Exception:
        pass
    st.markdown(f"[Open NBM Viewer in new tab]({NBM_VIEWER_URL})")


def render_sounding():
    st.header("Forecast Sounding & Freezing Level")
    st.caption("Open-Meteo ECMWF + GFS pressure-level data for Snoqualmie Pass. "
               "Shows vertical atmosphere structure, freezing/snow levels, and 850 hPa storm flow.")

    nc = load_nowcast()
    snd_data = nc.get("sounding", {})
    snd_df = load_sounding()

    if "error" in snd_data:
        st.warning(f"Sounding error: {snd_data['error']}")
    elif snd_data:
        fzl = snd_data.get("freezing_level_forecast", {})
        sl = snd_data.get("snow_level_ft")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Freezing Level Now", f"{fzl.get('current_ft', '?')}' ASL")
        c2.metric("48h Min", f"{fzl.get('min_48h_ft', '?')}'")
        c3.metric("48h Max", f"{fzl.get('max_48h_ft', '?')}'")
        if sl is not None:
            c4.metric("Snow Level (WB=32F)", f"{sl}' ASL")

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

        if not snd_df.empty:
            st.divider()
            st.subheader("850 hPa Wind Rose & Storm Track")
            wr_hours = st.selectbox("Time window", [48, 120], index=0,
                                    format_func=lambda h: f"{h}h",
                                    key="snd_wr_hours")

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


def render_nwac_telemetry():
    st.header("NWAC Station Telemetry")
    st.caption(
        "Current conditions from Northwest Avalanche Center stations "
        "near Snoqualmie Pass. Data from api.nwac.us."
    )

    nwac = load_nwac_current()
    stations = nwac.get("stations", [])
    fetched = nwac.get("fetched_utc", "unknown")

    if not stations:
        st.warning(
            "No NWAC station data available. The NWAC API may be restricted or "
            "data has not been fetched yet. Use the Pipeline Controls in the sidebar "
            "to run Fetch Predictors, or run `python fetch_new_predictors.py`."
        )
        st.caption(f"Last fetch attempt: {fetched}")

        st.info(
            "Fallback stations monitored: Snoqualmie Pass (3,000'), Alpental (2,900'), "
            "Crystal Mountain (4,400'), Stevens Pass (4,061'). "
            "Check nwac.us for current conditions."
        )
        st.markdown("[NWAC Current Conditions](https://nwac.us/avalanche-forecast/)")

        # Show WSDOT pass conditions as alternative
        wsdot_path = os.path.join(DATA, "wsdot_passes.json")
        if os.path.exists(wsdot_path):
            try:
                with open(wsdot_path) as f:
                    wsdot = json.load(f)
                passes = wsdot if isinstance(wsdot, list) else wsdot.get("passes", [])
                if passes:
                    st.divider()
                    st.subheader("WSDOT Mountain Pass Conditions")
                    for p in passes:
                        name = p.get("name") or p.get("PassName", "?")
                        elev = p.get("elevation_ft") or p.get("ElevationInFeet", "?")
                        wx = p.get("weather_desc") or p.get("WeatherCondition", "")
                        road = p.get("road_condition") or p.get("RoadCondition", "")
                        st.markdown(f"**{name}** ({elev}') -- {wx} | Road: {road}")
            except Exception:
                pass
        return

    st.caption(f"Last fetched: {fetched}")

    # Metrics row
    n_cols = min(4, len(stations))
    cols = st.columns(n_cols)
    for i, stn in enumerate(stations[:n_cols]):
        with cols[i]:
            st.subheader(stn.get("station_name", "Unknown"))
            st.caption(f"Elev: {stn.get('elevation_ft', '?')}' ASL")
            if stn.get("air_temp_f") is not None:
                st.metric("Air Temp", f"{stn['air_temp_f']:.0f} F")
            if stn.get("snow_depth_in") is not None:
                st.metric("Snow Depth", f'{stn["snow_depth_in"]:.0f}"')
            if stn.get("wind_speed_mph") is not None:
                st.metric("Wind", f"{stn['wind_speed_mph']:.0f} mph")
            if stn.get("precip_24h_in") is not None:
                st.metric("24h Precip", f'{stn["precip_24h_in"]:.2f}"')
            st.caption(f"Obs: {stn.get('obs_time', '?')}")

    if len(stations) > 4:
        st.subheader("All Stations")
        st.dataframe(pd.DataFrame(stations), width="stretch", hide_index=True)

    # Temperature vs elevation plot
    temps_with_elev = [
        s for s in stations
        if s.get("air_temp_f") is not None and s.get("elevation_ft") is not None
    ]
    if len(temps_with_elev) >= 2:
        st.subheader("Temperature vs Elevation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[s["air_temp_f"] for s in temps_with_elev],
            y=[s["elevation_ft"] for s in temps_with_elev],
            text=[s.get("station_name", "") for s in temps_with_elev],
            mode="markers+text",
            textposition="middle right",
            marker=dict(size=10, color="#1E88E5"),
        ))
        fig.add_vline(x=32, line_dash="dash", line_color="cyan",
                      annotation_text="32F")
        fig.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Temperature (F)",
            yaxis_title="Elevation (ft ASL)",
        )
        st.plotly_chart(fig, width="stretch")


def render_inversion():
    st.header("Inversion & Cross-Section Analysis")
    st.caption(
        "Vertical atmosphere structure analysis for Snoqualmie Pass. "
        "Identifies temperature inversions (warm layers aloft) that affect "
        "precipitation type and avalanche conditions."
    )

    snd_df = load_sounding()
    nc = load_nowcast()

    # -- Time-Height Hovmoller ---------------------------------------------
    if not snd_df.empty:
        st.subheader("Time-Height Diagram (Hovmoller)")
        st.caption(
            "Temperature or wet-bulb at pressure levels over the forecast period. "
            "Warm layers aloft (red above blue) indicate inversions. "
            "Horizontal axis = time, vertical axis = altitude/pressure."
        )

        hov_col1, hov_col2 = st.columns(2)
        with hov_col1:
            models_avail = snd_df["model"].unique().tolist()
            hov_model = st.selectbox("Model", models_avail, index=0, key="hov_model")
        with hov_col2:
            hov_var = st.selectbox("Variable",
                                   ["temp_f", "wetbulb_f", "rh"],
                                   index=0, key="hov_var",
                                   format_func=lambda v: {
                                       "temp_f": "Temperature (F)",
                                       "wetbulb_f": "Wet-Bulb (F)",
                                       "rh": "Relative Humidity (%)",
                                   }.get(v, v))

        ALT_MAP = {1000: 300, 925: 2600, 850: 4900, 700: 9800, 500: 18400}

        m_data = snd_df[snd_df["model"] == hov_model].copy()
        if not m_data.empty and hov_var in m_data.columns:
            pivot = m_data.pivot_table(index="level_hPa", columns="time",
                                       values=hov_var, aggfunc="mean")
            pivot = pivot.sort_index(ascending=False)  # 500 on top

            y_labels = [f"{lev} hPa (~{ALT_MAP.get(lev, lev):,}' ASL)"
                       for lev in pivot.index]

            colorscale = "RdBu_r" if hov_var in ("temp_f", "wetbulb_f") else "Blues"

            fig_hov = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=y_labels,
                colorscale=colorscale,
                colorbar_title=hov_var.replace("_", " ").title(),
            ))

            # Mark the 32F line for temp/wetbulb
            if hov_var in ("temp_f", "wetbulb_f"):
                # Find which levels are near 32F and add annotation
                mean_by_level = pivot.mean(axis=1)
                for i, (lev, avg) in enumerate(mean_by_level.items()):
                    if abs(avg - 32) < 10:
                        fig_hov.add_annotation(
                            x=pivot.columns[0], y=y_labels[i],
                            text="~32F zone", showarrow=False,
                            font=dict(color="white", size=9),
                            xanchor="left",
                        )

            fig_hov.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_title="Time (UTC)",
                yaxis_title="Level",
                title=f"{hov_var.replace('_', ' ').title()} -- {hov_model}",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_hov, width="stretch")

            # Inversion detection from Hovmoller
            st.caption(
                "How to read: In a normal atmosphere, temperature decreases with altitude "
                "(blue on top, warm on bottom). An inversion appears when a warm layer "
                "(red/orange) sits above a cooler layer -- this traps moisture and can "
                "create freezing rain or a rain/snow transition at the pass."
            )
        else:
            st.info("Sounding data missing the selected variable.")

    st.divider()

    # -- I-90 Corridor Vertical Profile ------------------------------------
    st.subheader("I-90 Corridor Vertical Profile")
    st.caption(
        "Current vertical atmosphere profile from sounding data, with station "
        "elevation markers along the I-90 corridor."
    )

    snd = nc.get("sounding", {}) if nc else {}
    vp = snd.get("vertical_profile", [])

    fig_xsec = go.Figure()

    if vp:
        altitudes = [p.get("altitude_ft", 0) for p in vp]
        temps = [p.get("temp_f", 0) for p in vp]
        wetbulbs = [p.get("wetbulb_f", 0) for p in vp]
        dewpoints = [p.get("dewpoint_f", 0) for p in vp]

        fig_xsec.add_trace(go.Scatter(
            x=temps, y=altitudes,
            mode="lines+markers", name="Temperature (F)",
            line=dict(color="#FF7043", width=3),
            marker=dict(size=10),
        ))
        fig_xsec.add_trace(go.Scatter(
            x=wetbulbs, y=altitudes,
            mode="lines+markers", name="Wet-Bulb (F)",
            line=dict(color="#42A5F5", width=2, dash="dot"),
            marker=dict(size=8),
        ))
        fig_xsec.add_trace(go.Scatter(
            x=dewpoints, y=altitudes,
            mode="lines+markers", name="Dewpoint (F)",
            line=dict(color="#66BB6A", width=2, dash="dash"),
            marker=dict(size=6),
        ))

        # Detect inversions in the profile
        for i in range(len(altitudes) - 1):
            if temps[i+1] > temps[i] and altitudes[i+1] > altitudes[i]:
                fig_xsec.add_annotation(
                    x=temps[i+1], y=altitudes[i+1],
                    text="INVERSION",
                    font=dict(color="yellow", size=11),
                    showarrow=True, arrowcolor="yellow",
                )

    # Station elevation markers
    station_elevs = {
        "SNO30 (Snoqualmie Pass)": 3010,
        "ALP31 (Alpental Base)": 3100,
        "SNO38 (Dodge Ridge)": 3760,
        "ALP44 (Alpental Mid)": 4350,
        "ALP55 (Alpental Summit)": 5400,
    }
    for name, elev in station_elevs.items():
        fig_xsec.add_hline(y=elev, line_dash="dot", line_color="gray",
                           annotation_text=name, annotation_position="bottom right",
                           annotation_font_size=9)

    # Pass elevation
    fig_xsec.add_hline(y=3022, line_dash="dash", line_color="red",
                       annotation_text="Snoqualmie Pass (3,022')")

    # 32F reference
    fig_xsec.add_vline(x=32, line_dash="dash", line_color="cyan",
                       annotation_text="32F")

    fig_xsec.update_layout(
        template="plotly_dark",
        height=550,
        xaxis_title="Temperature (F)",
        yaxis_title="Altitude (ft ASL)",
        title="Vertical Profile: I-90 Corridor",
        yaxis_range=[0, 20000],
    )
    st.plotly_chart(fig_xsec, width="stretch")

    # Inversion summary
    fl = nc.get("freezing_level", {}) if nc else {}
    if fl and "error" not in fl:
        inv_count = fl.get("inversions_48h", 0)
        lapse = fl.get("lapse_rate_avg")
        if inv_count and int(inv_count) > 0:
            st.warning(
                f"Inversion detected: {inv_count} hours of inverted lapse rate "
                f"in the last 48h. Average lapse rate: {lapse} F/1000ft "
                f"(normal is about -3.5 F/1000ft). Inversions can cause freezing rain "
                f"or a rain/snow line well below the freezing level."
            )
        elif lapse is not None:
            st.info(
                f"No inversions detected in the last 48h. "
                f"Average lapse rate: {lapse} F/1000ft."
            )
    elif not vp:
        st.info("No sounding data available. Run `python nowcast.py` to fetch.")


def render_model_performance():
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

    imp_target = st.selectbox("Feature importance target", ["wteq", "snow"], index=0,
                               key="imp_target")
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

    # -- Backtest Results --------------------------------------------------
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

        st.dataframe(bt_df.style.format({
            "Skill": "{:.1%}", "RMSE": "{:.2f}", "RMSE Clim": "{:.2f}",
            "Correlation": "{:.3f}", "Bias": "{:.2f}",
        }), width="stretch", hide_index=True)

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


def render_analog_years():
    now = datetime.now()
    wy = now.year if now.month >= 10 else now.year

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

    analog_plot = os.path.join(PLOTS, "analog_years.png")
    if os.path.exists(analog_plot):
        with st.expander("Static Analog Plot"):
            st.image(analog_plot, width="stretch")


def render_teleconnection():
    now = datetime.now()
    wy = now.year if now.month >= 10 else now.year

    st.header("Current Teleconnection State")
    st.caption("Large-scale ocean-atmosphere circulation indices that drive Snoqualmie Pass snowpack. "
               "These upstream signals are the predictors in the Layer 1 ML models.")

    bl = load_bottom_line()
    ctx = bl.get("context", {})
    tele = ctx.get("teleconnection", {})
    trend = ctx.get("teleconnection_trend", {})

    if tele:
        st.subheader("Current Index Values")
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

        if trend:
            st.subheader("1-Month Trends")
            t1, t2, t3, t4, t5 = st.columns(5)
            cols_list = [t1, t2, t3, t4, t5]
            for i, (k, v) in enumerate(trend.items()):
                if i < 5:
                    cols_list[i].metric(k.upper(), f"{tele.get(k, '?'):.2f}",
                                        delta=f"{v:+.2f}")

        mjo = ctx.get("mjo_note", "")
        if mjo:
            st.info(mjo)

    tele_plot = os.path.join(PLOTS, "current_telecon_state.png")
    if os.path.exists(tele_plot):
        with st.expander("Static Teleconnection State Plot"):
            st.image(tele_plot, width="stretch")


def render_correlation():
    st.header("Predictor Correlations")

    corr_plot = os.path.join(PLOTS, "correlation_heatmap.png")
    if os.path.exists(corr_plot):
        st.image(corr_plot, width="stretch")
    else:
        st.info("No correlation heatmap. Run `python forecast.py` to generate.")

    fi_plot = os.path.join(PLOTS, "feature_importance.png")
    if os.path.exists(fi_plot):
        with st.expander("Feature Importance Plot"):
            st.image(fi_plot, width="stretch")


def render_data_explorer():
    st.header("Data Explorer")

    data_choice = st.selectbox("Select dataset", [
        "Forecast Results",
        "Sounding Forecast",
        "Nowcast JSON",
        "NWAC Stations",
        "Analog Detail",
        "Feature Importance (SWE)",
        "Feature Importance (Snow)",
        "CV Scores",
        "Forecast vs Actual",
    ], key="data_explorer_choice")

    if data_choice == "Forecast Results":
        fc = load_forecast()
        if not fc.empty:
            st.dataframe(fc, width="stretch", hide_index=True)
    elif data_choice == "Sounding Forecast":
        snd = load_sounding()
        if not snd.empty:
            model_filter = st.multiselect("Model", snd["model"].unique().tolist(),
                                          default=snd["model"].unique().tolist(),
                                          key="de_model_filter")
            level_filter = st.multiselect("Level (hPa)", sorted(snd["level_hPa"].unique().tolist()),
                                          default=sorted(snd["level_hPa"].unique().tolist()),
                                          key="de_level_filter")
            filtered = snd[(snd["model"].isin(model_filter)) & (snd["level_hPa"].isin(level_filter))]
            st.dataframe(filtered, width="stretch", hide_index=True)
            st.caption(f"{len(filtered)} rows")
    elif data_choice == "Nowcast JSON":
        nc = load_nowcast()
        st.json(nc)
    elif data_choice == "NWAC Stations":
        nwac = load_nwac_current()
        st.json(nwac)
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


# ==============================================================================
# SIDEBAR -- Navigation, Pipeline Controls, Chatbot
# ==============================================================================

CATEGORIES = {
    "Overview": ["Forecast Overview"],
    "Weather": ["NWS / NBM", "Sounding & Freezing Level",
                "NWAC Telemetry", "Inversion & Cross-Section"],
    "Model / Data": ["Model Performance", "Analog Years", "Teleconnection State",
                      "Correlation Heatmap", "Data Explorer"],
}

with st.sidebar:
    st.title("Snoqualmie Pass")
    category = st.radio("Section", list(CATEGORIES.keys()), index=0,
                         label_visibility="collapsed")

    st.divider()
    with st.expander("Pipeline Controls", expanded=False):
        st.caption("Run pipeline scripts to refresh data.")
        if st.button("Run Nowcast", help="Refresh station telemetry + sounding"):
            _run_pipeline_script("nowcast.py")
        if st.button("Run Forecast", help="Full ML forecast pipeline (~2 min)"):
            _run_pipeline_script("forecast.py")
        if st.button("Fetch Predictors", help="Download latest teleconnection indices"):
            _run_pipeline_script("fetch_new_predictors.py")
        if st.button("Run Backtest", help="Leave-one-year-out backtest"):
            _run_pipeline_script("backtest.py")

    st.divider()
    with st.expander("Ask about the data", expanded=False):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Show history
        for msg in st.session_state.chat_messages[-6:]:  # last 6 messages
            role_label = "You" if msg["role"] == "user" else "Assistant"
            st.markdown(f"**{role_label}:** {msg['content']}")

        user_q = st.text_input("Ask a question:", key="chat_q",
                                placeholder="e.g., What is the SWE forecast for March?")
        if st.button("Send", key="chat_send") and user_q:
            st.session_state.chat_messages.append({"role": "user", "content": user_q})
            with st.spinner("Thinking..."):
                response = _chat_respond(user_q)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()


# ==============================================================================
# MAIN CONTENT -- Render selected category + sub-tabs
# ==============================================================================

if category == "Overview":
    render_forecast_overview()

elif category == "Weather":
    tabs = st.tabs(CATEGORIES["Weather"])
    with tabs[0]:
        render_nws_nbm()
    with tabs[1]:
        render_sounding()
    with tabs[2]:
        render_nwac_telemetry()
    with tabs[3]:
        render_inversion()

elif category == "Model / Data":
    tabs = st.tabs(CATEGORIES["Model / Data"])
    with tabs[0]:
        render_model_performance()
    with tabs[1]:
        render_analog_years()
    with tabs[2]:
        render_teleconnection()
    with tabs[3]:
        render_correlation()
    with tabs[4]:
        render_data_explorer()


# -- Footer -------------------------------------------------------------------
st.divider()
st.caption("Snoqualmie Pass Snowpack Forecast | Layer 1: Teleconnection ML | "
           "Layer 2: Station Telemetry + Open-Meteo Soundings")
