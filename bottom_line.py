"""
Natural-language "bottom line" for Snoqualmie Pass snowpack forecasts.

Produces a short summary that adds color and weight to a human's distillation
of other sources and experience. Designed to be shown in the dashboard with
room for the forecaster to append or edit their own notes.
See docs/BOTTOM_LINE_AND_DASHBOARD.md for dashboard usage and human_notes.

Two modes:
  - Template: always available, no API key; 2–4 sentences from forecast + analogs + teleconnections.
  - LLM: optional; set OPENAI_API_KEY or ANTHROPIC_API_KEY to get a richer narrative (same context).
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Load .env so OPENAI_API_KEY / ANTHROPIC_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional LLM: try OpenAI first, then Anthropic
def _llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def build_context(
    fc_df,
    analog_scores,
    df,
    target_year: int = 2026,
    extra: Optional[dict[str, Any]] = None,
    nowcast_data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a structured context dict from forecast results, analog years, and base data.
    fc_df: forecast DataFrame (month, wteq_ensemble, snow_ensemble, wteq_pct, etc.)
    analog_scores: DataFrame of analog years with distance
    df: full training/merged DataFrame (for current teleconnection state)
    nowcast_data: optional dict from nowcast.json (Layer 2 station + sounding data)
    """
    context: dict[str, Any] = {
        "target_year": target_year,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Forecast summary
    if fc_df is not None and not fc_df.empty:
        rows = fc_df.to_dict("records")
        context["forecast_months"] = [
            {
                "month": r.get("month_name", r.get("month")),
                "wteq_ensemble": r.get("wteq_ensemble"),
                "snow_ensemble": r.get("snow_ensemble"),
                "wteq_pct": r.get("wteq_pct"),
                "snow_pct": r.get("snow_pct"),
                "wteq_hist_mean": r.get("wteq_hist_mean"),
                "snow_hist_mean": r.get("snow_hist_mean"),
            }
            for r in rows
        ]
        wteq_vals = [r.get("wteq_ensemble") for r in rows if r.get("wteq_ensemble") is not None]
        snow_vals = [r.get("snow_ensemble") for r in rows if r.get("snow_ensemble") is not None]
        context["wteq_season_total"] = round(sum(wteq_vals), 1) if wteq_vals else None
        context["snow_season_total"] = round(sum(snow_vals), 1) if snow_vals else None
    else:
        context["forecast_months"] = []
        context["wteq_season_total"] = None
        context["snow_season_total"] = None

    # Analog years
    if analog_scores is not None and not analog_scores.empty:
        context["analog_years"] = analog_scores["year"].astype(int).tolist()
        context["analog_distances"] = analog_scores["distance"].round(3).tolist()
    else:
        context["analog_years"] = []
        context["analog_distances"] = []

    # Current teleconnection state (latest available from df)
    if df is not None and not df.empty:
        tele_cols = [
            "ao", "roni", "enso34", "pdo", "pna", "nao",
            "epo", "nino4_anom", "amo",
            "index4_140e_mjo", "index6_120w_mjo", "index7_40w_mjo",
        ]
        available = [c for c in tele_cols if c in df.columns]
        if available:
            recent = df.dropna(subset=available).tail(1)
            if not recent.empty:
                context["teleconnection"] = {
                    c: round(float(recent[c].iloc[0]), 2) for c in available
                }
            else:
                context["teleconnection"] = {}
        else:
            context["teleconnection"] = {}

        # Simple 3-month trend for key indices (current minus 3 months ago)
        trend_cols = ["roni", "enso34", "pna", "pdo", "ao"]
        trend_available = [c for c in trend_cols if c in df.columns]
        if trend_available:
            tail = df.dropna(subset=trend_available).tail(4)
            if len(tail) >= 2:
                first = tail.iloc[0]
                last = tail.iloc[-1]
                context["teleconnection_trend"] = {
                    c: round(float(last[c] - first[c]), 2) for c in trend_available
                }
            else:
                context["teleconnection_trend"] = {}
        else:
            context["teleconnection_trend"] = {}

        # Note for LLM: MJO sectors that matter for Sno Pass (from feature importance)
        context["mjo_note"] = (
            "MJO sectors 6 (120°W) and 7 (40°W) show meaningful influence on Sno Pass "
            "snowpack in model feature importance; sector 4 (140°E) also appears."
        )
    else:
        context["teleconnection"] = {}
        context["teleconnection_trend"] = {}
        context["mjo_note"] = (
            "MJO sectors 6 (120°W) and 7 (40°W) are relevant for Sno Pass in feature importance."
        )

    if extra:
        context["extra"] = extra

    # Layer 2 nowcast / sounding context
    if nowcast_data:
        snd = nowcast_data.get("sounding", {})
        if snd and "error" not in snd:
            fzl = snd.get("freezing_level_forecast", {})
            context["freezing_level"] = {
                "current_ft": fzl.get("current_ft"),
                "min_48h_ft": fzl.get("min_48h_ft"),
                "max_48h_ft": fzl.get("max_48h_ft"),
            }
            context["snow_level_ft"] = snd.get("snow_level_ft")
            sf = snd.get("snowfall_possible_hours", {})
            context["snowfall_hours"] = {
                "next_48h": sf.get("next_48h"),
                "next_120h": sf.get("next_120h"),
            }
            w850 = snd.get("wind_850hPa_48h", {})
            if w850:
                context["wind_850"] = {
                    "mean_dir_deg": w850.get("mean_dir_deg"),
                    "mean_speed_mph": w850.get("mean_speed_mph"),
                }
        pace = nowcast_data.get("pace", {})
        if pace and "error" not in pace:
            context["station_pace"] = {
                "actual_snowfall_in": pace.get("actual_snowfall_in"),
                "pace_snowfall_in": pace.get("pace_snowfall_in"),
                "days_elapsed": pace.get("days_elapsed"),
                "days_in_month": pace.get("days_in_month"),
            }
        pres = nowcast_data.get("pressure", {})
        if pres and "error" not in pres:
            context["pressure"] = {
                "slp_hpa": pres.get("slp_hpa"),
                "change_12h": pres.get("change_12h"),
                "pattern": pres.get("pattern"),
            }

    return context


def _template_bottom_line(context: dict[str, Any]) -> str:
    """Generate a 2–4 sentence bottom line from context without calling an LLM."""
    parts = []

    tc = context.get("teleconnection") or {}
    trend = context.get("teleconnection_trend") or {}
    # Prefer RONI for ENSO classification; fallback to enso34
    enso_val = tc.get("roni") if tc.get("roni") is not None else tc.get("enso34")
    pna = tc.get("pna")
    ao = tc.get("ao")
    pdo = tc.get("pdo")
    epo = tc.get("epo")
    mjo4 = tc.get("index4_140e_mjo")
    mjo6 = tc.get("index6_120w_mjo")
    mjo7 = tc.get("index7_40w_mjo")

    flavor = "La Niña-like"
    if enso_val is not None:
        if enso_val < -0.5:
            flavor = "La Niña-like"
        elif enso_val > 0.5:
            flavor = "El Niño-like"
        else:
            flavor = "neutral ENSO"
    if pna is not None and pna > 0.5:
        flavor += " with positive PNA (ridge risk)"
    if ao is not None and ao < -1:
        flavor += "; strong negative AO"

    # 1) Current teleconnection status
    status_parts = [f"Winter {context.get('target_year', '')} outlook at Snoqualmie Pass: "]
    status_parts.append(f"Current teleconnection pattern is {flavor}.")
    if pdo is not None:
        status_parts.append(f"PDO is {pdo:+.2f}.")
    if epo is not None:
        status_parts.append(f"EPO is {epo:+.2f}.")
    parts.append(" ".join(status_parts))

    # 2) Trend that matters (e.g. RONI cooling/warming over last 3 months)
    roni_trend = trend.get("roni") if trend else None
    if roni_trend is not None and roni_trend != 0:
        if roni_trend < -0.3:
            parts.append("RONI has cooled over the last 3 months (La Niña tendency).")
        elif roni_trend > 0.3:
            parts.append("RONI has warmed over the last 3 months (El Niño tendency).")

    # 3) Forecasted values and what influences them
    months = context.get("forecast_months") or []
    if months:
        wteq_total = context.get("wteq_season_total")
        snow_total = context.get("snow_season_total")
        if wteq_total is not None and snow_total is not None:
            parts.append(
                f"Model ensemble suggests seasonal totals on the order of "
                f"{wteq_total}\" SWE and {snow_total}\" snowfall for the forecast window, "
                "driven by the current teleconnection state and lagged predictors."
            )
        pcts = [m.get("wteq_pct") for m in months if m.get("wteq_pct") is not None]
        if pcts:
            avg_pct = sum(pcts) / len(pcts)
            if avg_pct < 25:
                parts.append("Forecast sits below the historical distribution (low percentile).")
            elif avg_pct > 75:
                parts.append("Forecast sits above the historical distribution (high percentile).")
            else:
                parts.append("Forecast is near the middle of the historical distribution.")

    # 4) MJO (sectors 6–7 matter for Sno Pass)
    mjo_parts = []
    if mjo6 is not None or mjo7 is not None or mjo4 is not None:
        mjo_parts.append("MJO: sectors 6 (120°W) and 7 (40°W) show meaningful influence on Sno Pass;")
        if mjo6 is not None:
            mjo_parts.append(f"current index6 {mjo6:+.2f};")
        if mjo7 is not None:
            mjo_parts.append(f"index7 {mjo7:+.2f}.")
        if mjo_parts:
            parts.append(" ".join(mjo_parts))
    else:
        parts.append(context.get("mjo_note", "MJO sectors 6–7 are relevant for Sno Pass in model feature importance."))

    # 5) Analogs
    analogs = context.get("analog_years") or []
    if analogs:
        parts.append(
            f"Analog years that match the current pattern include {analogs[:5]}—"
            "use those winters for context and experience-based adjustment."
        )

    # 6) Near-term weather outlook (from sounding/nowcast)
    fzl_ctx = context.get("freezing_level")
    sf_ctx = context.get("snowfall_hours")
    sl_ft = context.get("snow_level_ft")
    if fzl_ctx and fzl_ctx.get("current_ft") is not None:
        fzl_str = f"Freezing level currently at {fzl_ctx['current_ft']}' ASL"
        if sl_ft is not None:
            fzl_str += f" (snow level ~{sl_ft}')"
        fzl_str += "."
        if sf_ctx:
            h48 = sf_ctx.get("next_48h", 0)
            h120 = sf_ctx.get("next_120h", 0)
            if h48 or h120:
                fzl_str += f" {h48} snowfall hours expected in the next 48h ({h120} over 120h)."
        parts.append(fzl_str)

    # 7) 850 hPa flow regime
    w850_ctx = context.get("wind_850")
    if w850_ctx and w850_ctx.get("mean_dir_deg") is not None:
        d = w850_ctx["mean_dir_deg"]
        spd = w850_ctx.get("mean_speed_mph", 0)
        if 180 <= d < 315:
            regime = "Pacific moisture fetch (favorable for orographic snow)"
        elif d >= 315 or d < 45:
            regime = "post-frontal northwesterly (colder, scattered showers)"
        else:
            regime = "easterly/continental (rain shadow, generally dry)"
        parts.append(f"850 hPa flow from {d:.0f} deg at {spd:.0f} mph -> {regime}.")

    # 8) Station pacing
    pace_ctx = context.get("station_pace")
    if pace_ctx and pace_ctx.get("actual_snowfall_in") is not None:
        actual = pace_ctx["actual_snowfall_in"]
        pace_for = pace_ctx.get("pace_snowfall_in")
        elapsed = pace_ctx.get("days_elapsed", "?")
        total = pace_ctx.get("days_in_month", "?")
        pace_str = f"Station telemetry: {actual}\" snowfall so far this month (day {elapsed}/{total})"
        if pace_for is not None:
            pace_str += f", pacing for {pace_for}\"."
        else:
            pace_str += "."
        parts.append(pace_str)

    return " ".join(parts)


def _call_llm(context: dict[str, Any]) -> str:
    """Call OpenAI or Anthropic to generate a short bottom line. Falls back to template on failure."""
    prompt = (
        "You are a concise snowpack forecaster. Given the following structured context "
        "for Snoqualmie Pass, write a short 'bottom line' paragraph (3–5 sentences) that:\n"
        "1) Describes CURRENT TELECONNECTION STATUS: RONI (Relative Oceanic Niño Index), PNA, PDO, AO, EPO, and MJO if present—what the pattern implies for PNW/Sno Pass.\n"
        "2) Discusses FORECASTED VALUES: seasonal SWE and snowfall totals from the model ensemble, and what influences them (teleconnections, lags, analogs).\n"
        "3) Notes whether a TREND matters: e.g. if RONI or PNA has changed over the last few months and what that implies.\n"
        "4) Includes MJO: we analyze MJO; sectors 6 (120°W) and 7 (40°W) have meaningful influence on Sno Pass weather—mention current MJO state if provided.\n"
        "5) Mentions analog years for context.\n"
        "6) NEAR-TERM WEATHER: if freezing level, snow level, or snowfall hours are in the context, describe the near-term outlook.\n"
        "7) STATION PACING: if station_pace data is present, note current-month snowfall pacing from station telemetry.\n"
        "Be specific with numbers. Do not use markdown or bullets. Use RONI (not ONI) for ENSO classification.\n\nContext:\n"
        + json.dumps({k: v for k, v in context.items() if k != "extra"}, indent=2)
    )

    if os.environ.get("OPENAI_API_KEY"):
        try:
            import openai
            client = getattr(openai, "OpenAI", None) or openai
            if callable(client):
                client = client()
            resp = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You write brief, factual forecast bottom lines for ski area operators."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
            )
            return (resp.choices[0].message.content or "").strip()
        except ImportError:
            return _template_bottom_line(context)
        except Exception as e:
            return _template_bottom_line(context)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
                max_tokens=400,
                system="You write brief, factual forecast bottom lines for ski area operators.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            return text.strip()
        except ImportError:
            return _template_bottom_line(context)
        except Exception as e:
            return _template_bottom_line(context)

    return _template_bottom_line(context)


def generate_bottom_line(
    context: dict[str, Any],
    use_llm: Optional[bool] = None,
) -> str:
    """
    Generate the natural-language bottom line.

    If use_llm is True and an API key is set, uses LLM. If use_llm is False, uses template only.
    If use_llm is None, uses LLM when available (OPENAI_API_KEY or ANTHROPIC_API_KEY), else template.
    """
    if use_llm is None:
        use_llm = _llm_available()
    if use_llm and _llm_available():
        return _call_llm(context)
    return _template_bottom_line(context)


def save_bottom_line(
    text: str,
    context: dict[str, Any],
    out_path: Optional[Path] = None,
    human_notes: Optional[str] = None,
) -> Path:
    """
    Save bottom line and context to a JSON file for the dashboard.
    Dashboard can load this, show the generated bottom line, and allow appending human_notes.
    """
    if out_path is None:
        base = Path(__file__).resolve().parent
        out_path = base / "data" / "bottom_line.json"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "bottom_line": text,
        "human_notes": human_notes or "",
        "context": {k: v for k, v in context.items() if k != "extra"},
        "generated_at": context.get("generated_at", datetime.utcnow().isoformat() + "Z"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path
