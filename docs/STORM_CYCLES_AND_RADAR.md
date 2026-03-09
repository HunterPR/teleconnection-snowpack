# Storm cycles, radar & conditions

How to view **individual storms / storm cycles**, **winds and shifts**, **freezing level**, **convergence zone**, and **radar**, and how to add **inversion** tracking.

---

## 1. Storm cycles and individual storms

- **Idea**: Segment time series into storm events (e.g. by precip > threshold, or by synoptic “storm day” from SLP/buoy).
- **Data**: Use `target_snowfall_24h_in` and `target_precip_24h_in` from `data/processed/snoqualmie_daily_targets.csv` (or pass stations). Cluster consecutive high-precip/snow days into a “storm cycle.”
- **Winds and shifts**: From pipeline or synoptic:
  - `data/pipeline/model_forecast_daily.csv` or `data/synoptic_daily_features.csv` (when present) can have `wind_speed_10m`, `wind_direction`, or similar. Run `fetch_synoptic_features.py` and/or `build_snoqualmie_weather_pipeline.py` to populate.
  - Wind **shift** (e.g. W → SW) can be derived from day-over-day change in direction; strength from speed.
- **Dashboard**: Add a “Storm cycles” section that:
  - Lists recent N-day windows where snowfall or precip exceeds a threshold.
  - Optionally plots wind speed/direction and freezing level over those windows.

---

## 2. Freezing level

- **Already in pipeline**: Open-Meteo `freezing_level_height` (m ASL) is in:
  - `fetch_synoptic_features.py` → `syn_freezing_level_ft`, `syn_freezing_line_gap_ft` (gap vs Snoqualmie Pass elev).
  - `build_snoqualmie_weather_pipeline.py` → `freezing_level_m_mean` in forecast daily.
- **Beefing it up**:
  - Run `fetch_synoptic_features.py` for historical + forecast synoptic daily; run `build_snoqualmie_weather_pipeline.py` for multi-model forecast freezing level.
  - In the dashboard, plot **freezing level over time** (and optionally “freezing line gap” = freezing level − pass elevation). Negative gap = snow level below pass.
- **Better truth**: Use RAOB/soundings (e.g. Herbie GFS/HRRR or Open-Meteo pressure levels) to get 0 °C level; see `docs/FORECAST_SOUNDINGS_AND_CROSS_SECTIONS.md`.

---

## 3. Convergence zone (Puget Sound)

- **What**: Puget Sound convergence zone is a mesoscale feature (winds converging, enhanced precip). No single “convergence zone index” in our CSVs yet.
- **Options**:
  - **Radar**: Use radar reflectivity (see below) to see when banding/convergence is present.
  - **Link**: [NWS Seattle convergence zone discussion](https://www.weather.gov/sew/convergence) or local AFD.
  - **Future**: Derive a simple CZ proxy from wind direction difference (e.g. offshore vs inland) or from reanalysis divergence/convergence if we add gridded data.

---

## 4. Radar

- **Weather.gov**: NWS radar for Puget Sound / Seattle area:
  - **Radar lite**: https://radar.weather.gov/radar_lite.php?rid=ATX (KATX – Camano Island).
  - **Full radar**: https://radar.weather.gov/radar.php?rid=ATX
  - **All radars**: https://radar.weather.gov/
- **In the dashboard**: Add an iframe or link: “View live radar (KATX)” pointing to the lite or full URL. No API key needed.
- **NEXRAD (AWS)**: For programmatic use, NEXRAD Level II/III is on AWS. We could add a script that pulls a recent sweep for a bounding box and saves a GeoTIFF or PNG for the dashboard. That’s a larger step; start with the link.
- **Summary**: For “incorporate radar” we add a **radar link/iframe** in the Storms & conditions tab; optional later: script to fetch NEXRAD and plot.

---

## 5. Inversion likelihood and strength

- **What**: Inversions (temp increasing with height) matter for snowmaking, fog, and snow level.
- **Data**:
  - **Soundings**: From Open-Meteo pressure levels or Herbie (GFS/HRRR), compute T at 925 vs 850 vs surface. Inversion strength = T_925 − T_surface (or similar). “Inversion day” = day when inversion strength > threshold.
  - **Reanalysis**: ERA5 or NCEP/NCAR on pressure levels would give historical inversion proxy.
- **Track and forecast**: Add a small pipeline that:
  1. Fetches pressure-level T (and optionally Td) for Snoqualmie point.
  2. Computes inversion strength (e.g. 925 hPa T − 2 m T) and flags “inversion day.”
  3. Saves to `data/processed/inversion_daily.csv` (date, inversion_strength_K, is_inversion_day).
  4. Optionally train a small model or use climatology to “forecast” inversion days (e.g. from teleconnections or persistence).

---

## 6. Implementation order

1. **Dashboard**: Add “Storms & conditions” tab with:
   - Freezing level (from synoptic/pipeline when available).
   - Wind (from same sources).
   - Radar link (KATX).
   - Convergence zone link and placeholder for future CZ index.
   - Storm cycles: list of recent high-precip/snow windows (from daily targets).
2. **Data**: Ensure `fetch_synoptic_features.py` and `build_snoqualmie_weather_pipeline.py` run and write to `data/` or `data/pipeline/` so the dashboard can load freezing level and wind.
3. **Inversion**: Add script to fetch pressure-level T, compute inversion strength, write `inversion_daily.csv`; then add inversion to dashboard and optional forecast.
