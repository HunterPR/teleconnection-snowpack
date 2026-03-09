# Extra targets: temp, wet bulb, freezing level, snowpack gain/loss, inversion

How to **model additional variables** alongside WTEQ and snowfall: temp, wet bulb, freezing level, snowpack gain/loss, inversion likelihood/days.

---

## 1. Current targets

- **WTEQ** (SWE) and **snow_inches** (monthly snowfall) are the main forecast targets. Training data come from `Merged_Dataset.csv` (SNOTEL + pass stations, 20-yr correction).

---

## 2. Temperature

- **Source**: Station 2 m T from `snoqualmie_daily_targets.csv` (`target_mean_temp_f`, `target_min_temp_f`, `target_max_temp_f`) or from pipeline/synoptic.
- **Aggregate to monthly**: Mean/min/max per (year, month) and add to `Merged_Dataset.csv` as `temp_mean_f`, `temp_min_f`, `temp_max_f` (or similar).
- **Model**: Add a separate model in `forecast.py` (e.g. `train_models(df, "temp_mean_f", months=WINTER_MONTHS)`) and forecast monthly mean temp the same way we do WTEQ/snow. Use same teleconnection features.

---

## 3. Wet bulb

- **Source**: Already in `organize_data.py`: `target_mean_wet_bulb_f`, `target_min_wet_bulb_f`, `target_max_wet_bulb_f` from Stull formula (T + RH). In `snoqualmie_model_daily.csv` and daily targets.
- **Monthly**: Aggregate to (year, month) → `wet_bulb_mean_f`, etc. Add to merged dataset.
- **Model**: Add `train_models(df, "wet_bulb_mean_f", ...)` and forecast. Useful for “snowmaking window” and snow-level insight.

---

## 4. Freezing level

- **Source**: From `fetch_synoptic_features.py` → `syn_freezing_level_ft` or `freezing_level_m_mean` in pipeline/synoptic daily. Aggregate to monthly mean (or use “monthly mean freezing level”).
- **Merged**: Add column `freezing_level_ft` or `freezing_level_m` to merged dataset when synoptic/pipeline data exist.
- **Model**: Regress monthly mean freezing level on teleconnections; forecast next 1–3 months. High freezing level = snow level above pass (bad for snow).

---

## 5. Snowpack gain or loss

- **Definition**: Month-over-month change in WTEQ: `WTEQ_gain = WTEQ(t) − WTEQ(t−1)` (or daily delta from SNOTEL).
- **Source**: Compute from existing WTEQ in `Merged_Dataset.csv` or from daily SWE when available.
- **Model**: Train on `WTEQ_gain` (or `snowpack_delta`) as target. Positive = gain, negative = melt/loss. Helps forecast “will we gain or lose snowpack this month?”

---

## 6. Inversion likelihood and strength

- **Strength**: From soundings: e.g. T_925 − T_surface (positive = inversion). See `docs/STORM_CYCLES_AND_RADAR.md`.
- **Likelihood**: Proportion of days in month with inversion strength > threshold → `inversion_days` or `inversion_likelihood`.
- **Track**: Build `inversion_daily.csv` (date, strength, is_inversion_day). Aggregate to monthly.
- **Forecast**: Add inversion as target; predict “inversion days” next month from teleconnections or persistence.

---

## 7. Wiring into the pipeline

- **build_merged_dataset.py**: When building `Merged_Dataset.csv`, merge in:
  - Monthly temp and wet bulb from `data/processed/snoqualmie_daily_targets.csv` (aggregate by year, month).
  - Monthly freezing level from `data/synoptic_daily_features.csv` or `data/pipeline/` when present.
  - WTEQ delta (snowpack gain/loss) from WTEQ already in merged.
  - Inversion monthly from `data/processed/inversion_daily.csv` when that script exists.
- **forecast.py**: For each extra target (temp, wet_bulb_mean_f, freezing_level_ft, WTEQ_gain, inversion_days), add a `train_models(..., target=...)` and save forecasts to `data/forecast_results.csv` or a separate CSV. Dashboard can then show these alongside SWE and snowfall.

Start with **temp** and **wet bulb** (data already in daily targets); then **freezing level** once synoptic/pipeline is run; then **snowpack gain/loss** (derived from WTEQ); then **inversion** once the inversion script is in place.
