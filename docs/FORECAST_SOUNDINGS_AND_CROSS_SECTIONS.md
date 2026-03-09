# Forecast Soundings, Cross Sections & Wet Bulb — Options & Ideas

Brainstorm of open-source forecast data we can use for **point soundings**, **temp/wind cross sections** up to ~10k ft, and **wet-bulb** to improve snowfall/snowmaking insight.

---

## 1. What We Already Have

- **Surface forecasts**: Open-Meteo multi-model (ECMWF, GFS, HRRR) at 2 m / 10 m — in `build_snoqualmie_weather_pipeline.py`, no vertical levels.
- **Wet bulb on the ground**: Computed in `organize_data.py` from station T + RH (Stull approximation), with **snowmaking good/marginal** and **snowfall possible/likely** hour targets. Already in `snoqualmie_model_daily.csv`.
- **Freezing level**: Mentioned in DATA_REVIEW as a desired proxy; not yet in the pipeline.

---

## 2. Point Soundings (Vertical Profile at One Location)

### Option A: Open-Meteo pressure-level API (easiest, no new infra)

- **Same API** we use for surface forecasts supports **pressure-level variables** at 20 levels (1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, … hPa).
- At each level we get: **temperature**, **dew_point**, **relative_humidity**, **wind_speed**, **wind_direction**, **geopotential_height**.
- **925 hPa ≈ 800 m**, **850 hPa ≈ 1.5 km**, **700 hPa ≈ 3 km** → enough for “up to 10k ft” (≈ 3 km) with a few levels; 500 hPa adds ~5.6 km.
- **Freezing level**: Open-Meteo also exposes **`freezing_level_height`** (meters ASL) as an instant variable — we can pull that directly for a point.
- **Wet bulb in the vertical**: From T and dewpoint (or T and RH) at each pressure level we can compute **wet-bulb temperature** (same Stull-style approximation) at 925, 850, 700 hPa, etc. That gives a **wet-bulb profile** — valuable for when snow level will drop or when snowmaking is favorable aloft vs at the surface.

**Implementation**: Add a small script or branch in the pipeline that calls the **forecast** endpoint with `latitude`, `longitude` and the pressure-level variables for 1000–500 hPa (or 925–700 for 0–10k ft). Store one row per (valid_time, level) or a compact JSON/CSV per run. Reuse the same Stull wet-bulb formula on (T, RH) or (T, Td) per level.

### Option B: Herbie (GFS / HRRR) for full native soundings

- **Herbie** (Python) downloads GFS/HRRR GRIB2 from NOAA/cloud and has a **`pick_points`** accessor to extract **vertical profiles at (lat, lon)**.
- Gives **native model levels** (more vertical resolution than pressure levels) and is well-suited to “sounding” style output.
- **Cost**: Need to pull GRIB (larger downloads), depend on Herbie + xarray, and possibly cache locally. Better for “best possible” soundings; Open-Meteo is lighter and already in our stack.

**Recommendation**: Start with **Option A** (Open-Meteo pressure levels + `freezing_level_height`) for point soundings and wet-bulb profiles; add Herbie later if we need finer vertical resolution or model-native levels.

---

## 3. Temp / Wind / Terrain Cross Sections (Distance vs Height)

- **Idea**: A 2D cross section along a line (e.g. W–E across Snoqualmie Pass): horizontal distance vs height (m ASL), with **temperature**, **wind**, and **terrain**.
- **Data**:
  - **Weather**: For each of several points along the transect, get **pressure-level** data (Open-Meteo or Herbie). Then interpolate in the vertical using **geopotential_height** so we have T and wind on a height grid (e.g. 0–10k ft).
  - **Terrain**: Elevation along the same line from a **DEM** (e.g. SRTM 30 m, or a simple elevation API). WRF-cross-section style tools (e.g. wrf-python `vertcross`, MetPy `cross_section`) typically work on gridded model data; here we’re building the cross section from **point** forecasts.
- **Practical approach**:
  1. Define a transect (e.g. 5–10 points from lowland to pass to lee).
  2. For each point, fetch Open-Meteo pressure-level forecast (same as soundings).
  3. For each point, get elevation (e.g. Open-Meteo elevation or SRTM).
  4. Build a 2D array: distance (km) × height (m ASL). For each (distance, height), interpolate from the nearest points’ pressure-level data (using geopotential_height to convert level → height).
  5. Plot: terrain as a filled region; contours or color for T; barbs or quiver for wind (e.g. component along cross section and vertical or magnitude).
- **Libraries**: MetPy’s `cross_section` is for pre-gridded data; we’d do our own lightweight interpolation from point soundings. Alternatively, use Herbie to pull a **2D slice** (e.g. GFS/HRRR along a line) and use MetPy/wrf-python if we move to gridded model data.

**Recommendation**: Implement a **minimal cross section** using **Open-Meteo pressure levels at 5–10 points** plus elevation from the same API or a small DEM. That gives “good enough” temp/wind vs height and terrain without introducing full NWP grids yet.

---

## 4. Wet Bulb — Why It’s Valuable & How to Extend It

- **Already in use**: Surface wet bulb in `organize_data.py` drives **snowmaking good/marginal** and **snowfall possible/likely** hours. That’s exactly what we want at the surface.
- **Improvements**:
  - **Forecast wet bulb**: Add **forecast** wet-bulb (from Open-Meteo 2 m T + RH or from our pressure-level T + dewpoint at surface) so the dashboard/forecast can show “next 48 h snowmaking window” and “snowfall possible” windows.
  - **Vertical wet-bulb profile**: From pressure-level T and Td (or RH), compute wet bulb at 925, 850, 700 hPa. That tells us **where** the “snowmaking/snow” layer is (e.g. wet bulb &lt; 28 °F aloft but not at the surface → snow level will drop). Helps with “rain vs snow” and snow-level forecasting.
  - **Freezing level**: Pull **`freezing_level_height`** from Open-Meteo (and optionally from soundings as the 0 °C level). Use it as a **predictor** in the model and as a **dashboard** indicator (e.g. “freezing level 2500 m — snow level ~2500 m”).

---

## 5. Other Open-Source Data / Ideas to Improve

| Idea | Source | Notes |
|------|--------|------|
| **Freezing level (instant)** | Open-Meteo | Variable `freezing_level_height` — add to pipeline and to model/dashboard. |
| **Pressure-level forecast** | Open-Meteo | Same API; add 925–700 hPa (and 500 if desired) for soundings + cross sections. |
| **ERA5 reanalysis** | Copernicus / CDS | Historical vertical levels for calibration of freezing level and wet bulb aloft (more setup). |
| **HRRR/GFS soundings** | Herbie | When we need higher vertical resolution or model-native levels. |
| **Terrain elevation** | Open-Meteo (elevation in API), SRTM, or Mapbox | For cross-section base and “elevation of station” consistency. |
| **Snow level / wet-bulb profile** | Derived from our soundings | Compute from pressure-level T/Td; show in dashboard next to surface wet bulb. |
| **Ensemble spread** | Open-Meteo (if multi-model) / ECMWF open | For “uncertainty” of freezing level and snowmaking windows. |

---

## 6. Suggested Order of Work (While Waiting on Dashboard)

1. **Freezing level**: Add `freezing_level_height` from Open-Meteo forecast to the pipeline (and optionally to a small “sounding” output). Expose in processed data and in the dashboard.
2. **Point soundings**: Small script or pipeline option that requests **pressure-level** data for Snoqualmie Pass (one lat/lon). Output: time × level table (or JSON) with T, Td, RH, wind, geopotential_height; optionally **wet_bulb** per level (Stull from T + Td).
3. **Forecast wet bulb (surface)**: From forecast 2 m T + RH, compute wet-bulb and “snowmaking / snowfall possible” windows for the next 1–5 days; add to forecast output or dashboard.
4. **Cross section**: After soundings work, add 5–10 points along a transect, fetch pressure levels + elevation, build distance × height cross section (temp, wind, terrain).
5. **Herbie soundings**: Optional later step if we need finer vertical resolution or GFS/HRRR-specific products.

This keeps everything buildable with **open-source, free APIs** (Open-Meteo first; Herbie/ERA5 as upgrades) and ties directly into the existing wet-bulb and snowmaking logic in `organize_data.py`.
