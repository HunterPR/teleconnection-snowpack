# High-Resolution Lidar — Future Use Ideas

You have high-res lidar; below are ways it could add value to Snoqualmie Pass snowpack and operations. **Implementation is for later**; this doc captures options so we can prioritize when ready.

---

## 1. Snow depth / SWE mapping

- **DSM/DTM differencing**: Multi-date lidar (snow-on vs snow-off or earlier snow-on) gives **snow depth** at high spatial resolution (e.g. 1–5 m). Convert to **SWE** using density assumptions or sparse in-situ (e.g. SNOTEL) calibration.
- **Use**: Basin-scale SWE distribution, drift patterns, validation of point forecasts (SNOTEL, model grid).
- **Caveat**: Need snow-off and snow-on flights; density still often from models or sparse obs.

---

## 2. Terrain and cross-section context

- **DEM from lidar**: Very accurate **elevation** for the corridor and surrounding terrain. Improves:
  - **Cross sections**: Use real terrain in temp/wind cross sections (e.g. along I-90 or a ski run) instead of coarse DEMs.
  - **Snow level**: Map “where is the freezing level intersecting terrain” at lidar resolution for road/ski elevation bands.
- **Aspect/slope**: Derive aspect and slope; correlate with snow persistence and melt for site-specific insights.

---

## 3. Vegetation and canopy

- **Canopy height / closure**: Lidar can separate ground from vegetation. Use for:
  - **Under-canopy snow**: Where and when snow is visible vs obscured; bias in satellite snow products.
  - **Melt rates**: Shaded vs sun-exposed slopes; refine simple elevation-based melt.
- **Blow-off / wind-scour**: Identify wind-exposed vs sheltered terrain from vegetation structure; tie to observed drift patterns.

---

## 4. Operational / safety

- **Avalanche runout and terrain**: High-res topography improves runout modeling and “terrain traps” relevant to road and ski ops.
- **Drainage and ponding**: Fine-scale flow accumulation and depressions for wet spots, icing risk, or spring melt patterns along the corridor.
- **Structure and assets**: If lidar includes structures, clearance and loading (e.g. snow on roofs) in key zones.

---

## 5. Integration with current stack

- **Validation**: Compare lidar-derived snow depth/SWE to SNOTEL and to forecast model output (e.g. downscaled grid or point).
- **Features for models**: Basin-mean or banded (elevation/aspect) lidar SWE as a **predictor** or **target** in statistical models once we have a time series of lidar products.
- **Dashboard**: When available, show lidar-derived depth/SWE map or transect next to forecast and analogs.

---

## Suggested order when we get to it

1. **Document** lidar specs (resolution, extent, dates, snow-on/off).
2. **DEM + terrain**: Generate/ingest DEM; use for cross-section terrain and elevation bands.
3. **Snow depth from differencing**: If multi-date snow-on (and ideally snow-off) exists, pilot snow depth → SWE product for one basin or transect.
4. **Canopy/vegetation**: If point cloud allows, add canopy metrics and link to snow persistence.
5. **Ongoing**: Tie lidar-derived metrics into the forecast pipeline and dashboard as optional layers.

No code or pipeline changes yet; this is a roadmap for when lidar processing is prioritized.
