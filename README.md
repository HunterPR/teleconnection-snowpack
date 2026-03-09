# Snoqualmie Pass Snowpack Forecast

Teleconnection-driven ML forecasts for Snoqualmie Pass SWE and snowfall, with a Streamlit dashboard. Uses **Sno-Pass-first** targets (Stampede Pass SNOTEL, DOT/ALP stations) and 20-year SNOTEL correction where needed.

## Quick start

```bash
pip install -r requirements.txt
python fetch_data.py
python build_merged_dataset.py
python forecast.py
streamlit run dashboard.py
```

After adding DOT/ALP station CSVs (e.g. `sno38.csv`, `alp31.csv`) to `data/custom_sources/`:

```bash
python organize_data.py
python build_merged_dataset.py
python forecast.py
```

Then open the dashboard and click **Refresh data** in the sidebar to load the latest results.

## Run order (pipeline)

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `python fetch_data.py` | Teleconnections (ONI, PDO, PNA, AO, NAO, MJO) + Snoqualmie SNOTEL #908 |
| 2 | `python fetch_new_predictors.py` | Optional: EPO, Nino4, Z500, AMO, Stampede Pass SNOTEL |
| 3 | `python organize_data.py` | Optional: if using `data/custom_sources/` (DOT/ALP) for pass snowfall |
| 4 | `python build_merged_dataset.py` | Build `Merged_Dataset.csv` from data/ (so forecast can run) |
| 5 | `python forecast.py` | Train models, forecast, write `data/`, `models/`, `plots/` |
| 6 | `streamlit run dashboard.py` | View dashboard; use **Refresh data** after re-running forecast |

If `Merged_Dataset.csv` is missing, `forecast.py` will try to build it automatically via `build_merged_dataset`.

## Dashboard

- **Dashboard** reads from `data/` and `models/` (CSVs, parquet, pkl). After you run `forecast.py`, click **Refresh data (reload from disk)** in the sidebar so the dashboard shows the new run.
- Deploy: see **DEPLOY.md** (Streamlit Community Cloud + GitHub).

## Data

- **Pass snowfall stations** (priority): SNO38, ALP31, ALP44, ALP55, ALP43, SNO30 — place CSVs in `data/custom_sources/`. See `data/custom_sources/README.md`.
- **Other stations** (TTann, Thome, Tdenn, etc.): any other CSV in `custom_sources/` is ingested as custom features (prefix `custom_<name>_`).
- **SWE**: Stampede Pass SNOTEL preferred; Snoqualmie #908 with 20-yr correction when Stampede is missing. See `sno_pass_correction.py`.

## Docs

- **HANDOFF.md** — Next steps, Sno-Pass focus, key paths.
- **DEPLOY.md** — GitHub + Streamlit Cloud deployment.
- **data/README.md** — Data layout and workflow.

## Repo layout

- `forecast.py` — Main forecast pipeline (load, patch, train, forecast, save).
- `build_merged_dataset.py` — Build `Merged_Dataset.csv` from data/.
- `organize_data.py` — Process custom_sources and daily targets → `data/processed/`.
- `sno_pass_correction.py` — Sno-Pass-first WTEQ and snowfall (Stampede / 20-yr correction).
- `dashboard.py` — Streamlit app (run with `streamlit run dashboard.py`).
- `data/` — Input and output CSVs; `data/processed/` from organize_data.
- `models/` — Saved models and forecast parquet (from forecast.py).
- `plots/` — PNGs from forecast.py (correlation heatmap, forecast summary, etc.).
