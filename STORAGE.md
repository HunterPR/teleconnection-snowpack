# Storage & Cleanup

Quick reference for what uses space and how to trim or regenerate it.

## What’s ignored (not committed)

- `__pycache__/`, `*.pyc` — Python bytecode
- `data/processed/` — outputs of `organize_data.py` (regenerable)
- `data/pipeline/` — outputs of `build_snoqualmie_weather_pipeline.py` (regenerable)
- `plots/` — figures from `forecast.py` (regenerable)
- `models/` — `forecast_df.parquet`, `models_*.pkl` (regenerable via `forecast.py`)
- `*.csv`, `*.parquet`, `*.pkl` — data and model artifacts

## Regenerating after cleanup

| After deleting | Regenerate with |
|----------------|-----------------|
| `data/processed/*` | `python organize_data.py` (needs source CSVs and `data/` inputs) |
| `data/pipeline/*` | `python build_snoqualmie_weather_pipeline.py` |
| `plots/*` | `python forecast.py` |
| `models/*` | `python forecast.py` |
| `data/analysis_*.csv` (buoy/ops) | `python analyze_conditions_and_buoy_lags.py` (after organize_data) |

## Optional space savings

- **Skip pipeline hourly** — The pipeline can write only daily forecasts; hourly is optional for dashboard or scripts that need it. To avoid recreating `model_forecast_hourly.csv`, don’t depend on it or delete it and use `model_forecast_daily.csv` only.
- **Thin pipeline history** — Use `--start-date` / `--end-date` in `build_snoqualmie_weather_pipeline.py` to fetch a shorter range and keep `openmeteo_station_daily.csv` smaller.
- **Clear caches** — Remove `__pycache__` under the project (e.g. `Remove-Item -Recurse -Force __pycache__` in PowerShell).

## One-off cleanup commands (PowerShell)

```powershell
# Remove Python caches
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force

# Optional: remove only pipeline outputs (keep data/ and data/processed/)
Remove-Item -Recurse -Force data\pipeline\*

# Optional: remove processed and re-run organize (keeps raw data)
Remove-Item -Force data\processed\* ; python organize_data.py
```

Or use the helper script (from repo root):

```bash
python scripts/cleanup_storage.py              # __pycache__ only
python scripts/cleanup_storage.py --dry-run   # show what would be removed
python scripts/cleanup_storage.py --pipeline  # also clear data/pipeline/*
python scripts/cleanup_storage.py --processed # also clear data/processed/*
```

Run from the repo root. Back up anything you care about before deleting.
