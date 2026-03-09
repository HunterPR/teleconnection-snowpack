# Custom Source Drops

Place any downloaded DOT or ski-area historical CSV files in this folder.

`organize_data.py` will automatically:
- detect a date/time column (for example: `date`, `datetime`, `time`, `timestamp`)
- coerce numeric columns
- aggregate to daily values
- prefix columns with `custom_<filename>_...`
- merge them into `data/processed/snoqualmie_model_daily.csv`

Useful outputs:
- `data/processed/custom_daily_features.csv`
- `data/processed/custom_source_manifest.csv`

Notes:
- Keep one source per file when possible.
- Include a clear date or datetime column.
- If a file does not parse, check `custom_source_manifest.csv` for status/details.

---

## Snoqualmie Pass snowfall (SNO38, ALP31, ALP43, ALP44, ALP55, SNO30)

To **focus snowfall on Snoqualmie Pass** using multiple nearby stations, add CSVs named exactly:

- **sno38.csv**, **alp31.csv**, **alp44.csv**, **alp55.csv**, **alp43.csv**, **sno30.csv**

Each file must have:
- A **date or datetime** column
- A **snow depth** or **new snow** column (e.g. `snow_depth`, `snow_depth_in`, `snow_interval_set_1_in`, `new_snow`)

`organize_data.py` will:
- Compute daily snowfall per station (day-over-day change in daily max snow depth, or sum of interval/new_snow)
- Combine stations by **mean** per date (pass-representative extrapolation)
- Overwrite the main **target_snowfall_24h_in** in `snoqualmie_daily_targets.csv` and the model table with this series
- Add **n_stations_snowfall** (how many stations contributed each day) when multiple files are present

Priority order for which stations are used: SNO38, ALP31, ALP44, ALP55, ALP43, SNO30. You can include any subset; one station is enough.

## Other stations (TTann, Thome, Tdenn, etc.)

Any other CSV in this folder (e.g. **TTann.csv**, **Thome.csv**, **Tdenn.csv**) is ingested as **custom features**: columns are prefixed with `custom_<filename>_` and merged into `custom_daily_features.csv` and `snoqualmie_model_daily.csv`. Use these for temperature, precipitation, or other daily series; they do not overwrite the pass snowfall target unless the filename is one of the pass station names above.
