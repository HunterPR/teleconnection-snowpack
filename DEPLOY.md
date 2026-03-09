# Deploying the dashboard (Streamlit)

Deploy the Snoqualmie Pass Snowpack dashboard to **Streamlit Community Cloud** so the app runs from your GitHub repo.

## 1. Push your repo to GitHub

- Repo name: **teleconnection-snowpack** (or your fork).
- Ensure `dashboard.py`, `requirements.txt`, and this file are on the branch you want to deploy (e.g. `main`).

## 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub.
3. Click **New app**.
4. **Repository**: `your-username/teleconnection-snowpack`  
   **Branch**: `main`  
   **Main file path**: `dashboard.py`
5. Click **Deploy**.

The app will install dependencies from `requirements.txt` and run `streamlit run dashboard.py`.

## 3. Data and models for the deployed app

The dashboard reads from:

- `data/` — e.g. `forecast_results.csv`, `analog_years.csv`, `bottom_line.json`, `forecast_vs_actual_recent.csv`, `snoqualmie_snotel.csv`, etc.
- `models/` — `forecast_df.parquet`, `models_wteq.pkl`, `models_snow.pkl`
- `plots/` — optional images

Your `.gitignore` currently excludes `data/processed/`, `data/pipeline/`, `models/`, `plots/`, and many `*.csv`/`*.pkl` files. So after a fresh clone, the app may show “No data” unless you either:

**Option A — Commit generated outputs (simplest for Cloud)**  
Run locally once:

```bash
python fetch_data.py
python forecast.py
```

Then force-add and commit the artifacts the dashboard needs:

```bash
git add -f data/forecast_results.csv data/analog_years.csv data/analog_detail.csv
git add -f data/bottom_line.json data/forecast_vs_actual_recent.csv
git add -f data/feature_importance_wteq.csv data/feature_importance_snow.csv
git add -f data/cv_scores_WTEQ.csv data/cv_scores_snow_inches.csv
git add -f data/snoqualmie_snotel.csv data/oni.csv data/pdo.csv data/pna.csv data/ao.csv data/nao.csv
git add -f models/forecast_df.parquet models/models_wteq.pkl models/models_snow.pkl
git add -f plots/*.png
git commit -m "Add generated data and models for dashboard deploy"
git push
```

Then redeploy the app on Streamlit Cloud (or it will auto-redeploy on push if connected).

**Option B — Run pipeline in the cloud**  
In Streamlit Cloud → app → **Settings** → **Advanced**:

- Set “Run before app” to something like:  
  `pip install -r requirements.txt && python fetch_data.py && python build_merged_dataset.py && python forecast.py`
  This builds Merged_Dataset.csv from data/ when missing, then runs the forecast.

Option A is usually more reliable; Option B keeps the repo smaller but needs the full pipeline to succeed in the cloud.

## 4. Secrets (optional)

If you use API keys (e.g. for the bottom-line LLM or WSDOT), add them in Streamlit Cloud → app → **Settings** → **Secrets**. Example:

```toml
OPENAI_API_KEY = "sk-..."
# or
ANTHROPIC_API_KEY = "sk-ant-..."
```

## 5. What else you can do

- **Scheduled runs**: Use GitHub Actions (or a cron job elsewhere) to run `fetch_data.py` and `forecast.py` periodically and commit updated `data/` and `models/` so the deployed app stays current.
- **CI**: Add a GitHub Action that runs `python forecast.py` (or tests) on push to catch breakage.
- **Alerts**: Have the pipeline or a script email/Slack when forecast or conditions cross a threshold.
- **Custom domain**: Streamlit Cloud allows a custom domain in paid plans.
