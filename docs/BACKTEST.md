# Backtesting the Snoqualmie Pass Forecast

Leave-one-year-out backtesting checks whether the model would have beaten **climatology** (predicting the historical mean for each month) when applied to past years.

## Run backtest

**Option 1 – from forecast.py (quick)**  
```bash
python forecast.py --backtest
```  
- If `data/tune_backtest_results.csv` exists (from running `tune_backtest.py`), uses the **best config** from that file for WTEQ and reports snowfall with the same config.
- Otherwise runs built-in configs (full ensemble, Ridge only, core telecons, Ridge+core+50% clim) and prints WTEQ + snowfall metrics.
- Writes `data/backtest_WTEQ.csv` and, when snowfall data exist, `data/backtest_snow_inches.csv`.

To ignore the tune results and always use built-in configs:  
```bash
python forecast.py --backtest --no-tune-results
```

**Option 2 – full backtest script (multiple configs)**  
```bash
python backtest.py
```  
Runs three configs (full ensemble, Ridge only, Ridge+RF+GBR), prints a summary table, saves `data/backtest_metrics.json` and the same CSVs.

## Metrics

- **RMSE**: Root mean squared error of predictions vs actuals.
- **RMSE_clim**: RMSE if we had predicted the historical mean for each month.
- **Skill**: `1 - RMSE / RMSE_clim`. Positive = model beats climatology.
- **Correlation**: Pearson correlation between predicted and actual (direction only).
- **Bias**: Mean(pred) - Mean(actual). Positive = model tends to overpredict.

## If performance is poor

1. **WTEQ skill ≤ 0**  
   - Try **Ridge only** in `backtest.py` (already compared as "Ridge only"); if Ridge has better skill, the full ensemble may be overfitting.  
   - In `forecast.py`, reduce **CORE_TELE** to a smaller set (e.g. `ao, roni, pdo, pna` only) and re-run backtest.  
   - Increase **min_train_rows** in `run_backtest()` so we use only years with enough data (e.g. 100+ rows).

2. **Snowfall skill ≤ 0 or very noisy**  
   - Monthly snowfall is hard to predict from teleconnections alone. Consider:  
     - Using **climatology** for snowfall in the dashboard and only forecasting WTEQ.  
     - Or keeping the snow model but clearly labeling it as experimental.

3. **High bias**  
   - If the model systematically over- or under-predicts, consider a **bias correction** in the forecast step (e.g. subtract mean backtest error by month).

4. **Tweak models**  
   - In `run_backtest()`, pass `model_names=["Ridge"]` or `["Ridge", "RF", "GBR"]` to compare.  
   - In `forecast.py`, `train_models` uses CV R²-weighted ensemble; you could try equal weights or inverse-RMSE weights from backtest (see `tune_ensemble_weights_from_recent`).

## Files

- `forecast.run_backtest()` – leave-one-year-out logic.  
- `backtest.py` – loads data, runs several configs, prints table and saves metrics.  
- `data/backtest_WTEQ.csv`, `data/backtest_snow_inches.csv` – year, month, actual, pred, clim_pred per point.  
- `data/backtest_metrics.json` – metrics per config and target.
