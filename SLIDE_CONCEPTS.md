# Snoqualmie Forecast Slide Concepts

## 1) Why This Matters
- Snoqualmie Pass operations are highly sensitive to snowfall phase and freezing level.
- Goal: improve storm-period decisions with earlier probabilistic signal.

## 2) Data Coverage
- Daily targets from Snoqualmie station history (2003+).
- Teleconnections + nearby SNOTEL + streamflow + historical marine predictors.
- New marine context from historical NDBC backfill improves long-period coverage.

## 3) Bottom-Line Findings
- Freezing-line/temperature metrics are strongest direct snowfall-phase signals.
- Marine pressure/wind adds useful storm-intensity context.
- Teleconnections are weaker daily predictors but useful regime indicators at monthly scale.

## 4) Proposed Forecast Products
- Event probability: P(snowfall_24h >= 3 inches).
- Event probability: P(freezing_line_gap_ft < 0).
- Daily expected snowfall/precip range with confidence bands.

## 5) Next Data Acquisitions
- WSDOT RWIS nearest-pass stations (road temp + precip type).
- Reanalysis 500mb height / SLP gradient indices.
- Higher-resolution freezing-level truth source for calibration.

## 6) Operational Decision Framing
- Use event probabilities with threshold triggers (plow staffing, chain messaging).
- Communicate forecast in tiers: low / moderate / high impact windows.
