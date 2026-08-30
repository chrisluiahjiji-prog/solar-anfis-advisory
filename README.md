# Solar Energy Output Prediction & Advisory System

**Entrepreneurship Minor Project — ANFIS-based solar advisory tool for South India**

🔗 **Live dashboard:** https://chrisluiahjiji-prog.github.io/solar-anfis-advisory/



# What this is

A locally-calibrated solar output prediction and advisory tool built on an
Adaptive Neuro-Fuzzy Inference System (ANFIS). Rather than stopping at a
single accuracy number, the system wraps its core prediction in five
product-oriented features aimed at making solar forecasts genuinely useful
to a non-technical homeowner or installer — not just accurate.

The model is trained separately for each of four South Indian locations
(Kochi, Kozhikode, Thiruvananthapuram, Coimbatore), and the trained model
is reimplemented in plain JavaScript so the dashboard can fetch live
weather and compute fresh predictions directly in the browser — no server,
no MATLAB, no manual data refresh required.

# Core model

- **Inputs:** temperature, relative humidity, wind speed (daily)
- **Output:** solar irradiance (kWh/m²/day) — a direct proxy for panel output
- **Data source:** [NASA POWER](https://power.larc.nasa.gov/), Jan 2020–Jul 2026
- **Model:** first-order Sugeno ANFIS, 3 Gaussian membership functions per
  input, trained in MATLAB's Fuzzy Logic Toolbox (grid partitioning, 27
  rules per location)
- **Validation:** chronological 80/20 train/test split (never randomly
  shuffled, since this is a time series)

# Accuracy vs. baselines (Kochi)

| Model | RMSE | MAE |
|---|---|---|
| **ANFIS** | **0.906** | **0.665** |
| Naive (tomorrow = today) | 1.008 | 0.739 |
| Linear Regression | 0.920 | 0.702 |

# Accuracy by location

| Location | RMSE | MAE |
|---|---|---|
| Coimbatore | 0.773 | 0.598 |
| Kochi | 0.887 | 0.659 |
| Kozhikode | 1.087 | 0.810 |
| Thiruvananthapuram | 1.237 | 0.905 |

Accuracy tracks local climate stability — drier, less coastal locations
predict more reliably than humid, coastal ones.

# The five add-on features

1. **Monsoon-aware confidence flag** — every prediction is tagged
   High/Low confidence based on season and recent weather variability.
2. **Best-day recommendation** — ranks the next 7 forecasted days to
   suggest the best day to run high-power appliances (washing machine, AC,
   EV charging).
3. **Baseline comparison** — ANFIS is benchmarked against a naive
   "tomorrow = today" guess and a linear regression model.
4. **Solar-day streak tracker** — surfaces historical patterns like the
   longest run of high-irradiance days per year, and year-over-year
   comparisons for a given month.
5. **Explainability layer** — extracts the actual fuzzy rules ANFIS
   learned and translates them into a plain-English sentence, e.g.
   *"Today's lower-than-average output is mainly due to high humidity
   and low wind."*

# The live dashboard

The dashboard is a single self-contained HTML file with no backend:

- Pick any of the 4 locations, each running its own separately-trained model
- Browse the full historical test set day-by-day with predictions,
  confidence, and explanations
- See real actual-vs-predicted charts and per-location accuracy comparisons
- **Live forecast:** fetches real weather from Open-Meteo and computes
  predictions for the next 16 days directly in-browser, using a JavaScript
  reimplementation of the trained ANFIS rules (membership functions +
  linear rule outputs extracted from MATLAB)
- **Gap-bridging:** automatically fills the reporting lag between NASA
  POWER's historical data and today using Open-Meteo's historical archive,
  clearly labeling those days as "not yet published" for actual irradiance
  (only NASA's satellite data can confirm ground truth)
- Falls back gracefully to a static snapshot if live fetching isn't
  available (e.g. no internet connection)

# Repository contents

```
solar_split.m           Core pipeline: load, clean, split, train, evaluate,
                         and add all 5 features for a single location
generateSolarModel.m    Reusable function: trains a full model for any
                         given latitude/longitude
exportModelJSON.m       Exports a trained model's fuzzy rules to JSON for
                         use in the JavaScript dashboard
generateBridgeData.m    Fills the gap between historical data and today
                         using Open-Meteo's archive
model_*.json            Exported fuzzy rules per location
dashboard_data_*.csv    Historical test-set results per location
forecast_data_*.csv     Snapshot forecasts per location
index.html              The live dashboard
```

# Tools & technology

- MATLAB with the Fuzzy Logic Toolbox (ANFIS Editor)
- NASA POWER Data Access Viewer / API for historical weather and solar data
- Open-Meteo API for live and archived weather (forecast + historical)
- Vanilla JavaScript + HTML5 Canvas for the dashboard (no external
  dependencies at runtime)

# Why this is novel

Most undergraduate ANFIS solar projects stop at reporting a single
accuracy metric. This project keeps the core prediction task simple and
well-supported by existing research, while adding layers — confidence
awareness, actionable recommendations, model justification, pattern
discovery, and plain-English explainability — that shift it from a purely
technical exercise toward a usable, product-oriented tool. The client-side
reimplementation of the trained model further demonstrates the underlying
fuzzy-inference math is understood well enough to be rebuilt independently
of MATLAB, and lets the tool run as a genuinely standalone web application.
