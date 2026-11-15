# Fire Pulse 🔥

**NASA Space Apps Challenge 2026 · Harmonization of MODIS and VIIRS Hot Spots**

Fire Pulse turns NASA FIRMS active-fire detections from two different satellite sensors (MODIS and VIIRS) into one consistent **burning-activity calendar**, so fire managers, scientists and land managers can compare fire seasons across years without being fooled by sensor changes.

## The problem

VIIRS has smaller pixels (375 m vs 1 km) and much better night-time sensitivity, so it records about **4.3× more detections** than MODIS for the same fires in Australia. If a fire record switches from MODIS to VIIRS, fire activity appears to jump several-fold even though nothing changed on the ground.

## The solution

1. **Extract** NASA FIRMS archive CSVs (MODIS C6.1 Terra+Aqua, VIIRS 375 m NOAA-20).
2. **Clean**: vegetation fires only (`type = 0`), MODIS confidence ≥ 30, VIIRS confidence nominal/high.
3. **Grid**: count detections per 0.5° square per day, per sensor.
4. **Calibrate**: learn a conversion factor per square, pulled toward the national factor where data is thin:

   ```
   K       = ΣMODIS ÷ ΣVIIRS                       (national factor)
   k_cell  = (M_cell + λ·K) ÷ (V_cell + λ),  λ = 200
   Harmonised = VIIRS × k_cell                      (MODIS-equivalent detections)
   ```
5. **Machine learning**: a gradient-boosting model (scikit-learn `HistGradientBoostingRegressor`, Poisson loss) predicts each square-day's MODIS count from VIIRS features: detections, day/night split, high-confidence share, fire radiative power (sum, max, mean), time of year, location and the square's ratio k. Settings were fixed before looking at the test year.
6. **Validate**: learn on 2023–2024, test on 2025, scoring every method the same way.
7. **Visualise**: interactive web app (switch between the ML and ratio methods) with
   - region map (drag a box or pick a preset)
   - daily burning calendar (harmonised / MODIS raw / VIIRS raw on one colour scale)
   - fire season timing: start, midpoint, peak week and end for each year
   - "Is this year unusual?": one year against the range of all years
   - plain-English summary for the selected region
   - weekly raw vs harmonised comparison
   - "false trend" demo: what happens when a record switches sensors
   - busiest weeks and High/Low months
   - 2026 forecasting contest (six algorithms scored against real 2026 data) and a 2027 forecast for any region

## Results (Australia, 2023–2025)

| Method (tested on 2025) | Weekly error | Per-square error |
|---|---|---|
| No harmonisation | 376.3% | 349.8% |
| One national K (0.231) | 14.2% | 21.3% |
| Per-square ratio k | 11.4% | 18.7% |
| **Machine learning (gradient boosting)** | **8.5%** | **17.2%** |

Not luck: trained on 2023 alone and tested on 2024–2025, the ML model scores 8.9% weekly error against 12.2% for the ratio. The features it relies on most are the VIIRS detection count, total fire radiative power, latitude and the night-time share of detections.

| Year | MODIS | VIIRS | K | K day | K night |
|---|---|---|---|---|---|
| 2023 | 373,213 | 1,577,641 | 0.237 | 0.316 | 0.134 |
| 2024 | 210,724 | 947,693 | 0.222 | 0.273 | 0.139 |
| 2025 | 239,970 | 1,119,901 | 0.214 | 0.268 | 0.135 |

**Finding:** the night-time factor is stable, while the daytime factor drifts. This is consistent with the ageing MODIS satellites' changing overpass times, so harmonisation should be recalibrated every year.

## Live demo

Hosted on GitHub Pages from the `docs/` folder: https://phuongnguyen2212.github.io/firepulse/

## 2026 forecasting contest and 2027 outlook

Six algorithms predicted MODIS detections for every 0.5° square and month of 2026 using only 2023–2025 data. The predictions were locked (timestamp + SHA-256 in `output/predictions_2026_LOCK.json`) before any 2026 data was downloaded, then scored against real NASA FIRMS MODIS C6.1 detections for January–October 2026 (128,407 detections).

| Rank | Algorithm | Monthly error | Total error | Per-region | Per-square |
|---|---|---|---|---|---|
| 1 | **Median of 3 years** | **36.4%** | **1.2%** | **41.0%** | 57.6% |
| 2 | Machine learning (gradient boosting) | 75.4% | 39.0% | 64.1% | **55.6%** |
| 3 | Ensemble (average + last year) | 93.9% | 61.5% | 58.2% | 57.7% |
| 4 | Climatology (3-year average) | 85.5% | 66.2% | 59.8% | 58.8% |
| 5 | Same as last year (2025) | 103.2% | 56.8% | 55.3% | 62.6% |
| 6 | Ensemble + El Niño analog | 105.8% | 78.6% | 60.1% | 59.5% |

2026 was a quiet year: both MODIS satellites recorded fewer detections (Aqua alone fell to about 84,000 in Jan–Oct, from 102,000–162,000 in 2023–2025), so the drop is not a sensor artefact. An independent check with VIIRS S-NPP agrees: S-NPP recorded 652,051 detections in Jan–Oct 2026, fewer than NOAA-20 recorded in any of 2023–2025 (719,328 to 1,194,519), and the MODIS ÷ S-NPP factor in 2026 (0.197) matches the 2019–20 value (0.195), so MODIS has not lost sensitivity. The median is robust to 2023's extreme spring; the El Niño analog performed worst.

The winner forecasts 2027: for each square and month, the median of 2024, 2025 and 2026 (November and December use 2023–2025). 80% ranges are sized from the winner's real 2026 errors. Forecast for all of Australia: about 132,000 MODIS detections (80% range 98,000–178,000).

```bash
python pipeline/forecast_2026.py predict                       # lock predictions (2023–2025 only)
python pipeline/forecast_2026.py score --actual data/raw/fire_archive_M-C61_*.csv data/raw/fire_nrt_M-C61_*.csv
python pipeline/forecast_2026.py forecast2027 --actual data/raw/fire_archive_M-C61_*.csv data/raw/fire_nrt_M-C61_*.csv
```

## Run it

```bash
pip install -r requirements.txt

# 1. Download FIRMS country-year CSVs (Australia) into data/raw/
#    modis_2023_Australia.csv ... viirs-jpss1_2025_Australia.csv

# 2. Run the pipeline
python pipeline/harmonize.py \
  --modis data/raw/modis_202{3,4,5}_Australia.csv \
  --viirs data/raw/viirs-jpss1_202{3,4,5}_Australia.csv \
  --split 2025-01-01

# 3. Build the web app (inlines output/firepulse_data.json)
python pipeline/ml_harmonize.py \
  --modis data/raw/modis_202{3,4,5}_Australia.csv \
  --viirs data/raw/viirs-jpss1_202{3,4,5}_Australia.csv \
  --split 2025-01-01 --robust-split 2024-01-01 --export

python app/build.py
cp app/index.html docs/index.html
# open app/index.html in a browser
```

## Project structure

```
pipeline/harmonize.py   extract → clean → grid → calibrate → validate → export
pipeline/ml_harmonize.py  gradient-boosting model, validation, adds ML values to the app data
app/template.html       web app (no external libraries)
app/build.py            inlines the data into app/index.html
app/index.html          built app, ready to open or host
docs/index.html         copy of the built app for GitHub Pages
data/outline.json       Australia coastline (Natural Earth, public domain)
output/validation.json  validation results (ratio methods)
output/ml_validation.json  ML vs ratio results, robustness check, feature importance
pipeline/forecast_2026.py  2026 forecasting contest (predict, score) and 2027 forecast
output/forecast_2026_results.json, forecast_2027_*  contest results and 2027 forecast
```

## Limitations

- 3 years of harmonised data (2023–2025), plus 2026 for the forecast test; the method is built to run on the full 2003–present archive.
- A detection is a fire pixel, not a fire: counts measure burning activity, not number of fires.
- Fire Radiative Power (FRP) was tested but its MODIS/VIIRS ratio is unstable over time, so the calendar uses detection counts.
- Prototype for exploring historical patterns, not an operational warning system.

## Future enhancements

| # | Enhancement | What it adds | Who benefits |
|---|---|---|---|
| 1 | **20-year baseline** | Add MODIS 2003–2022, processed year by year into Parquet (about 550 MB of raw CSV becomes about 15 MB), so "unusual" means compared with two decades instead of three years | Scientists, land managers |
| 2 | **Every VIIRS satellite** | Calibrate S-NPP, NOAA-20 and NOAA-21 separately ("one satellite, one factor") and recalibrate every year, since the daytime factor drifts | Scientists, NASA |
| 3 | **Daily automated pipeline** | Ingest FIRMS near-real-time data every day, so the calendar stays current and can flag unusual weeks as they happen | Fire and emergency crews |
| 4 | **Fire risk outlook** | Add NASA soil moisture (SMAP), rainfall (GPM) and vegetation dryness (MODIS NDVI) plus ENSO state; any new model must beat the median method that won the 2026 test | Fire agencies, planners |
| 5 | **Controlled-burn planner** | Suggest the safest burn windows for each region from its historical peak weeks | Land managers |
| 6 | **Fire event tracking and context** | Group hotspots into individual fire events, track their growth day by day, and add map layers (parks, land cover, towns); extend from Australia to the whole world | Responders, land managers |

## Use of AI

This project was built with help from Claude (Anthropic), an AI assistant, which was used to draft and debug the Python pipeline, the forecasting contest and the web app code, to suggest the validation and forecasting approach, and to help write the documentation and presentation text. All data comes from NASA FIRMS; all results were computed by running the code on that data. The team chose the challenge, sourced the data, reviewed the method and results, and made the design decisions.

## Data

NASA FIRMS (Fire Information for Resource Management System):
- MODIS Collection 6.1 (Terra + Aqua), Australia, 2023–2025 (country-year files) and January–November 2026 (archive + near-real-time files)
- VIIRS 375 m from NOAA-20, Australia, 2023–2025 (harmonisation)
- VIIRS 375 m from S-NPP, Australia, January–November 2026 (independent check of the 2026 result)

Raw files are kept out of the repository (`data/raw/` is git-ignored); download them from https://firms.modaps.eosdis.nasa.gov/
