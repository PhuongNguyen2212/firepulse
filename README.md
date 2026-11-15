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
5. **Validate**: learn on 2023–2024, test on 2025.
6. **Visualise**: interactive web app with
   - region map (drag a box or pick a preset)
   - daily burning calendar (harmonised / MODIS raw / VIIRS raw on one colour scale)
   - fire season timing: start, midpoint, peak week and end for each year
   - "Is this year unusual?": one year against the range of all years
   - plain-English summary for the selected region
   - weekly raw vs harmonised comparison
   - "false trend" demo: what happens when a record switches sensors
   - busiest weeks and High/Low months

## Results (Australia, 2023–2025)

| Method (tested on 2025) | Weekly error | Per-square error |
|---|---|---|
| No harmonisation | 376.3% | 349.8% |
| One national K (0.231) | 14.2% | 21.3% |
| **Per-square k (Fire Pulse)** | **11.4%** | **18.7%** |

| Year | MODIS | VIIRS | K | K day | K night |
|---|---|---|---|---|---|
| 2023 | 373,213 | 1,577,641 | 0.237 | 0.316 | 0.134 |
| 2024 | 210,724 | 947,693 | 0.222 | 0.273 | 0.139 |
| 2025 | 239,970 | 1,119,901 | 0.214 | 0.268 | 0.135 |

**Finding:** the night-time factor is stable, while the daytime factor drifts. This is consistent with the ageing MODIS satellites' changing overpass times, so harmonisation should be recalibrated every year.

## Live demo

Hosted on GitHub Pages from the `docs/` folder: https://phuongnguyen2212.github.io/firepulse/

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
python app/build.py
cp app/index.html docs/index.html
# open app/index.html in a browser
```

## Project structure

```
pipeline/harmonize.py   extract → clean → grid → calibrate → validate → export
app/template.html       web app (no external libraries)
app/build.py            inlines the data into app/index.html
app/index.html          built app, ready to open or host
docs/index.html         copy of the built app for GitHub Pages
data/outline.json       Australia coastline (Natural Earth, public domain)
output/validation.json  validation results
```

## Limitations

- 3 years of data; the method is built to run on the full 2003–present archive for a 20-year baseline.
- A detection is a fire pixel, not a fire: counts measure burning activity, not number of fires.
- Fire Radiative Power (FRP) was tested but its MODIS/VIIRS ratio is unstable over time, so the calendar uses detection counts.
- Prototype for exploring historical patterns, not an operational warning system.

## Use of AI

This project was built with help from Claude (Anthropic), an AI assistant, which was used to draft and debug the Python pipeline and the web app code, to suggest the validation approach, and to help write the documentation and presentation text. All data comes from NASA FIRMS; all results were computed by running the code on that data. The team chose the challenge, sourced the data, reviewed the method and results, and made the design decisions.

## Data

NASA FIRMS (Fire Information for Resource Management System): MODIS Collection 6.1 and VIIRS 375 m active fire products.
