# Fire Pulse — Space Apps 2026 submission text

Paste each section into the matching field on the Project Submission page.

## Project Name
Fire Pulse

## High-Level Project Summary
Fire Pulse turns NASA's satellite fire record into one consistent burning-activity calendar. Active-fire hotspots come from two sensors, MODIS and VIIRS, that count the same fires very differently: in Australia, VIIRS records about 4.3 times more detections than MODIS. If a long record switches from one sensor to the other, fire activity appears to jump several-fold even though nothing changed on the ground. We built a Python pipeline that rescales VIIRS detections to the MODIS baseline, first with a conversion factor for every 0.5° grid square and then with a machine-learning model that also uses fire power, the day/night split, season and location, then an interactive web app where anyone can pick a region and see its daily burning calendar, when each fire season started, peaked and ended, and whether a year was unusual. Tested on 2025, a year the models never saw, the machine-learning version has an 8.5% weekly error, compared with 11.4% for the per-square ratio and 376% without harmonisation. This gives emergency responders, land managers and scientists a fire record they can compare across years and sensors.

## Link to Project Demo
[public link to the slide deck: export the deck as PDF and share it publicly, e.g. Google Drive "Anyone with the link"]

## Link to Final Project
https://phuongnguyen2212.github.io/firepulse/  (code: https://github.com/PhuongNguyen2212/firepulse)

## Detailed Project Description
**What it does.** Fire Pulse harmonises NASA FIRMS active-fire detections from MODIS (Terra and Aqua, Collection 6.1) and VIIRS 375 m (NOAA-20) into a single MODIS-equivalent record, and presents it as an interactive burning-activity calendar for Australia, 2023–2025.

**How it works.**
1. Extract: read FIRMS country-year archive CSVs for both sensors (about 5 million rows).
2. Clean: keep vegetation fires (type 0) with MODIS confidence ≥ 30 and VIIRS confidence nominal or high, leaving 4.47 million detections.
3. Grid: count detections per 0.5° square (about 50 km) per day, separately for each sensor. A detection is a fire pixel, not a whole fire, so we measure burning activity rather than counting fires.
4. Calibrate: compute the national factor K = ΣMODIS ÷ ΣVIIRS, then a factor for each square, k = (M + λ·K) ÷ (V + λ) with λ = 200, so squares with little data are pulled toward the national value. Harmonised activity = VIIRS × k.
5. Machine learning: a gradient-boosting model (scikit-learn, Poisson loss) predicts each square-day's MODIS count from VIIRS features: detections, day/night split, high-confidence share, fire radiative power, time of year, location and the square's ratio. Its settings were fixed before looking at the test year.
6. Validate: learn on 2023–2024 and test on 2025, scoring every method the same way. Weekly error: 376% with no harmonisation, 14.2% with one national factor (K = 0.231), 11.4% with per-square factors and 8.5% with the ML model (per-square error 17.2% vs 18.7%). Trained on 2023 alone and tested on 2024–2025, ML still wins: 8.9% against 12.2%.
7. Visualise: a self-contained web app, switchable between the ML and ratio methods, with a region map, a daily burning calendar (harmonised, MODIS raw or VIIRS raw on one colour scale), fire season timing (start, midpoint, peak week and end for each year), a week-by-week "is this year unusual?" chart against the range of all years, a plain-English summary for the selected region, a raw-versus-harmonised weekly chart, a "false trend" demo showing what happens when a record switches sensors, and the busiest weeks with High and Low months flagged.

**Key finding.** The night-time factor is stable (about 0.134 to 0.139) while the daytime factor drifts from 0.316 to 0.268 over three years. VIIRS is far more sensitive at night, and the daytime drift is consistent with the ageing MODIS satellites' changing overpass times. Harmonisation should therefore be recalibrated every year rather than fixed once, and each new VIIRS satellite needs its own factor.

**Benefits.** Fire and emergency crews can see when their region's season usually starts and peaks and staff up in time. Land managers can plan controlled burns outside peak weeks. Scientists get a long record that no longer jumps when satellites change. NASA's 25-year MODIS archive stays comparable as missions retire.

**Goals.** Extend the record to MODIS 2003–present for a 20-year baseline, calibrate S-NPP, NOAA-20 and NOAA-21 separately, and automate daily ingestion from the FIRMS API.

**Tools.** Python, pandas, NumPy, scikit-learn (gradient boosting); HTML, CSS and JavaScript (no external libraries) for the web app; GitHub Pages for hosting; Natural Earth for the coastline.

## NASA Data
NASA FIRMS (Fire Information for Resource Management System) active-fire archive data for Australia, 2023–2025:
- MODIS Collection 6.1 active fire product (Terra and Aqua, 1 km), 874,876 records.
- VIIRS 375 m active fire product from NOAA-20 (JPSS-1), 4,089,598 records.
Both were used to learn the MODIS/VIIRS conversion factors, to validate them on 2025, and as the data behind every view in the app.

## Space Agency Partner & Other Data
- Natural Earth 1:110m country outlines (public domain) for the Australia map.
- No other external data.

## Use of Artificial Intelligence
We used Claude (Anthropic), an AI assistant, to help draft and debug the Python pipeline and the web app code, to suggest the validation approach (train on 2023–2024, test on 2025) and the machine-learning model setup and to help write the documentation, slide text and this submission. All numbers were produced by running the code on NASA FIRMS data. The slide deck contains screenshots of our own app and no AI-generated images. AI assistance is also acknowledged in the README, in the code header and in the app footer.
