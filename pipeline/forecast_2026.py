"""
Fire Pulse — 2026 forecasting contest
(code drafted with help from Claude, an AI assistant by Anthropic; see README "Use of AI")

Step 1 (predict):  python pipeline/forecast_2026.py predict
    Uses ONLY 2023–2025 MODIS detections (output/firepulse_data.json) to predict
    MODIS detections for every 0.5° square and month of 2026 with six methods,
    then freezes the predictions (timestamp + SHA-256) before any 2026 data is seen.

Step 2 (score):    python pipeline/forecast_2026.py score --actual data/raw/modis_2026_Australia.csv
    Cleans the real 2026 MODIS file the same way, scores every method on the months
    that have data, and ranks them.

Target: MODIS detections (vegetation fires, confidence >= 30). Using MODIS itself as the
target means the "truth" in 2026 needs no harmonisation model at all.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

G = 0.5
OUT = Path("output")
REGIONS = {
    "All Australia": None,
    "Top End (NT)": [-16, 129, -10.5, 138],
    "Kimberley (WA)": [-19, 121, -13.5, 129],
    "Cape York (QLD)": [-16, 141, -10, 146],
    "Central Australia": [-27, 129, -20, 138],
    "South East QLD": [-28.7, 151, -26, 153.6],
    "South West WA": [-35.2, 114.9, -30, 119.5],
}
METHODS = {
    "climatology": "Climatology: average of the same month in 2023–2025",
    "median": "Robust climatology: median of the same month in 2023–2025",
    "persistence": "Persistence: same month last year (2025)",
    "ensemble": "Ensemble: average of climatology and persistence",
    "ensemble_enso": "Ensemble + El Niño analog: adds 2023 (an El Niño year) for Jun–Dec, as BoM declared El Niño in June 2026",
    "ml_gbm": "Machine learning: gradient boosting (Poisson) on each square's history, month and location",
}


def in_region(lat0, lon0, box):
    if box is None:
        return np.ones(len(lat0), bool)
    return (lat0 + G > box[0]) & (lat0 < box[2]) & (lon0 + G > box[1]) & (lon0 < box[3])


def history_cube():
    """Square × year × month MODIS counts for 2023–2025 from the app data."""
    d = json.loads((OUT / "firepulse_data.json").read_text())
    start = pd.Timestamp(d["meta"]["start"])
    keys = sorted(d["cells"])
    idx = {k: i for i, k in enumerate(keys)}
    cube = np.zeros((len(keys), 3, 12))
    for k, c in d["cells"].items():
        for r in c["s"]:
            dt = start + pd.Timedelta(days=r[0])
            if 2023 <= dt.year <= 2025:
                cube[idx[k], dt.year - 2023, dt.month - 1] += r[1]
    gy = np.array([int(k.split("_")[0]) for k in keys])
    gx = np.array([int(k.split("_")[1]) for k in keys])
    return keys, gy, gx, cube


def ml_forecast(gy, gx, cube):
    """Train on 'predict year t from years before t' (t = 2024, 2025), then predict 2026."""
    def rows(target_year_idx):
        lag1 = cube[:, target_year_idx - 1, :]
        lag2 = cube[:, target_year_idx - 2, :] if target_year_idx >= 2 else np.full_like(lag1, np.nan)
        prev_tot = lag1.sum(axis=1, keepdims=True).repeat(12, axis=1)
        n, m = lag1.shape
        month = np.tile(np.arange(12), (n, 1))
        lat = np.repeat(((gy + 0.5) * G)[:, None], 12, axis=1)
        lon = np.repeat(((gx + 0.5) * G)[:, None], 12, axis=1)
        X = np.stack([lag1, lag2, np.nanmean(np.stack([lag1, lag2]), axis=0), prev_tot,
                      np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), lat, lon], axis=-1)
        return X.reshape(-1, X.shape[-1])
    Xtr = np.vstack([rows(1), rows(2)])
    ytr = np.concatenate([cube[:, 1, :].ravel(), cube[:, 2, :].ravel()])
    model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=42)
    model.fit(Xtr, ytr)
    # predict 2026: lag1 = 2025, lag2 = 2024
    ext = np.concatenate([cube, np.zeros((cube.shape[0], 1, 12))], axis=1)
    lag1, lag2 = ext[:, 2, :], ext[:, 1, :]
    prev_tot = lag1.sum(axis=1, keepdims=True).repeat(12, axis=1)
    n = lag1.shape[0]
    month = np.tile(np.arange(12), (n, 1))
    lat = np.repeat(((gy + 0.5) * G)[:, None], 12, axis=1)
    lon = np.repeat(((gx + 0.5) * G)[:, None], 12, axis=1)
    X = np.stack([lag1, lag2, (lag1 + lag2) / 2, prev_tot, np.sin(2 * np.pi * month / 12),
                  np.cos(2 * np.pi * month / 12), lat, lon], axis=-1).reshape(-1, 8)
    return model.predict(X).reshape(n, 12)


def predict():
    keys, gy, gx, cube = history_cube()
    y23, y24, y25 = cube[:, 0], cube[:, 1], cube[:, 2]
    clim = cube.mean(axis=1)
    preds = {
        "climatology": clim,
        "median": np.median(cube, axis=1),
        "persistence": y25,
        "ensemble": (clim + y25) / 2,
        "ensemble_enso": np.where(np.arange(12) >= 5, (clim + y25 + y23) / 3, (clim + y25) / 2),
        "ml_gbm": ml_forecast(gy, gx, cube),
    }
    lat0, lon0 = gy * G, gx * G
    rows = []
    for meth, P in preds.items():
        for r, box in REGIONS.items():
            mask = in_region(lat0, lon0, box)
            for m in range(12):
                rows.append({"method": meth, "region": r, "month": m + 1, "prediction": round(float(P[mask, m].sum()), 1)})
    reg = pd.DataFrame(rows)
    cell = pd.DataFrame({"cell": np.repeat(keys, 12), "month": np.tile(np.arange(1, 13), len(keys))})
    for meth, P in preds.items():
        cell[meth] = P.ravel().round(2)
    OUT.mkdir(exist_ok=True)
    reg.to_csv(OUT / "predictions_2026_regions.csv", index=False)
    cell.to_csv(OUT / "predictions_2026_cells.csv.gz", index=False)
    h = hashlib.sha256()
    for f in ("predictions_2026_regions.csv", "predictions_2026_cells.csv.gz"):
        h.update((OUT / f).read_bytes())
    lock = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_on": "MODIS 2023–2025 only; no 2026 data used", "sha256": h.hexdigest(),
            "methods": METHODS}
    (OUT / "predictions_2026_LOCK.json").write_text(json.dumps(lock, indent=2))
    print(json.dumps(lock, indent=2))
    print(reg[reg.region == "All Australia"].pivot(index="month", columns="method", values="prediction").round(0))


def score(actual_paths):
    lock = json.loads((OUT / "predictions_2026_LOCK.json").read_text())
    h = hashlib.sha256()
    for f in ("predictions_2026_regions.csv", "predictions_2026_cells.csv.gz"):
        h.update((OUT / f).read_bytes())
    assert h.hexdigest() == lock["sha256"], "predictions changed after they were frozen"
    cols = ["latitude", "longitude", "acq_date", "confidence"]
    a = pd.concat([pd.read_csv(p) for p in actual_paths], ignore_index=True)
    # FIRMS zips hold an archive file (standard quality, has `type`) and an NRT file
    # (latest ~3 months, no `type`). Keep vegetation fires from the archive and every NRT row.
    if "type" in a.columns:
        n_nrt = int(a["type"].isna().sum())
        a = a[(a["type"] == 0) | a["type"].isna()]
        type_note = f"archive rows: vegetation fires only (type 0); {n_nrt:,} NRT rows kept (no type column)"
    else:
        type_note = "no type column (NRT data): all detections kept"
    a = a.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"]) if "acq_time" in a.columns else a
    a = a[pd.to_numeric(a["confidence"], errors="coerce") >= 30]
    a["date"] = pd.to_datetime(a["acq_date"])
    a = a[a.date.dt.year == 2026]
    last = a.date.max()
    full_months = [m for m in range(1, 13) if pd.Timestamp(2026, m, 1) + pd.offsets.MonthEnd(0) <= last]
    a["gy"] = np.floor(a.latitude / G).astype(int)
    a["gx"] = np.floor(a.longitude / G).astype(int)
    a["month"] = a.date.dt.month
    a = a[a.month.isin(full_months)]
    reg = pd.read_csv(OUT / "predictions_2026_regions.csv")
    reg = reg[reg.month.isin(full_months)]
    truth = []
    for r, box in REGIONS.items():
        mask = in_region(a.gy.to_numpy() * G, a.gx.to_numpy() * G, box)
        t = a[mask].groupby("month").size().reindex(full_months, fill_value=0)
        truth += [{"region": r, "month": m, "actual": int(v)} for m, v in t.items()]
    truth = pd.DataFrame(truth)
    j = reg.merge(truth, on=["region", "month"])
    cells = pd.read_csv(OUT / "predictions_2026_cells.csv.gz")
    cells = cells[cells.month.isin(full_months)]
    act_c = a.assign(cell=a.gy.astype(str) + "_" + a.gx.astype(str)).groupby("cell").size()
    results = []
    for meth in METHODS:
        q = j[j.method == meth]
        nat = q[q.region == "All Australia"]
        sub = q[q.region != "All Australia"]
        c = cells.groupby("cell")[meth].sum()
        cc = pd.concat([act_c.rename("actual"), c.rename("pred")], axis=1).fillna(0)
        cc = cc[cc.actual >= 20]
        results.append({
            "method": meth, "description": METHODS[meth],
            "national_monthly_error_pct": round(float(np.mean(np.abs(nat.prediction - nat.actual) / nat.actual.clip(lower=1))) * 100, 1),
            "national_total_error_pct": round(float(abs(nat.prediction.sum() - nat.actual.sum()) / nat.actual.sum()) * 100, 1),
            "regional_monthly_median_error_pct": round(float(np.median(np.abs(sub.prediction - sub.actual) / sub.actual.clip(lower=1))) * 100, 1),
            "per_square_median_error_pct": round(float(np.median(np.abs(cc.pred - cc.actual) / cc.actual)) * 100, 1),
        })
    res = pd.DataFrame(results)
    base = res.set_index("method").loc["climatology"]
    for col in ("national_monthly_error_pct", "regional_monthly_median_error_pct", "per_square_median_error_pct"):
        res["skill_" + col.replace("_error_pct", "")] = (1 - res[col] / base[col]).round(3)
    res["rank_score"] = res[["national_monthly_error_pct", "regional_monthly_median_error_pct",
                             "per_square_median_error_pct"]].rank().mean(axis=1)
    res = res.sort_values("rank_score")
    out = {"scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "predictions_frozen_at_utc": lock["frozen_at_utc"], "hash_verified": True,
           "months_scored": full_months, "last_detection_date": str(last.date()), "data_note": type_note,
           "actual_2026_detections_scored": int(len(a)), "results": res.to_dict(orient="records"),
           "monthly": j.to_dict(orient="records")}
    (OUT / "forecast_2026_results.json").write_text(json.dumps(out, indent=2))
    print(res.drop(columns="description").to_string(index=False))
    print("months scored:", full_months, "| last detection:", last.date(), "|", type_note)


# ---------------------------------------------------------------- step 3: forecast 2027 with the winner
def forecast_2027(actual_paths):
    """Apply the 2026 winner (median of the latest three years, per square and month) to 2027.
    2026 counts come from the real MODIS 2026 files; months without a full 2026 record
    (Nov, Dec) fall back to the latest three complete years. 80% ranges are calibrated on the
    winner's real 2026 errors (region-month log ratios), the way forecasters size intervals."""
    res = json.loads((OUT / "forecast_2026_results.json").read_text())
    winner = res["results"][0]["method"]
    keys, gy, gx, cube = history_cube()
    idx = {k: i for i, k in enumerate(keys)}
    a = pd.concat([pd.read_csv(p) for p in actual_paths], ignore_index=True)
    if "type" in a.columns:
        a = a[(a["type"] == 0) | a["type"].isna()]
    a = a[pd.to_numeric(a["confidence"], errors="coerce") >= 30]
    a = a.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    a["date"] = pd.to_datetime(a["acq_date"])
    full = res["months_scored"]
    a = a[(a.date.dt.year == 2026) & a.date.dt.month.isin(full)]
    a["cell"] = np.floor(a.latitude / G).astype(int).astype(str) + "_" + np.floor(a.longitude / G).astype(int).astype(str)
    y26 = np.full((len(keys), 12), np.nan)
    y26[:, [m - 1 for m in full]] = 0
    extra = {}
    for (c, m), n in a.groupby(["cell", a.date.dt.month]).size().items():
        if c in idx:
            y26[idx[c], m - 1] = n
        else:
            extra[(c, m)] = n  # squares with no 2023–2025 history
    f27 = np.zeros((len(keys), 12))
    for m in range(12):
        hist = [cube[:, 1, m], cube[:, 2, m], y26[:, m]] if not np.isnan(y26[0, m]) else [cube[:, 0, m], cube[:, 1, m], cube[:, 2, m]]
        f27[:, m] = np.median(np.stack(hist), axis=0)
    # interval width from the winner's 2026 errors across regions and months
    mo = pd.DataFrame(res["monthly"])
    w = mo[(mo.method == winner) & (mo.region != "All Australia")]
    lr = np.log((w.actual + 1) / (w.prediction + 1))
    bias, sd = float(lr.mean()), float(lr.std(ddof=1))
    z = 1.2816
    lat0, lon0 = gy * G, gx * G
    reg = []
    for r, box in REGIONS.items():
        mask = in_region(lat0, lon0, box)
        for m in range(12):
            p = float(f27[mask, m].sum())
            act = None if np.isnan(y26[0, m]) else float(np.nansum(y26[mask, m]))
            reg.append({"region": r, "month": m + 1, "forecast_2027": round(p, 1),
                        "low_80": round(float(np.expm1(np.log1p(p) - z * sd)), 1),
                        "high_80": round(float(np.expm1(np.log1p(p) + z * sd)), 1),
                        "actual_2026": act})
    pd.DataFrame(reg).to_csv(OUT / "forecast_2027_regions.csv", index=False)
    out = {"winner": winner, "rule": "median of the latest three years for each 0.5° square and month",
           "years_used": "2024, 2025, 2026 for Jan–Oct; 2023–2025 for Nov–Dec (2026 not complete)",
           "interval": {"level": 0.8, "log_sd_from_2026_errors": round(sd, 3), "mean_log_bias_2026": round(bias, 3)},
           "made_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "forecast_2027_meta.json").write_text(json.dumps(out, indent=2))
    return keys, f27, y26, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["predict", "score", "forecast2027"])
    ap.add_argument("--actual", nargs="+")
    a = ap.parse_args()
    if a.step == "predict":
        predict()
    elif a.step == "score":
        score(a.actual)
    else:
        print(json.dumps(forecast_2027(a.actual)[3], indent=2))
