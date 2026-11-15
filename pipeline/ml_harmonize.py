"""
Fire Pulse — machine-learning harmonisation experiment
(code drafted with help from Claude, an AI assistant by Anthropic; see README "Use of AI")

Question: can a model that sees MORE than the raw VIIRS count predict MODIS better
than the per-square ratio used by harmonize.py?

For every 0.5° square and day, the model gets VIIRS features:
  detections, day vs night split, high-confidence share, fire radiative power (sum, max),
  time of year, location, and the square's ratio k learned on the training years.
It predicts the MODIS detection count for that square-day (Poisson loss, since the
target is a count). Learn on years before --split, test on the rest, scored exactly
like harmonize.py so the comparison is fair.

Usage
  python pipeline/ml_harmonize.py --modis <csvs...> --viirs <csvs...> --split 2025-01-01
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

GRID = 0.5
LAMBDA = 200.0
COLS = ["latitude", "longitude", "acq_date", "confidence", "frp", "type", "daynight"]


def read(paths, sensor):
    df = pd.concat([pd.read_csv(p, usecols=COLS) for p in paths], ignore_index=True)
    df = df[df["type"] == 0]
    if sensor == "MODIS":
        df = df[pd.to_numeric(df["confidence"], errors="coerce") >= 30]
    else:
        df = df[df["confidence"].astype(str).isin(["n", "h"])]
    df["date"] = pd.to_datetime(df["acq_date"])
    df["gy"] = np.floor(df["latitude"] / GRID).astype(int)
    df["gx"] = np.floor(df["longitude"] / GRID).astype(int)
    return df


def cell_days(m, v):
    """One row per square-day with VIIRS features and the MODIS target."""
    v = v.assign(day=(v.daynight == "D").astype(int), night=(v.daynight == "N").astype(int),
                 high=(v.confidence.astype(str) == "h").astype(int))
    fv = (v.groupby(["gy", "gx", "date"])
            .agg(v_n=("frp", "size"), v_day=("day", "sum"), v_night=("night", "sum"),
                 v_high=("high", "sum"), frp_sum=("frp", "sum"), frp_max=("frp", "max"))
            .reset_index())
    fm = m.groupby(["gy", "gx", "date"]).size().rename("m").reset_index()
    t = fv.merge(fm, on=["gy", "gx", "date"], how="outer").fillna(0)
    t["night_share"] = np.where(t.v_n > 0, t.v_night / t.v_n.clip(lower=1), 0)
    t["high_share"] = np.where(t.v_n > 0, t.v_high / t.v_n.clip(lower=1), 0)
    t["frp_mean"] = np.where(t.v_n > 0, t.frp_sum / t.v_n.clip(lower=1), 0)
    doy = t.date.dt.dayofyear
    t["doy_sin"], t["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    t["lat"], t["lon"] = (t.gy + 0.5) * GRID, (t.gx + 0.5) * GRID
    return t


FEATURES = ["v_n", "v_day", "v_night", "night_share", "high_share", "frp_sum", "frp_max",
            "frp_mean", "doy_sin", "doy_cos", "lat", "lon", "k_cell"]
LABELS = {"v_n": "VIIRS detections", "v_day": "VIIRS day detections", "v_night": "VIIRS night detections",
          "night_share": "Night share", "high_share": "High-confidence share", "frp_sum": "Fire power (sum)",
          "frp_max": "Fire power (max)", "frp_mean": "Fire power (mean)", "doy_sin": "Time of year (sin)",
          "doy_cos": "Time of year (cos)", "lat": "Latitude", "lon": "Longitude", "k_cell": "Square's ratio k"}


def make_model():
    # settings fixed before looking at the test year; not tuned on it
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=600,
                                         max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15,
                                         n_iter_no_change=30, random_state=42)


def score(test, pred):
    """Same two metrics as harmonize.py: median per-square error, mean weekly national error."""
    t = test.assign(p=pred)
    c = t.groupby(["gy", "gx"])[["m", "p"]].sum()
    c = c[c.m >= 20]
    w = t.groupby(pd.Grouper(key="date", freq="W"))[["m", "p"]].sum()
    w = w[w.m > 0]
    return {"median_cell_error_pct": round(float(np.median(np.abs(c.p - c.m) / c.m)) * 100, 1),
            "weekly_national_error_pct": round(float(np.mean(np.abs(w.p - w.m) / w.m)) * 100, 1),
            "test_cells": int(len(c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", nargs="+", required=True)
    ap.add_argument("--viirs", nargs="+", required=True)
    ap.add_argument("--split", default="2025-01-01")
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--export", action="store_true",
                    help="refit on all years and add ML values to output/firepulse_data.json for the app")
    ap.add_argument("--robust-split", default=None,
                    help="optional second split (e.g. 2024-01-01) to check the result is not luck")
    a = ap.parse_args()
    split = pd.Timestamp(a.split)

    print("Loading and building square-day table")
    t = cell_days(read(a.modis, "MODIS"), read(a.viirs, "VIIRS"))
    tr, te = t[t.date < split].copy(), t[t.date >= split].copy()

    # per-square ratio learned on training years only (also the baseline method)
    K = tr.m.sum() / tr.v_n.sum()
    agg = tr.groupby(["gy", "gx"])[["m", "v_n"]].sum()
    k = (agg.m + LAMBDA * K) / (agg.v_n + LAMBDA)
    for d in (tr, te):
        d["k_cell"] = k.reindex(pd.MultiIndex.from_frame(d[["gy", "gx"]])).fillna(K).to_numpy()
    print(f"  {len(tr):,} training square-days, {len(te):,} test square-days, national K {K:.4f}")

    # train only where VIIRS saw something; square-days with no VIIRS get 0 from every method
    fit = tr[tr.v_n > 0]
    model = make_model()
    model.fit(fit[FEATURES], fit.m)
    print(f"  model trained: {model.n_iter_} boosting rounds")

    pred_ml = np.where(te.v_n > 0, model.predict(te[FEATURES]), 0.0)
    results = {
        "identity (no harmonisation)": score(te, te.v_n.to_numpy(dtype=float)),
        "national factor": score(te, te.v_n.to_numpy() * K),
        "cell factor + shrinkage (Fire Pulse)": score(te, te.v_n.to_numpy() * te.k_cell.to_numpy()),
        "gradient boosting (ML)": score(te, pred_ml),
    }
    for name, r in results.items():
        print(f"  {name:40s} weekly {r['weekly_national_error_pct']:6.1f}%   per-square {r['median_cell_error_pct']:6.1f}%")

    # what the model relies on: permutation importance on a test sample
    sample = te[te.v_n > 0].sample(min(40000, int((te.v_n > 0).sum())), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample.m, n_repeats=3, random_state=0,
                                 scoring="neg_mean_poisson_deviance")
    order = np.argsort(-imp.importances_mean)
    importance = [{"feature": LABELS[FEATURES[i]], "importance": round(float(imp.importances_mean[i]), 4)}
                  for i in order]
    print("  top features:", ", ".join(f"{x['feature']} ({x['importance']})" for x in importance[:5]))

    out = {"split": str(split.date()), "train_square_days": int(len(fit)), "boosting_rounds": int(model.n_iter_),
           "results": results, "importance": importance}

    if a.robust_split:
        s2 = pd.Timestamp(a.robust_split)
        tr2, te2 = t[t.date < s2].copy(), t[t.date >= s2].copy()
        K2 = tr2.m.sum() / tr2.v_n.sum()
        g2 = tr2.groupby(["gy", "gx"])[["m", "v_n"]].sum()
        k2 = (g2.m + LAMBDA * K2) / (g2.v_n + LAMBDA)
        for d in (tr2, te2):
            d["k_cell"] = k2.reindex(pd.MultiIndex.from_frame(d[["gy", "gx"]])).fillna(K2).to_numpy()
        m2 = make_model().fit(tr2[tr2.v_n > 0][FEATURES], tr2[tr2.v_n > 0].m)
        p2 = np.where(te2.v_n > 0, m2.predict(te2[FEATURES]), 0.0)
        out["robustness"] = {"split": str(s2.date()),
                             "cell factor + shrinkage (Fire Pulse)": score(te2, te2.v_n.to_numpy() * te2.k_cell.to_numpy()),
                             "gradient boosting (ML)": score(te2, p2)}
        print("  robustness (train before", s2.date(), "):", out["robustness"])

    if a.export:
        # product model: k_cell and the model refit on every year, like harmonize.py's export
        Kf = t.m.sum() / t.v_n.sum()
        gf = t.groupby(["gy", "gx"])[["m", "v_n"]].sum()
        kf = (gf.m + LAMBDA * Kf) / (gf.v_n + LAMBDA)
        t["k_cell"] = kf.reindex(pd.MultiIndex.from_frame(t[["gy", "gx"]])).fillna(Kf).to_numpy()
        full = make_model().fit(t[t.v_n > 0][FEATURES], t[t.v_n > 0].m)
        t["h_ml"] = np.where(t.v_n > 0, full.predict(t[FEATURES]), 0.0)
        data_path = a.out / "firepulse_data.json"
        data = json.loads(data_path.read_text())
        d0 = pd.Timestamp(data["meta"]["start"])
        lut = {(int(r.gy), int(r.gx), int((r.date - d0).days)): round(float(r.h_ml), 2)
               for r in t[["gy", "gx", "date", "h_ml"]].itertuples(index=False)}
        for key, c in data["cells"].items():
            gy, gx = map(int, key.split("_"))
            c["s"] = [row[:3] + [lut.get((gy, gx, row[0]), 0.0)] for row in c["s"]]
        data["ml"] = out
        data_path.write_text(json.dumps(data, separators=(",", ":")))
        print(f"  added ML values to {data_path}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "ml_validation.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out / 'ml_validation.json'}")


if __name__ == "__main__":
    main()
