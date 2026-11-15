"""
Fire Pulse  (code drafted with help from Claude, an AI assistant by Anthropic; see README "Use of AI")
 — MODIS / VIIRS active-fire harmonisation pipeline
NASA Space Apps Challenge 2026 · "Harmonization of MODIS and VIIRS Hot Spots"

Steps
  1. Extract   : read NASA FIRMS archive CSVs (MODIS C6 + VIIRS S-NPP 375 m)
  2. Clean     : keep vegetation fires (type 0) with nominal/high confidence
  3. Grid      : snap every detection to a 0.5° cell, aggregate per cell per day
  4. Calibrate : learn a cell-level VIIRS→MODIS scaling factor on a training
                 window, shrunk toward the national factor where data is thin
  5. Validate  : test on later, unseen months against two baselines
  6. Export    : compact JSON consumed by the web app

Usage
  python pipeline/harmonize.py --modis data/raw/<modis>.csv --viirs data/raw/<viirs>.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GRID = 0.5                 # degrees
SOURCE = "NASA FIRMS: MODIS C6.1 (Terra+Aqua) and VIIRS 375 m (NOAA-20)"
LAMBDA = 200.0             # shrinkage strength, in VIIRS detections
MODIS_MIN_CONF = 30        # MODIS confidence is 0-100; <30 = "low"
VIIRS_OK_CONF = {"n", "h"}  # VIIRS confidence is l / n / h


# ---------------------------------------------------------------- 1-2 extract + clean
def load(paths, sensor: str) -> pd.DataFrame:
    """Read one or more FIRMS CSVs (e.g. one per year) for a single sensor."""
    cols = ["latitude", "longitude", "acq_date", "confidence", "frp", "type", "daynight"]
    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    df = pd.concat([pd.read_csv(p, usecols=cols) for p in paths], ignore_index=True)
    n0 = len(df)
    df = df[df["type"] == 0]  # drop volcanoes, static land sources, offshore
    if sensor == "MODIS":
        df = df[pd.to_numeric(df["confidence"], errors="coerce") >= MODIS_MIN_CONF]
    else:
        df = df[df["confidence"].astype(str).isin(VIIRS_OK_CONF)]
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])
    print(f"  {sensor}: {n0:,} rows -> {len(df):,} after cleaning")
    df["date"] = pd.to_datetime(df["acq_date"])
    return df[["latitude", "longitude", "date", "frp", "daynight"]]


# ---------------------------------------------------------------- 3 grid
def grid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gy"] = np.floor(df["latitude"] / GRID).astype(int)
    df["gx"] = np.floor(df["longitude"] / GRID).astype(int)
    return (df.groupby(["gy", "gx", "date"])
              .agg(n=("frp", "size"), frp=("frp", "sum"))
              .reset_index())


# ---------------------------------------------------------------- 4 calibrate
def fit_factors(m: pd.DataFrame, v: pd.DataFrame, col: str):
    """k_cell = (M_cell + λ·K) / (V_cell + λ), where K = national M/V ratio.
    Sparse cells borrow strength from the national factor; busy cells use their own."""
    K = m[col].sum() / v[col].sum()
    cm = m.groupby(["gy", "gx"])[col].sum()
    cv = v.groupby(["gy", "gx"])[col].sum()
    both = pd.concat([cm, cv], axis=1, keys=["m", "v"]).fillna(0)
    lam = LAMBDA if col == "n" else LAMBDA * (v[col].sum() / v["n"].sum())
    k = (both["m"] + lam * K) / (both["v"] + lam)
    return K, k


def apply(v: pd.DataFrame, K: float, k: pd.Series, col: str) -> pd.Series:
    idx = pd.MultiIndex.from_frame(v[["gy", "gx"]])
    return v[col].to_numpy() * k.reindex(idx).fillna(K).to_numpy()


# ---------------------------------------------------------------- 5 validate
def validate(m: pd.DataFrame, v: pd.DataFrame, split: pd.Timestamp) -> dict:
    out = {}
    for col, label in [("n", "detections"), ("frp", "frp")]:
        mtr, vtr = m[m.date < split], v[v.date < split]
        mte, vte = m[m.date >= split], v[v.date >= split]
        K, k = fit_factors(mtr, vtr, col)
        vte = vte.copy()
        preds = {
            "identity (no harmonisation)": vte[col].to_numpy(),
            "national factor": vte[col].to_numpy() * K,
            "cell factor + shrinkage (Fire Pulse)": apply(vte, K, k, col),
        }
        obs_cell = mte.groupby(["gy", "gx"])[col].sum()
        obs_wk = mte.groupby(pd.Grouper(key="date", freq="W"))[col].sum()
        res = {}
        for name, p in preds.items():
            vte["p"] = p
            pc = vte.groupby(["gy", "gx"])["p"].sum()
            j = pd.concat([obs_cell, pc], axis=1, keys=["o", "p"]).fillna(0)
            j = j[j.o >= 20]  # cells with meaningful MODIS activity
            cell_err = float(np.median(np.abs(j.p - j.o) / j.o))
            pw = vte.groupby(pd.Grouper(key="date", freq="W"))["p"].sum()
            w = pd.concat([obs_wk, pw], axis=1, keys=["o", "p"]).fillna(0)
            w = w[w.o > 0]
            wk_err = float(np.mean(np.abs(w.p - w.o) / w.o))
            res[name] = {"median_cell_error_pct": round(cell_err * 100, 1),
                         "weekly_national_error_pct": round(wk_err * 100, 1)}
        out[label] = {"train_until": str(split.date()), "national_factor": round(float(K), 4),
                      "test_cells": int(len(j)), "results": res}
    return out


# ---------------------------------------------------------------- 6 export
def yearly(m_raw, v_raw):
    """Per-year K, plus day and night K, to check stability."""
    out = []
    for y in sorted(set(m_raw.date.dt.year) & set(v_raw.date.dt.year)):
        my, vy = m_raw[m_raw.date.dt.year == y], v_raw[v_raw.date.dt.year == y]
        row = {"year": int(y), "modis": int(len(my)), "viirs": int(len(vy)),
               "k": round(len(my) / len(vy), 4)}
        for dn, key in (("D", "k_day"), ("N", "k_night")):
            nv = int((vy.daynight == dn).sum())
            row[key] = round(int((my.daynight == dn).sum()) / nv, 4) if nv else None
        out.append(row)
    return out


def export(m, v, outdir: Path, outline, validation, years):
    Kn, kn = fit_factors(m, v, "n")
    d0 = min(m.date.min(), v.date.min())
    d1 = max(m.date.max(), v.date.max())
    j = pd.merge(m, v, on=["gy", "gx", "date"], how="outer",
                 suffixes=("_m", "_v")).fillna(0)
    j["d"] = (j.date - d0).dt.days
    cells = {}
    for (gy, gx), g in j.groupby(["gy", "gx"]):
        key = f"{gy}_{gx}"
        rows = g.sort_values("d")[["d", "n_m", "n_v", "frp_m", "frp_v"]].to_numpy()
        cells[key] = {
            "kn": round(float(kn.get((gy, gx), Kn)), 4),
            "s": [[int(r[0]), int(r[1]), int(r[2])] for r in rows],
        }
    payload = {
        "meta": {
            "title": "Fire Pulse",
            "source": SOURCE,
            "region": "Australia",
            "start": str(d0.date()), "end": str(d1.date()),
            "days": int((d1 - d0).days + 1),
            "grid_deg": GRID,
            "national_factor_detections": round(float(Kn), 4),
            "totals": {"modis_n": int(m.n.sum()), "viirs_n": int(v.n.sum()),
                       "modis_frp": int(m.frp.sum()), "viirs_frp": int(v.frp.sum())},
            "filters": {"type": 0, "modis_confidence_min": MODIS_MIN_CONF,
                        "viirs_confidence": sorted(VIIRS_OK_CONF)},
            "shrinkage_lambda": LAMBDA,
        },
        "validation": validation,
        "yearly": years,
        "outline": outline,
        "cells": cells,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "firepulse_data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    print(f"  wrote {len(cells):,} cells -> {outdir/'firepulse_data.json'}")


def load_outline(path: Path | None):
    if path and path.exists():
        return json.loads(path.read_text())
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", required=True, type=Path, nargs="+")
    ap.add_argument("--viirs", required=True, type=Path, nargs="+")
    ap.add_argument("--split", default=None, help="train/test split date (default: 40%% of window)")
    ap.add_argument("--outline", type=Path, default=Path("data/outline.json"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--source", default=None, help="label for the data source")
    a = ap.parse_args()
    global SOURCE
    if a.source:
        SOURCE = a.source

    print("1-2 Extract + clean")
    m_raw, v_raw = load(a.modis, "MODIS"), load(a.viirs, "VIIRS")
    print("3 Grid")
    m, v = grid(m_raw), grid(v_raw)
    d0, d1 = m.date.min(), m.date.max()
    split = pd.Timestamp(a.split) if a.split else d0 + (d1 - d0) * 0.4
    split = split.normalize()
    print(f"4-5 Calibrate on < {split.date()}, validate on >= {split.date()}")
    val = validate(m, v, split)
    for metric, r in val.items():
        print(f"  [{metric}] national factor {r['national_factor']}")
        for name, s in r["results"].items():
            print(f"    {name:40s} cell err {s['median_cell_error_pct']:5.1f}%   "
                  f"weekly err {s['weekly_national_error_pct']:5.1f}%")
    print("6 Export (factors refit on the full window)")
    yr = yearly(m_raw, v_raw)
    for r in yr:
        print(f"  {r['year']}: K {r['k']}  day {r['k_day']}  night {r['k_night']}")
    export(m, v, a.out, load_outline(a.outline), val, yr)


if __name__ == "__main__":
    main()
