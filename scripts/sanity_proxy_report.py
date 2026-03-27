#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, glob, math, os
from collections import defaultdict
import numpy as np
from scripts.proxy_eval.kinematics import compute_proxy_timeseries

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logs_glob", default="logs_proxy/*/*.csv")
    p.add_argument("--summary_overall", default="proxy_results/all_models_kappa/summary_overall.csv")
    p.add_argument("--summary_by_bin", default="proxy_results/all_models_kappa/summary_by_kappa_bin.csv")
    p.add_argument("--out_md", default="proxy_results/all_models_kappa/sanity_report.md")
    p.add_argument("--L2", type=float, default=1.804)
    p.add_argument("--eps", type=float, default=0.704)
    p.add_argument("--v_min", type=float, default=1.0)
    p.add_argument("--bias_rad", type=float, default=0.05)
    p.add_argument("--sample_file", default="")
    return p.parse_args()

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def load_log(path):
    cols = {"t": [], "v": [], "yaw_rate": [], "gamma_gt": [], "gamma_pred": []}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k in cols:
                cols[k].append(float(row[k]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}

def choose_sample(files, pref):
    if pref:
        return pref
    for fp in files:
        if "highway_5.csv" in fp:
            return fp
    return files[0]

def main():
    a = parse_args()
    files = sorted(glob.glob(a.logs_glob))
    if not files:
        raise FileNotFoundError(a.logs_glob)

    s_overall = read_rows(a.summary_overall)
    s_bin = read_rows(a.summary_by_bin)
    sample = choose_sample(files, a.sample_file)

    # unit range
    gmin, gmax = float("inf"), -float("inf")
    for fp in files:
        with open(fp, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                g0 = float(row["gamma_gt"]); g1 = float(row["gamma_pred"])
                gmin = min(gmin, g0, g1); gmax = max(gmax, g0, g1)

    # zero + bias
    log = load_log(sample)
    zlog = {k: v.copy() for k, v in log.items()}
    zlog["gamma_pred"] = zlog["gamma_gt"].copy()
    oz = compute_proxy_timeseries(zlog, a.L2, a.eps, 1, a.v_min)
    eyz = np.abs(oz["ey"][oz["valid_mask"]])

    blog = {k: v.copy() for k, v in log.items()}
    blog["gamma_pred"] = blog["gamma_gt"] + a.bias_rad
    ob = compute_proxy_timeseries(blog, a.L2, a.eps, 1, a.v_min)
    eyb = np.abs(ob["ey"][ob["valid_mask"]])
    expected = abs(a.L2 * math.sin(a.bias_rad))

    # sign check
    op = compute_proxy_timeseries(log, a.L2, a.eps, 1, a.v_min)
    om = compute_proxy_timeseries(log, a.L2, a.eps, -1, a.v_min)
    eyp = np.abs(op["ey"][op["valid_mask"]]); eym = np.abs(om["ey"][om["valid_mask"]])

    # n_valid==0
    seq_rows = [r for r in s_overall if r["sequence"] != "__ALL__"]
    bad = [(r["method"], r["sequence"]) for r in seq_rows if int(float(r["n_valid"])) == 0]

    # trend
    bm = defaultdict(list)
    for r in s_bin:
        if r["sequence"] == "__ALL__":
            bm[r["method"]].append(r)

    lines = []
    lines += ["# Proxy Sanity Report", ""]
    lines += [f"- files: **{len(files)}**", f"- sample_file: `{sample}`", ""]
    lines += ["## 1) Definition / Unit / Sign"]
    lines += [f"- gamma range: [{gmin:.4f}, {gmax:.4f}] rad"]
    lines += [f"- rad-plausible (|max| < 1.5*pi): **{abs(gmax) < 1.5*math.pi}**"]
    lines += [f"- mean(|ey|), gamma_sign=+1: {np.mean(eyp):.6f} m"]
    lines += [f"- mean(|ey|), gamma_sign=-1: {np.mean(eym):.6f} m", ""]
    lines += ["## 2) Physical Scale Sanity"]
    lines += [f"- zero test mean(|ey|): {np.mean(eyz):.8f} m"]
    lines += [f"- zero test max(|ey|): {np.max(eyz):.8f} m"]
    lines += [f"- bias test mean(|ey|): {np.mean(eyb):.6f} m"]
    lines += [f"- expected L2*sin(b): {expected:.6f} m"]
    lines += [f"- ratio obs/exp: {(np.mean(eyb)/expected):.4f}", ""]
    lines += ["## 3) Valid Sample Coverage"]
    lines += [f"- n_valid==0 rows: {len(bad)} / {len(seq_rows)}"]
    for m, s in bad[:10]:
        lines += [f"  - {m} / {s}"]
    lines += ["", "## 4) Curvature Trend"]
    for m, rows in sorted(bm.items()):
        rows = sorted(rows, key=lambda r: float(r["kappa_bin_center"]))
        frows = [r for r in rows if r["p95_abs_ey"] not in ("", "nan", "NaN")]
        if len(frows) >= 2:
            x = np.array([float(r["kappa_bin_center"]) for r in frows], dtype=float)
            y = np.array([float(r["p95_abs_ey"]) for r in frows], dtype=float)
            corr = float(np.corrcoef(x, y)[0,1])
        else:
            corr = float("nan")
        lines += [f"- {m}: finite_bins={len(frows)}/{len(rows)}, corr(|kappa|,p95)={corr:.4f}"]

    os.makedirs(os.path.dirname(a.out_md), exist_ok=True)
    with open(a.out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] Saved: {a.out_md}")

if __name__ == "__main__":
    main()
