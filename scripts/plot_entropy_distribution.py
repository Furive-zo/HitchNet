#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import numpy as np
import open3d as o3d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
})


TRAILERS = [
    ("charger_trailer", "Short-Tall trailer", "#1D4E89"),
    ("temporary_trailer", "Compact trailer", "#6BAED6"),
    ("dummy_trailer", "Long-Flat trailer", "#2C7FB8"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot entropy distributions for 3 trailer datasets.")
    parser.add_argument("--dataset_root", type=str, default="datasets/LI-HAE/dataset")
    parser.add_argument("--split_dir", type=str, default="datasets/LI-HAE/splits")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--joint_shift_x", type=float, default=0.8)
    parser.add_argument("--x_min", type=float, default=-2.5)
    parser.add_argument("--x_max", type=float, default=2.5)
    parser.add_argument("--y_min", type=float, default=-4.0)
    parser.add_argument("--y_max", type=float, default=0.5)
    parser.add_argument("--crop", action="store_true", help="Crop points to ROI before entropy computation.")
    parser.add_argument("--entropy_res", type=float, default=0.1, help="Grid resolution (m) for entropy.")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames per trailer (0 = all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--divergence_bins", type=int, default=100, help="Histogram bins for JS divergence.")
    parser.add_argument("--min_entropy", type=float, default=0.1, help="Drop samples with entropy below this value.")
    parser.add_argument("--from_json", type=str, default=None, help="Load cached entropy values from JSON and only render plot.")
    parser.add_argument("--out_png", type=str, default="results/entropy_distribution_compare.png")
    parser.add_argument("--out_json", type=str, default="results/entropy_distribution_compare.json")
    return parser.parse_args()


def load_split_sequences(split_path, split):
    with open(split_path, "r") as f:
        data = json.load(f)
    return data.get(split, [])


def list_frames(root, seqs):
    frames = []
    for seq in seqs:
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for d in sorted(os.listdir(seq_dir)):
            if d.startswith("frame_"):
                frames.append(os.path.join(seq_dir, d))
    return frames


def read_pcd_xy(frame_dir, joint_shift_x):
    pcd_path = os.path.join(frame_dir, "trailer_point.pcd")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts.reshape(0, 2)
    pts = pts.copy()
    pts[:, 0] += float(joint_shift_x)
    return pts[:, :2]


def compute_entropy(pts, xlim, ylim, res, crop=False):
    if pts.shape[0] == 0:
        return 0.0

    p = pts
    if crop:
        m = (
            (p[:, 0] >= xlim[0]) & (p[:, 0] <= xlim[1]) &
            (p[:, 1] >= ylim[0]) & (p[:, 1] <= ylim[1])
        )
        p = p[m]
        if p.shape[0] == 0:
            return 0.0

    x_edges = np.arange(xlim[0], xlim[1] + res, res, dtype=np.float32)
    y_edges = np.arange(ylim[0], ylim[1] + res, res, dtype=np.float32)
    if x_edges.size < 2 or y_edges.size < 2:
        return 0.0

    hist, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[x_edges, y_edges])
    occ = hist > 0
    if not np.any(occ):
        return 0.0

    occ_counts = hist[occ]
    prob = occ_counts / (occ_counts.sum() + 1e-12)
    return float(-(prob * np.log2(prob + 1e-12)).sum())


def entropy_norm_factor(xlim, ylim, res):
    x_edges = np.arange(xlim[0], xlim[1] + res, res, dtype=np.float32)
    y_edges = np.arange(ylim[0], ylim[1] + res, res, dtype=np.float32)
    n_cells = max((x_edges.size - 1) * (y_edges.size - 1), 2)
    return float(np.log2(float(n_cells)))


def js_divergence_from_samples(a, b, bins=100):
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Use smoothed KDE-based PDFs on fixed [0, 1] support to avoid binning artifacts.
    grid = np.linspace(0.0, 1.0, max(int(bins), 64), dtype=np.float64)
    dx = float(grid[1] - grid[0])
    eps = 1e-12

    def kde_pdf(samples):
        s = np.clip(samples.astype(np.float64), 0.0, 1.0)
        n = max(s.size, 1)
        std = float(np.std(s))
        bw = 1.06 * std * (n ** (-1.0 / 5.0))  # Silverman
        bw = max(bw, 0.015)  # prevent too-sharp densities
        z = (grid[:, None] - s[None, :]) / bw
        k = np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * bw)
        p = np.mean(k, axis=1)
        p = p / max(float(np.sum(p) * dx), eps)
        return p

    p = kde_pdf(a)
    q = kde_pdf(b)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2((p + eps) / (m + eps))) * dx
    kl_qm = np.sum(q * np.log2((q + eps) / (m + eps))) * dx
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.clip(js, 0.0, 1.0))


def wasserstein_1d(a, b):
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = np.sort(a.astype(np.float64))
    b = np.sort(b.astype(np.float64))
    z = np.sort(np.concatenate([a, b]))
    if z.size < 2:
        return 0.0
    cdf_a = np.searchsorted(a, z, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, z, side="right") / float(b.size)
    dz = np.diff(z)
    return float(np.sum(np.abs(cdf_a[:-1] - cdf_b[:-1]) * dz))


def collect_entropies(root, split_json, split, joint_shift_x, xlim, ylim, res, crop, max_frames, seed):
    if split == "all":
        seqs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    else:
        seqs = load_split_sequences(split_json, split)
    frames = list_frames(root, seqs)

    rng = np.random.default_rng(seed)
    if max_frames > 0 and len(frames) > max_frames:
        idx = rng.choice(len(frames), size=max_frames, replace=False)
        frames = [frames[i] for i in sorted(idx.tolist())]

    ent = []
    for fr in frames:
        pts = read_pcd_xy(fr, joint_shift_x)
        ent.append(compute_entropy(pts, xlim, ylim, res, crop=crop))
    return np.array(ent, dtype=np.float32), len(frames)


def main():
    args = parse_args()
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    h_norm = entropy_norm_factor(xlim, ylim, args.entropy_res)

    if args.from_json:
        with open(args.from_json, "r") as f:
            payload_in = json.load(f)
        results = []
        for t in payload_in.get("trailers", []):
            ent = np.array(t.get("entropy_values", []), dtype=np.float32)
            ent = ent[ent >= float(args.min_entropy)]
            norm_vals = np.array(t.get("normalized_entropy_values", []), dtype=np.float32)
            if norm_vals.size > 0:
                norm_vals = np.clip(norm_vals, 0.0, 1.0)
                # Keep filtering consistent with raw entropy threshold when available.
                if norm_vals.size == np.array(t.get("entropy_values", []), dtype=np.float32).size:
                    raw_all = np.array(t.get("entropy_values", []), dtype=np.float32)
                    norm_vals = norm_vals[raw_all >= float(args.min_entropy)]
            else:
                norm_vals = np.clip(ent / max(h_norm, 1e-8), 0.0, 1.0)
            results.append({
                "name": t["name"],
                "label": t.get("label", t["name"]),
                "color": t.get("color", "#1D4E89"),
                "entropy": ent,
                "norm_entropy": norm_vals,
                "n_frames": int(t.get("n_frames", 0)),
            })
    else:
        results = []
        for i, (folder, label, color) in enumerate(TRAILERS):
            root = os.path.join(args.dataset_root, folder)
            split = os.path.join(args.split_dir, f"{folder}_split.json")
            ent, n_frames = collect_entropies(
                root=root,
                split_json=split,
                split=args.split,
                joint_shift_x=args.joint_shift_x,
                xlim=xlim,
                ylim=ylim,
                res=args.entropy_res,
                crop=args.crop,
                max_frames=args.max_frames,
                seed=args.seed + i,
            )
            results.append({
                "name": folder,
                "label": label,
                "color": color,
                "entropy": ent[ent >= float(args.min_entropy)],
                "norm_entropy": np.clip(ent[ent >= float(args.min_entropy)] / max(h_norm, 1e-8), 0.0, 1.0),
                "n_frames": n_frames,
            })

    # Enforce panel order: short -> compact -> long
    order_map = {"charger_trailer": 0, "temporary_trailer": 1, "dummy_trailer": 2}
    results = sorted(results, key=lambda r: order_map.get(r["name"], 999))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    labels = [r["label"] for r in results]
    vals = [r["norm_entropy"] if r["norm_entropy"].size > 0 else np.array([0.0], dtype=np.float32) for r in results]
    positions = np.arange(1, len(results) + 1)

    vp = ax.violinplot(vals, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(results[i]["color"])
        body.set_edgecolor(results[i]["color"])
        body.set_alpha(0.45)

    box = ax.boxplot(
        vals,
        positions=positions,
        widths=0.20,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set(facecolor="white", edgecolor="#2a2a2a", linewidth=1.0)
    for k in ("whiskers", "caps", "medians"):
        for artist in box[k]:
            artist.set(color="#2a2a2a", linewidth=1.0)

    means = [float(np.mean(v)) for v in vals]
    ax.scatter(
        positions, means,
        marker="D", s=16,
        facecolors="white", edgecolors="#1f1f1f", linewidths=0.9,
        zorder=5,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("Normalized entropy")
    ax.set_ylim(0.0, 0.7)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Line2D([0], [0], color="#2a2a2a", lw=1.2, label="median (line)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#1f1f1f", markersize=4.5, label="mean (marker)"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", ncol=1)

    fig.tight_layout()
    out_dir = os.path.dirname(args.out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(args.out_png, dpi=220)
    plt.close(fig)

    payload = {
        "split": args.split,
        "xlim": [args.x_min, args.x_max],
        "ylim": [args.y_min, args.y_max],
        "entropy_res": args.entropy_res,
        "min_entropy": args.min_entropy,
        "entropy_norm_factor": h_norm,
        "crop": bool(args.crop),
        "trailers": [],
        "pairwise_divergence": [],
    }
    for r in results:
        e = r["entropy"]
        payload["trailers"].append({
            "name": r["name"],
            "label": r["label"],
            "color": r["color"],
            "n_frames": int(r["n_frames"]),
            "entropy_values": e.tolist(),
            "normalized_entropy_values": r["norm_entropy"].tolist(),
            "mean_entropy": float(np.mean(e)) if e.size > 0 else float("nan"),
            "std_entropy": float(np.std(e)) if e.size > 0 else float("nan"),
            "p50_entropy": float(np.percentile(e, 50)) if e.size > 0 else float("nan"),
            "p95_entropy": float(np.percentile(e, 95)) if e.size > 0 else float("nan"),
            "mean_norm_entropy": float(np.mean(r["norm_entropy"])) if e.size > 0 else float("nan"),
            "p95_norm_entropy": float(np.percentile(r["norm_entropy"], 95)) if e.size > 0 else float("nan"),
        })

    # Pairwise divergence data for table usage
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            js = js_divergence_from_samples(
                a["norm_entropy"], b["norm_entropy"], bins=int(args.divergence_bins)
            )
            wd = wasserstein_1d(a["norm_entropy"], b["norm_entropy"])
            payload["pairwise_divergence"].append({
                "pair": [a["label"], b["label"]],
                "js_divergence": js,
                "wasserstein_distance": wd,
            })

    out_json_dir = os.path.dirname(args.out_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[INFO] Saved plot: {args.out_png}")
    print(f"[INFO] Saved stats: {args.out_json}")
    for r in results:
        e = r["entropy"]
        if e.size == 0:
            print(f"[INFO] {r['name']}: no frames")
        else:
            print(f"[INFO] {r['name']}: n={r['n_frames']}, mean={np.mean(e):.3f}, p95={np.percentile(e,95):.3f}")
    for d in payload["pairwise_divergence"]:
        print(
            f"[INFO] Pair {d['pair'][0]} vs {d['pair'][1]}: "
            f"JS={d['js_divergence']:.4f}, W={d['wasserstein_distance']:.4f}"
        )


if __name__ == "__main__":
    main()
