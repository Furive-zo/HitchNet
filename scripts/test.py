#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from datetime import datetime
import time
import json
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.collate import collate_fn
from utils.loss import hitch_loss
from utils.angle import wrap_rad_torch

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="HitchNet evaluation script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_csv", action="store_true")
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def count_parameters(model):
    """Return total number of trainable parameters (million scale)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def main():
    args = parse_args()

    # ============================
    # 1) Config & Device
    # ============================
    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    dset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    test_cfg = cfg.get("test", {})

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    out_dir = exp_cfg.get("output_dir", os.path.join("ckpts", exp_name))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ============================
    # 2) Dataset (test split)
    # ============================
    num_workers = args.num_workers or dset_cfg.get("num_workers", 4)
    batch_size = test_cfg.get("batch_size", cfg["train"].get("batch_size", 8))

    test_dataset = HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split="test",
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        pcd_max_points=dset_cfg.get("pcd_max_points", 1000),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"[INFO] HitchDataset split=test, frames={len(test_dataset)}")

    # ============================
    # 3) Load model & checkpoint
    # ============================
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # 2가지 저장 형식 대비
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model.eval()

    # ============================
    # 4) Parameter Count
    # ============================
    mparams = count_parameters(model)
    print(f"[INFO] Model parameters: {mparams:.3f} M")

    # ============================
    # 5) Test loop
    # ============================
    angle_errs = []
    theta_pred_list = []
    theta_true_list = []
    infer_times = []

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="[Test]")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)

            start_t = time.time()
            pred = model(batch)
            infer_times.append(time.time() - start_t)

            gt = batch["gt"]
            loss = hitch_loss(pred, gt)
            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # angle
            cos_p, sin_p = pred[:, 1], pred[:, 0]
            cos_g, sin_g = gt[:, 1], gt[:, 0]

            theta_p = torch.atan2(sin_p, cos_p)
            theta_g = torch.atan2(sin_g, cos_g)

            err_deg = wrap_rad_torch(theta_p - theta_g) * 180.0 / np.pi

            angle_errs.append(err_deg.cpu())
            theta_pred_list.append(theta_p.cpu())
            theta_true_list.append(theta_g.cpu())

    avg_loss = total_loss / max(n_batches, 1)

    # ============================
    # 6) Metrics
    # ============================
    angle_errs = torch.cat(angle_errs)
    abs_err = torch.abs(angle_errs)

    rmse = torch.sqrt(torch.mean(angle_errs ** 2)).item()
    mae = torch.mean(abs_err).item()
    max_err = abs_err.max().item()

    p95 = float(np.percentile(abs_err.numpy(), 95))
    p99 = float(np.percentile(abs_err.numpy(), 99))

    theta_pred = torch.cat(theta_pred_list)
    theta_true = torch.cat(theta_true_list)

    # R2
    ss_res = torch.sum((theta_true - theta_pred) ** 2).item()
    mean_true = torch.mean(theta_true).item()
    ss_tot = torch.sum((theta_true - mean_true) ** 2).item()
    R2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # latency
    avg_infer_ms = 1000.0 * np.mean(infer_times)
    fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

    # ============================
    # 7) Print summary
    # ============================
    print("=========================================")
    print(f"[TEST] Loss={avg_loss:.6f}")
    print(f"[TEST] RMSE={rmse:.3f}° | MAE={mae:.3f}° | MaxErr={max_err:.3f}°")
    print(f"[TEST] p95={p95:.3f}° | p99={p99:.3f}° | R²={R2:.3f}")
    print(f"[TEST] Infer={avg_infer_ms:.2f} ms/step ({fps:.1f} FPS)")
    print(f"[TEST] Parameters={mparams:.3f} M")
    print("=========================================")

    # ============================
    # 8) Save metrics JSON
    # ============================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics = {
        "timestamp": timestamp,
        "config": args.config,
        "checkpoint": args.ckpt,
        "loss": float(avg_loss),
        "RMSE_deg": float(rmse),
        "MAE_deg": float(mae),
        "MaxErr_deg": float(max_err),
        "p95_deg": float(p95),
        "p99_deg": float(p99),
        "R2": float(R2),
        "infer_ms": float(avg_infer_ms),
        "fps": float(fps),
        "Params_M": float(mparams),
    }

    out_dir = cfg.get("experiment", {}).get("output_dir", "ckpts")
    json_path = os.path.join(out_dir, f"test_metrics_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Test metrics saved → {json_path}")

    # ============================
    # 9) Optionally save per-frame CSV
    # ============================
    if args.save_csv:
        csv_path = os.path.join(out_dir, f"test_errors_{timestamp}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["theta_true_deg", "theta_pred_deg", "err_deg"])

            theta_true_deg = theta_true * 180.0 / np.pi
            theta_pred_deg = theta_pred * 180.0 / np.pi

            for tg, tp, e in zip(theta_true_deg.numpy(),
                                 theta_pred_deg.numpy(),
                                 angle_errs.numpy()):
                writer.writerow([float(tg), float(tp), float(e)])

        print(f"[INFO] Per-frame errors saved → {csv_path}")


if __name__ == "__main__":
    main()
