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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.collate import collate_fn
from utils.loss import hitch_loss
from utils.angle import wrap_rad_torch, wrap_deg_torch

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="HitchNet training script")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def main():
    args = parse_args()

    # ============================
    # 1) Config
    # ============================
    cfg = load_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    dset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    exp_name = exp_cfg.get("name", Path(args.config).stem)
    out_dir = exp_cfg.get("output_dir", os.path.join("ckpts", exp_name))
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ============================
    # 2) Dataset & Loader
    # ============================
    num_workers = args.num_workers or dset_cfg.get("num_workers", 4)
    batch_size = train_cfg.get("batch_size", 8)

    root = dset_cfg["root"]
    split_json = dset_cfg["split"]

    temporal_window = dset_cfg.get("temporal_window", 20)
    micro_seq_length = dset_cfg.get("micro_seq_length", 10)
    pcd_max_points = dset_cfg.get("pcd_max_points", 1000)

    train_dataset = HitchDataset(
        root=root,
        split_json=split_json,
        split="train",
        temporal_window=temporal_window,
        micro_seq_length=micro_seq_length,
        pcd_max_points=pcd_max_points,
    )
    val_dataset = HitchDataset(
        root=root,
        split_json=split_json,
        split="val",
        temporal_window=temporal_window,
        micro_seq_length=micro_seq_length,
        pcd_max_points=pcd_max_points,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ============================
    # 3) Model
    # ============================
    model = build_model(model_cfg).to(device)

    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    epochs = train_cfg.get("epochs", 50)
    use_amp = train_cfg.get("amp", True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ============================
    # 4) Train loop
    # ============================
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ============================
        # ---- TRAIN ----
        # ============================
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[Train {epoch+1}/{epochs}]")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(batch)         # (B,2)
                gt = batch["gt"]            # (B,2)
                loss = hitch_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.6f}")

        # ============================
        # ---- VALIDATION ----
        # ============================
        model.eval()
        val_loss_sum = 0.0
        angle_errs = []
        infer_times = []
        all_theta_pred = []
        all_theta_true = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Val {epoch+1}/{epochs}]")
            for batch in pbar:
                batch = move_batch_to_device(batch, device)

                start_t = time.time()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch)
                    gt = batch["gt"]
                    loss = hitch_loss(pred, gt)
                infer_times.append(time.time() - start_t)

                val_loss_sum += loss.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                # angle error accumulation
                # pred / gt 는 (sin, cos) 순서라고 가정
                cos_p, sin_p = pred[:, 1], pred[:, 0]
                cos_g, sin_g = gt[:, 1], gt[:, 0]

                theta_p = torch.atan2(sin_p, cos_p)  # [B], rad
                theta_g = torch.atan2(sin_g, cos_g)  # [B], rad

                err_rad = wrap_rad_torch(theta_p - theta_g)
                err_deg = err_rad * 180.0 / np.pi

                angle_errs.append(err_deg.cpu())
                all_theta_pred.append(theta_p.cpu())
                all_theta_true.append(theta_g.cpu())

        val_loss = val_loss_sum / len(val_loader)
        angle_errs = torch.cat(angle_errs)          # (N,)
        theta_pred = torch.cat(all_theta_pred)      # (N,) rad
        theta_true = torch.cat(all_theta_true)      # (N,) rad

        # Metrics (deg)
        rmse = torch.sqrt(torch.mean(angle_errs ** 2)).item()
        mae = torch.mean(torch.abs(angle_errs)).item()

        # p95, p99 (deg)
        abs_err = torch.abs(angle_errs).numpy()
        p95 = float(np.percentile(abs_err, 95))
        p99 = float(np.percentile(abs_err, 99))

        # R² (rad 기준)
        ss_res = torch.sum((theta_true - theta_pred) ** 2).item()
        mean_true = torch.mean(theta_true).item()
        ss_tot = torch.sum((theta_true - mean_true) ** 2).item()
        R2 = 1.0 - ss_res / (ss_tot + 1e-12)

        # inference time
        avg_infer_ms = 1000.0 * np.mean(infer_times)
        fps = 1000.0 / avg_infer_ms if avg_infer_ms > 0 else 0.0

        print(
            f"[Epoch {epoch+1}] "
            f"Val Loss={val_loss:.6f} | RMSE={rmse:.3f}° | MAE={mae:.3f}° | "
            f"R²={R2:.3f} | p95={p95:.2f}° | p99={p99:.2f}° | "
            f"Infer={avg_infer_ms:.2f}ms ({fps:.1f} FPS)"
        )

        scheduler.step()

        # ============================
        # Save last ckpt
        # ============================
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(out_dir, "last.pth"))

        # ============================
        # Save best checkpoint + log
        # ============================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(out_dir, "best.pth")
            torch.save(ckpt, best_path)

            # ---- Best metrics dict ----
            best_info = {
                "epoch": epoch + 1,
                "val_loss": float(val_loss),
                "RMSE_deg": float(rmse),
                "MAE_deg": float(mae),
                "R2": float(R2),
                "p95_deg": float(p95),
                "p99_deg": float(p99),
                "infer_ms": float(avg_infer_ms),
                "fps": float(fps),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "exp_name": exp_name,
                    "batch_size": batch_size,
                    "temporal_window": temporal_window,
                    "micro_seq_length": micro_seq_length,
                    "pcd_max_points": pcd_max_points,
                    "lr": lr,
                    "epochs": epochs,
                    "model": model_cfg,
                },
                "checkpoint_path": best_path,
            }

            # ---- 1) JSON (best only) ----
            json_path = os.path.join(out_dir, "best_metrics.json")
            with open(json_path, "w") as f:
                json.dump(best_info, f, indent=2)

            # ---- 2) CSV (append) ----
            csv_path = os.path.join(out_dir, "metrics_log.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(best_info.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(best_info)

            print(f"[INFO] New best saved to {best_path}")
            print(f"[INFO] Best metrics saved → {json_path}")
            print(f"[INFO] Metrics appended → {csv_path}")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
