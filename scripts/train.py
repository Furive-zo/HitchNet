#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from datetime import datetime

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

from models import build_model     

def parse_args():
    parser = argparse.ArgumentParser(description="HitchNet training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config yaml (e.g. configs/experiments/e1_charger_hitchnet.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override num_workers in dataset config (optional)",
    )
    return parser.parse_args()


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def main():
    args = parse_args()

    # ============================
    # 1) Config load
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
    # 2) Dataset & DataLoader
    # ============================
    num_workers = args.num_workers if args.num_workers is not None else dset_cfg.get("num_workers", 4)
    batch_size = train_cfg.get("batch_size", 8)

    # HitchDataset은 하나의 split_json에서 split="train"/"val"을 나누는 구조
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
    # 3) Model, Optimizer, Scheduler
    # ============================
    model = build_model(model_cfg)   # 내부에서 HitchNet(model_cfg) 생성된다고 가정
    model = model.to(device)

    lr = train_cfg.get("lr", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    epochs = train_cfg.get("epochs", 50)
    use_amp = train_cfg.get("amp", True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ============================
    # 4) Training loop
    # ============================
    best_val_loss = float("inf")
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        # ------------------------
        # Train one epoch
        # ------------------------
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[Train {epoch+1}/{epochs}]")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(batch)              # (B,2)
                gt = batch["gt"]                # (B,2)
                loss = hitch_loss(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.6f}")

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_loss = 0.0

        # accumulate angle errors
        angle_errs = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Val {epoch+1}/{epochs}]")
            for batch in pbar:
                batch = move_batch_to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(batch)      # (B,2)
                    gt = batch["gt"]        # (B,2)
                    loss = hitch_loss(pred, gt)

                val_loss += loss.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                # ---- angle RMSE accumulation ----
                cos_p, sin_p = pred[:,0], pred[:,1]
                cos_g, sin_g = gt[:,0], gt[:,1]

                theta_p = torch.atan2(sin_p, cos_p)
                theta_g = torch.atan2(sin_g, cos_g)

                err = theta_p - theta_g                          # rad
                err = (err + torch.pi) % (2 * torch.pi) - torch.pi # wrap
                err_deg = err * 180.0 / torch.pi

                angle_errs.append(err_deg.cpu())

        val_loss /= len(val_loader)

        # ----- Compute RMSE and MAE -----
        angle_errs = torch.cat(angle_errs)         # (total_val_samples,)
        rmse = torch.sqrt(torch.mean(angle_errs**2)).item()
        mae = torch.mean(torch.abs(angle_errs)).item()

        print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}  |  RMSE: {rmse:.3f}°  |  MAE: {mae:.3f}°")

        scheduler.step()

        # ------------------------
        # Checkpoint 저장
        # ------------------------
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": cfg,
        }

        # last ckpt
        last_path = os.path.join(out_dir, "last.pth")
        torch.save(ckpt, last_path)

        # best ckpt
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(out_dir, "best.pth")
            torch.save(ckpt, best_path)
            print(f"[INFO] New best val loss: {best_val_loss:.6f}, saved to {best_path}")

    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
