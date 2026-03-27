#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
import yaml
from tqdm import tqdm

from models import build_model
from utils.collate import collate_fn
from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.loss import hitch_loss


def parse_args():
    p = argparse.ArgumentParser(description="EMA self-training UDA")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_cfg_path(exp_cfg_path, rel_or_abs):
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    base_dir = os.path.dirname(exp_cfg_path)
    return os.path.normpath(os.path.join(base_dir, "..", rel_or_abs))


def _build_dataset(dset_cfg, split, temporal_window, micro_seq_length, joint_shift_x):
    return HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split=split,
        temporal_window=temporal_window,
        micro_seq_length=micro_seq_length,
        trailer_type=dset_cfg.get("name", "charger"),
        normalize_xy=False,
        bev_add_xy=False,
        bev_add_orient=False,
        bev_use_hmax=True,
        bev_use_dlog=True,
        occ_binary=False,
        add_observed_mask=False,
        aug_rotate_deg=0.0,
        aug_rotate_prob=0.0,
        occ_prob=0.0,
        trans_prob=0.0,
        trans_max_m=0.0,
        joint_shift_x=joint_shift_x,
    )


def angle_metrics(pred, gt):
    cos_p, sin_p = pred[:, 0], pred[:, 1]
    cos_g, sin_g = gt[:, 0], gt[:, 1]
    theta_p = torch.atan2(sin_p, cos_p)
    theta_g = torch.atan2(sin_g, cos_g)
    err = (theta_p - theta_g + math.pi) % (2 * math.pi) - math.pi
    err_deg = err * 180.0 / math.pi
    rmse = torch.sqrt(torch.mean(err_deg**2)).item()
    mae = torch.mean(torch.abs(err_deg)).item()
    return rmse, mae


def wrap_deg(x):
    return torch.remainder(x + 180.0, 360.0) - 180.0


def ema_update(teacher, student, m):
    with torch.no_grad():
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=(1.0 - m))


def make_weak_aug_batch(batch, noise_std):
    out = dict(batch)
    if noise_std > 0.0 and "bev" in batch:
        bev = batch["bev"]
        out["bev"] = torch.clamp(bev + torch.randn_like(bev) * noise_std, 0.0, 1.0)
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config)
    raw_exp = _load_yaml(args.config)

    target_dataset_cfg_rel = raw_exp.get("target_dataset_config", "datasets/dummy.yaml")
    target_dataset_cfg_path = _resolve_cfg_path(args.config, target_dataset_cfg_rel)
    target_dataset_cfg = _load_yaml(target_dataset_cfg_path)["dataset"]

    exp_name = cfg["experiment"].get("name", "selftrain_experiment")
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Seed set to: {seed}")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    source_dataset_cfg = cfg["dataset"]

    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    use_amp = bool(train_cfg.get("amp", True))
    num_workers = args.num_workers if args.num_workers is not None else int(source_dataset_cfg.get("num_workers", 8))

    # losses
    huber_delta_deg = float(train_cfg.get("huber_delta_deg", 0.5))
    loss_alpha = float(train_cfg.get("loss_alpha", 0.7))
    loss_beta = float(train_cfg.get("loss_beta", 0.3))
    weight_factor = float(train_cfg.get("weight_factor", 3.0))
    pseudo_weight_factor = float(train_cfg.get("st_pseudo_weight_factor", 0.0))

    # self-training
    target_use_ratio = float(train_cfg.get("target_use_ratio", 1.0))
    st_warmup_epochs = int(train_cfg.get("st_warmup_epochs", 5))
    st_lambda_max = float(train_cfg.get("st_lambda_max", 0.5))
    st_conf_tau_start = float(train_cfg.get("st_conf_tau_start_deg", 8.0))
    st_conf_tau_end = float(train_cfg.get("st_conf_tau_end_deg", 3.0))
    st_ema_momentum = float(train_cfg.get("st_ema_momentum", 0.999))
    st_noise_std = float(train_cfg.get("st_noise_std", 0.01))

    joint_shift_x = float(train_cfg.get("joint_shift_x", 0.0))
    temporal_window = source_dataset_cfg.get("temporal_window", 20)
    micro_seq_length = source_dataset_cfg.get("micro_seq_length", 10)

    src_train = _build_dataset(source_dataset_cfg, "train", temporal_window, micro_seq_length, joint_shift_x)
    src_val = _build_dataset(source_dataset_cfg, "val", temporal_window, micro_seq_length, joint_shift_x)
    tgt_train = _build_dataset(target_dataset_cfg, "train", temporal_window, micro_seq_length, joint_shift_x)
    if target_use_ratio < 1.0:
        n_total = len(tgt_train)
        n_keep = max(1, int(round(n_total * max(0.0, target_use_ratio))))
        idx = np.random.RandomState(seed).choice(n_total, size=n_keep, replace=False)
        tgt_train = Subset(tgt_train, idx.tolist())
        print(f"[INFO] Target subset enabled: {n_keep}/{n_total} ({target_use_ratio:.3f})")

    src_train_loader = DataLoader(
        src_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    src_val_loader = DataLoader(
        src_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    tgt_train_loader = DataLoader(
        tgt_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    student = build_model(model_cfg).to(device)
    teacher = copy.deepcopy(student).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    out_dir = os.path.join("ckpts", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pth")
    last_path = os.path.join(out_dir, "last.pth")
    best_metrics_path = os.path.join(out_dir, "best_metrics.json")
    log_csv = os.path.join(out_dir, "metrics_log.csv")

    best_rmse = float("inf")
    total_steps = epochs * len(src_train_loader)
    global_step = 0

    with open(log_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_src_loss",
                "train_tgt_loss",
                "pseudo_keep_ratio",
                "val_loss",
                "val_rmse_deg",
                "val_mae_deg",
            ]
        )

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        tgt_iter = iter(tgt_train_loader)
        train_loss_sum = 0.0
        src_loss_sum = 0.0
        tgt_loss_sum = 0.0
        keep_ratio_sum = 0.0

        pbar = tqdm(src_train_loader, desc=f"[Train {epoch+1}/{epochs}]")
        for src_batch in pbar:
            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_train_loader)
                tgt_batch = next(tgt_iter)

            src_batch = move_batch_to_device(src_batch, device)
            tgt_batch = move_batch_to_device(tgt_batch, device)
            tgt_batch_aug = make_weak_aug_batch(tgt_batch, st_noise_std)

            p = global_step / max(total_steps - 1, 1)
            global_step += 1
            tau = st_conf_tau_start + (st_conf_tau_end - st_conf_tau_start) * p
            if epoch < st_warmup_epochs:
                st_lambda = 0.0
            else:
                q = (epoch - st_warmup_epochs + 1) / max(epochs - st_warmup_epochs, 1)
                st_lambda = st_lambda_max * min(max(q, 0.0), 1.0)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                src_pred = student(src_batch)
                src_loss = hitch_loss(
                    src_pred,
                    src_batch["gt"],
                    alpha=loss_alpha,
                    beta=loss_beta,
                    weight_factor=weight_factor,
                    huber_delta_deg=huber_delta_deg,
                )

                with torch.no_grad():
                    t1 = teacher(tgt_batch)
                    t2 = teacher(tgt_batch_aug)
                    th1 = torch.atan2(t1[:, 1], t1[:, 0]) * 180.0 / math.pi
                    th2 = torch.atan2(t2[:, 1], t2[:, 0]) * 180.0 / math.pi
                    dth = torch.abs(wrap_deg(th1 - th2))
                    keep = dth < tau
                    pseudo = t1.detach()

                keep_ratio = keep.float().mean().item()
                if st_lambda > 0.0 and bool(torch.any(keep)):
                    tgt_pred = student(tgt_batch)
                    tgt_loss_all = hitch_loss(
                        tgt_pred,
                        pseudo,
                        alpha=1.0,
                        beta=0.0,
                        weight_factor=pseudo_weight_factor,
                        huber_delta_deg=huber_delta_deg,
                    )
                    # compute masked per-sample huber (simple fallback by scaling batch loss)
                    # to keep implementation lightweight with current hitch_loss interface.
                    tgt_loss = tgt_loss_all * keep.float().mean()
                else:
                    tgt_loss = torch.zeros((), device=device, dtype=src_loss.dtype)

                loss = src_loss + st_lambda * tgt_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_update(teacher, student, st_ema_momentum)

            train_loss_sum += loss.item()
            src_loss_sum += src_loss.item()
            tgt_loss_sum += tgt_loss.item()
            keep_ratio_sum += keep_ratio
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "keep": f"{keep_ratio:.2f}", "tau": f"{tau:.1f}"})

        scheduler.step()

        train_loss = train_loss_sum / len(src_train_loader)
        train_src = src_loss_sum / len(src_train_loader)
        train_tgt = tgt_loss_sum / len(src_train_loader)
        keep_ratio_epoch = keep_ratio_sum / len(src_train_loader)

        student.eval()
        val_loss_sum = 0.0
        preds = []
        gts = []
        with torch.no_grad():
            for batch in tqdm(src_val_loader, desc=f"[Val {epoch+1}/{epochs}]"):
                batch = move_batch_to_device(batch, device)
                with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    pred = student(batch)
                    loss = hitch_loss(
                        pred,
                        batch["gt"],
                        alpha=loss_alpha,
                        beta=loss_beta,
                        weight_factor=weight_factor,
                        huber_delta_deg=huber_delta_deg,
                    )
                val_loss_sum += loss.item()
                preds.append(pred.float().cpu())
                gts.append(batch["gt"].float().cpu())

        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0)
        val_rmse, val_mae = angle_metrics(preds, gts)
        val_loss = val_loss_sum / len(src_val_loader)

        print(
            f"[Epoch {epoch+1}] train={train_loss:.6f} (src={train_src:.6f}, tgt={train_tgt:.6f}, keep={keep_ratio_epoch:.3f}) "
            f"| val={val_loss:.6f} | RMSE={val_rmse:.3f}° | MAE={val_mae:.3f}°"
        )

        with open(log_csv, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch + 1, train_loss, train_src, train_tgt, keep_ratio_epoch, val_loss, val_rmse, val_mae])

        ckpt = {
            "model": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "config": cfg,
        }
        torch.save(ckpt, last_path)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(ckpt, best_path)
            best_metrics = {
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "RMSE_deg": val_rmse,
                "MAE_deg": val_mae,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": cfg,
                "target_dataset": target_dataset_cfg,
                "checkpoint_path": best_path,
            }
            with open(best_metrics_path, "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"[INFO] New best saved to {best_path}")

    print("[INFO] Self-training finished.")


if __name__ == "__main__":
    main()
