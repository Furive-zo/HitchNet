#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from models import build_model
from utils.collate import collate_fn
from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from utils.loss import hitch_loss


def parse_args():
    p = argparse.ArgumentParser(description="Deep CORAL-DG training (source-only)")
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


def configure_runtime():
    # Optional speed/compat toggles via environment variables.
    if os.environ.get("HITCHNET_DISABLE_CUDNN", "0") == "1":
        torch.backends.cudnn.enabled = False
        print("[INFO] cuDNN disabled by HITCHNET_DISABLE_CUDNN=1")
    if os.environ.get("HITCHNET_DETERMINISTIC", "1") == "0":
        torch.backends.cudnn.deterministic = False
    if os.environ.get("HITCHNET_CUDNN_BENCHMARK", "0") == "1":
        torch.backends.cudnn.benchmark = True
    if os.environ.get("HITCHNET_TF32", "0") == "1":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


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


def angle_metrics(pred, gt):
    cos_p, sin_p = pred[:, 0], pred[:, 1]
    cos_g, sin_g = gt[:, 0], gt[:, 1]
    theta_p = torch.atan2(sin_p, cos_p)
    theta_g = torch.atan2(sin_g, cos_g)
    err = (theta_p - theta_g + np.pi) % (2 * np.pi) - np.pi
    err_deg = err * 180.0 / np.pi
    rmse = torch.sqrt(torch.mean(err_deg**2)).item()
    mae = torch.mean(torch.abs(err_deg)).item()
    return rmse, mae


def _covariance(x: torch.Tensor, eps: float = 1e-6):
    n, d = x.shape
    xc = x - x.mean(dim=0, keepdim=True)
    c = (xc.t() @ xc) / max(n - 1, 1)
    c = c + eps * torch.eye(d, device=x.device, dtype=x.dtype)
    return c


def coral_dg_loss(
    feat: torch.Tensor,
    domain_ids: torch.Tensor,
    min_samples_per_domain: int = 16,
    feat_zscore: bool = True,
):
    """
    DG-CORAL (domain-to-mean):
      L = sum_e || C_e - C_bar ||_F^2 / (4 d^2)
    where C_e is covariance of features from source domain e in a mini-batch.
    """
    feat = feat.float()
    if feat_zscore:
        mu = feat.mean(dim=0, keepdim=True)
        sd = feat.std(dim=0, keepdim=True).clamp_min(1e-6)
        feat = (feat - mu) / sd

    unique_ids = torch.unique(domain_ids)
    covs = []
    used_domains = 0
    for did in unique_ids:
        idx = torch.where(domain_ids == did)[0]
        if idx.numel() < int(min_samples_per_domain):
            continue
        covs.append(_covariance(feat[idx]))
        used_domains += 1

    if len(covs) < 2:
        zero = torch.zeros((), device=feat.device, dtype=feat.dtype)
        return zero, used_domains

    c_bar = torch.stack(covs, dim=0).mean(dim=0)
    d = feat.shape[1]
    loss = torch.zeros((), device=feat.device, dtype=feat.dtype)
    for c in covs:
        diff = c - c_bar
        loss = loss + (diff * diff).sum()
    loss = loss / (4.0 * d * d)
    return loss, used_domains


def build_dataset(dset_cfg, split, train_cfg):
    return HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split=split,
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        trailer_type=dset_cfg.get("name", "charger"),
        normalize_xy=train_cfg.get("normalize_xy", False),
        bev_add_xy=train_cfg.get("bev_add_xy", False),
        bev_add_orient=train_cfg.get("bev_add_orient", False),
        bev_use_hmax=train_cfg.get("bev_use_hmax", True),
        bev_use_dlog=train_cfg.get("bev_use_dlog", True),
        occ_binary=train_cfg.get("occ_binary", False),
        add_observed_mask=train_cfg.get("add_observed_mask", False),
        observed_bins=int(train_cfg.get("observed_bins", 360)),
        observed_margin=float(train_cfg.get("observed_margin", 0.0)),
        joint_shift_x=float(train_cfg.get("joint_shift_x", 0.0)),
        # DG baseline: no geometric augmentation here by default
        aug_rotate_prob=0.0,
        occ_prob=0.0,
        trans_prob=0.0,
        trans_max_m=0.0,
    )


class DomainOffsetDataset(torch.utils.data.Dataset):
    def __init__(self, base, domain_offset: int):
        self.base = base
        self.domain_offset = int(domain_offset)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if "domain_id" in item:
            item = dict(item)
            item["domain_id"] = int(item["domain_id"]) + self.domain_offset
        return item


def main():
    args = parse_args()
    cfg = load_config(args.config)
    raw_exp = _load_yaml(args.config)

    exp_name = cfg["experiment"].get("name", "coral_dg_experiment")
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    configure_runtime()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Seed set to: {seed}")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    dset_cfg = cfg["dataset"]

    # Optional multi-source setting for DG:
    # source_dataset_configs: [datasets/dummy.yaml, datasets/temporary.yaml]
    src_cfg_paths = raw_exp.get("source_dataset_configs", None)
    source_dsets = []
    if src_cfg_paths:
        for p in src_cfg_paths:
            ap = _resolve_cfg_path(args.config, p)
            source_dsets.append(_load_yaml(ap)["dataset"])
    else:
        source_dsets = [dset_cfg]

    epochs = int(train_cfg.get("epochs", 50))
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    use_amp = bool(train_cfg.get("amp", True))
    num_workers = args.num_workers if args.num_workers is not None else int(dset_cfg.get("num_workers", 8))

    coral_w = float(train_cfg.get("coral_w", 0.05))
    coral_min_samples_per_domain = int(train_cfg.get("coral_min_samples_per_domain", 16))
    coral_feat_zscore = bool(train_cfg.get("coral_feat_zscore", True))
    huber_delta_deg = float(train_cfg.get("huber_delta_deg", 0.5))
    loss_alpha = float(train_cfg.get("loss_alpha", 0.7))
    loss_beta = float(train_cfg.get("loss_beta", 0.3))
    weight_factor = float(train_cfg.get("weight_factor", 3.0))

    train_sets = []
    val_sets = []
    dom_offset = 0
    for ds in source_dsets:
        tr = build_dataset(ds, "train", train_cfg)
        va = build_dataset(ds, "val", train_cfg)
        train_sets.append(DomainOffsetDataset(tr, dom_offset))
        val_sets.append(DomainOffsetDataset(va, dom_offset))
        dom_offset += len(getattr(tr, "domain_name_to_id", {}))

    train_set = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    val_set = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = build_model(model_cfg).to(device)
    if model_cfg.get("name") != "bev_resnet_coral":
        raise ValueError(f"Use model name `bev_resnet_coral` for CORAL-DG training, got: {model_cfg.get('name')}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    out_dir = os.path.join("ckpts", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pth")
    last_path = os.path.join(out_dir, "last.pth")
    best_metrics_path = os.path.join(out_dir, "best_metrics.json")
    log_csv = os.path.join(out_dir, "metrics_log.csv")

    best_rmse = float("inf")
    with open(log_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            ["epoch", "train_loss", "train_reg_loss", "train_coral_loss", "val_loss", "val_rmse_deg", "val_mae_deg"]
        )

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        reg_loss_sum = 0.0
        coral_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"[Train {epoch+1}/{epochs}]")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                pred, feat = model.forward_with_feat(batch)
                reg_loss = hitch_loss(
                    pred,
                    batch["gt"],
                    alpha=loss_alpha,
                    beta=loss_beta,
                    weight_factor=weight_factor,
                    huber_delta_deg=huber_delta_deg,
                )
                closs, used_domains = coral_dg_loss(
                    feat=feat,
                    domain_ids=batch["domain_id"],
                    min_samples_per_domain=coral_min_samples_per_domain,
                    feat_zscore=coral_feat_zscore,
                )
                loss = reg_loss + coral_w * closs

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            reg_loss_sum += reg_loss.item()
            coral_loss_sum += closs.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "n_dom": int(used_domains)})

        scheduler.step()

        train_loss = train_loss_sum / len(train_loader)
        train_reg = reg_loss_sum / len(train_loader)
        train_coral = coral_loss_sum / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        preds, gts = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val {epoch+1}/{epochs}]"):
                batch = move_batch_to_device(batch, device)
                with autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    pred = model(batch)
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
        val_loss = val_loss_sum / len(val_loader)

        print(
            f"[Epoch {epoch+1}] train={train_loss:.6f} (reg={train_reg:.6f}, coral={train_coral:.6f}) "
            f"| val={val_loss:.6f} | RMSE={val_rmse:.3f}° | MAE={val_mae:.3f}°"
        )

        with open(log_csv, "a", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([epoch + 1, train_loss, train_reg, train_coral, val_loss, val_rmse, val_mae])

        ckpt = {
            "model": model.state_dict(),
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
                "checkpoint_path": best_path,
            }
            with open(best_metrics_path, "w") as f:
                json.dump(best_metrics, f, indent=2)
            print(f"[INFO] New best saved to {best_path}")

    print("[INFO] CORAL-DG training finished.")


if __name__ == "__main__":
    main()
