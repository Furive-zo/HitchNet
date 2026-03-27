#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import yaml
import os

from fvcore.nn import FlopCountAnalysis, parameter_count_table

from utils.load_config import load_config
from utils.load_dataset import HitchDataset
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Compute model GFLOPs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--trailer_type", type=str, default=None)
    return parser.parse_args()


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def main():

    args = parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    dset_cfg = cfg["dataset"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[INFO] building model...")
    model = build_model(model_cfg).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print("[INFO] checkpoint loaded")

    model.eval()

    print("[INFO] loading dataset sample...")
    dataset = HitchDataset(
        root=dset_cfg["root"],
        split_json=dset_cfg["split"],
        split="test",
        temporal_window=dset_cfg.get("temporal_window", 20),
        micro_seq_length=dset_cfg.get("micro_seq_length", 10),
        trailer_type=dset_cfg.get("name", "charger"),
    )

    sample = dataset[0]

    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v

    batch.pop("gt", None)

    batch = move_to_device(batch, device)

    print("[INFO] computing FLOPs...")

    flops = FlopCountAnalysis(model, batch)
    total_flops = flops.total()

    params_m = count_parameters(model)

    print("====================================")
    print(f"Params : {params_m:.3f} M")
    print(f"FLOPs  : {total_flops/1e9:.3f} GFLOPs / inference")
    print("====================================")

    print("\nDetailed parameter table:")
    print(parameter_count_table(model))


if __name__ == "__main__":
    main()