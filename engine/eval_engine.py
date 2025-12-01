# engine/eval_engine.py

import torch
from tqdm import tqdm
import numpy as np
from .metrics import compute_angle_metrics


class Evaluator:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def run(self, loader):
        self.model.eval()

        preds = []
        gts = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="[Evaluating]"):
                pcd = batch["pcd"].to(self.device)
                imu = batch["imu"].to(self.device)
                vel = batch["vel"].to(self.device)
                steer = batch["steer"].to(self.device)
                target = batch["target"].to(self.device)

                pred = self.model(pcd, imu, vel, steer)  # (B,2)

                preds.append(pred.cpu().numpy())
                gts.append(target.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        gts = np.concatenate(gts, axis=0)

        return compute_angle_metrics(preds, gts)
