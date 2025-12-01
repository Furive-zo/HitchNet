# engine/train_engine.py

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, scheduler, device="cuda", use_amp=True, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        self.logger = logger

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"[Train Epoch {epoch}]")

        for batch in pbar:
            # ---- Unpack ----
            pcd = batch["pcd"].to(self.device)                # (B,T,N,3)
            imu = batch["imu"].to(self.device)                # (B,T,10,3)
            vel = batch["vel"].to(self.device)                # (B,T,10,1)
            steer = batch["steer"].to(self.device)            # (B,T,10,1)
            target = batch["target"].to(self.device)          # (B,2)  cos,sin

            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                pred = self.model(pcd, imu, vel, steer)       # (B,2)
                loss = torch.mean((pred - target)**2)          # MSE(cos,sin)

            # AMP backprop
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(loader)

        if self.logger:
            self.logger.log({"train/loss": epoch_loss, "epoch": epoch})

        return epoch_loss

    def validate(self, loader, epoch):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(loader, desc=f"[Valid Epoch {epoch}]")
            for batch in pbar:
                pcd = batch["pcd"].to(self.device)
                imu = batch["imu"].to(self.device)
                vel = batch["vel"].to(self.device)
                steer = batch["steer"].to(self.device)
                target = batch["target"].to(self.device)

                with autocast(enabled=self.use_amp):
                    pred = self.model(pcd, imu, vel, steer)
                    loss = torch.mean((pred - target)**2)

                running_loss += loss.item()
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        val_loss = running_loss / len(loader)

        if self.logger:
            self.logger.log({"valid/loss": val_loss, "epoch": epoch})

        return val_loss

