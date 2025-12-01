# utils/collate.py

import torch
import numpy as np

def collate_fn(batch):
    """
    batch: list of dicts
      each dict:
        pcd: (N_i, 3)
        imu: (T, M, 3)
        vel: ...
        steer: ...
        gt: (2,)
    """

    B = len(batch)
    
    # 1) pcd 개수 확인
    Ns = [item["pcd"].shape[0] for item in batch]
    N_max = max(Ns)

    # 2) 패딩된 텐서 생성
    pcd_batch = []
    mask_batch = []

    for item in batch:
        pcd_i = item["pcd"]         # (N_i, 3)
        N_i = pcd_i.shape[0]

        # pad => (N_max, 3)
        if N_i < N_max:
            pad = torch.zeros((N_max - N_i, 3), dtype=pcd_i.dtype)
            pcd_full = torch.cat([pcd_i, pad], dim=0)
        else:
            pcd_full = pcd_i

        pcd_batch.append(pcd_full)     # (N_max,3)
        
        mask = torch.zeros(N_max, dtype=torch.bool)
        mask[:N_i] = True
        mask_batch.append(mask)

    pcd_batch = torch.stack(pcd_batch, dim=0)   # (B, N_max, 3)
    mask_batch = torch.stack(mask_batch, dim=0) # (B, N_max)

    # 3) 나머지 텐서들은 그대로 stack
    imu_batch = torch.stack([item["imu"] for item in batch], dim=0)
    vel_batch = torch.stack([item["velocity"] for item in batch], dim=0)
    steer_batch = torch.stack([item["steering"] for item in batch], dim=0)
    gt_batch = torch.stack([item["gt"] for item in batch], dim=0)

    return {
        "pcd": pcd_batch,
        "pcd_mask": mask_batch,     # 새로 추가됨
        "imu": imu_batch,
        "velocity": vel_batch,
        "steering": steer_batch,
        "gt": gt_batch,
    }
