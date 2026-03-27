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
    # ✅ NEW: BEV
    if "bev" in batch[0]:
        bev_batch = torch.stack([item["bev"] for item in batch], dim=0)  # (B,3,H,W)
    else:
        bev_batch = None

    out = {
        "pcd": pcd_batch,
        "pcd_mask": mask_batch,
        "imu": imu_batch,
        "velocity": vel_batch,
        "steering": steer_batch,
        "gt": gt_batch,
    }
    if bev_batch is not None:
        out["bev"] = bev_batch

    if all("pcd_orig" in item for item in batch):
        Ns_o = [item["pcd_orig"].shape[0] for item in batch]
        N_max_o = max(Ns_o)
        pcd_orig_batch = []
        mask_orig_batch = []
        for item in batch:
            pcd_i = item["pcd_orig"]
            N_i = pcd_i.shape[0]
            if N_i < N_max_o:
                pad = torch.zeros((N_max_o - N_i, 3), dtype=pcd_i.dtype)
                pcd_full = torch.cat([pcd_i, pad], dim=0)
            else:
                pcd_full = pcd_i
            pcd_orig_batch.append(pcd_full)
            mask = torch.zeros(N_max_o, dtype=torch.bool)
            mask[:N_i] = True
            mask_orig_batch.append(mask)
        out["pcd_orig"] = torch.stack(pcd_orig_batch, dim=0)
        out["pcd_orig_mask"] = torch.stack(mask_orig_batch, dim=0)

    if all("bev_orig" in item for item in batch):
        out["bev_orig"] = torch.stack([item["bev_orig"] for item in batch], dim=0)

    if all("gt_orig" in item for item in batch):
        out["gt_orig"] = torch.stack([item["gt_orig"] for item in batch], dim=0)

    if all("aug_rot_deg" in item for item in batch):
        out["aug_rot_deg"] = torch.tensor([item["aug_rot_deg"] for item in batch], dtype=torch.float32)

    if all("joint_shift_x" in item for item in batch):
        out["joint_shift_x"] = torch.tensor([item["joint_shift_x"] for item in batch], dtype=torch.float32)

    if all("centroid_xy" in item for item in batch):
        out["centroid_xy"] = torch.stack([item["centroid_xy"] for item in batch], dim=0)

    if all("occ_box" in item for item in batch):
        out["occ_box"] = torch.stack([item["occ_box"] for item in batch], dim=0)

    if all("occ_applied" in item for item in batch):
        out["occ_applied"] = torch.tensor([item["occ_applied"] for item in batch], dtype=torch.bool)

    if all("occ_rear_count" in item for item in batch):
        out["occ_rear_count"] = torch.tensor([item["occ_rear_count"] for item in batch], dtype=torch.int32)

    if all("domain_id" in item for item in batch):
        out["domain_id"] = torch.tensor([item["domain_id"] for item in batch], dtype=torch.int64)

    return out
