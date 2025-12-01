import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


def linear_interpolate(seq, target_len):
    seq = np.asarray(seq)
    L = len(seq)

    if L == 0:
        return np.zeros((target_len, seq.shape[1]))

    if L == target_len:
        return seq

    xp = np.linspace(0, L - 1, num=L)
    x_new = np.linspace(0, L - 1, num=target_len)

    C = seq.shape[1]
    out = []
    for d in range(C):
        out.append(np.interp(x_new, xp, seq[:, d]))
    return np.stack(out, axis=1)


class HitchDataset(Dataset):
    def __init__(self,
                 root,
                 split_json,
                 split="train",
                 temporal_window=20,
                 micro_seq_length=10,
                 pcd_max_points=1000):

        self.root = root
        self.temporal_window = temporal_window
        self.micro_seq_length = micro_seq_length
        self.pcd_max_points = pcd_max_points

        # Load frame directories
        with open(split_json) as f:
            seqs = json.load(f)[split]

        self.frame_dirs = []
        for seq in seqs:
            seq_dir = os.path.join(root, seq)
            frames = sorted([d for d in os.listdir(seq_dir)
                             if d.startswith("frame_")])
            for fr in frames:
                self.frame_dirs.append(os.path.join(seq_dir, fr))

        print(f"[INFO] HitchDataset split={split}, frames={len(self.frame_dirs)}")

    def __len__(self):
        return len(self.frame_dirs)

    # -----------------------------
    # Loaders
    # -----------------------------
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _load_pcd(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points)

        # Downsample to max points
        if pts.shape[0] > self.pcd_max_points:
            idx = np.random.choice(len(pts), self.pcd_max_points, replace=False)
            pts = pts[idx]

        return pts.astype(np.float32)

    def _load_imu(self, path):
        js = self._load_json(path)

        keys = ["linear_acceleration_x",
                "linear_acceleration_y",
                "angular_velocity_z"]

        L = len(js.get(keys[0], []))
        seq = []
        for i in range(L):
            sample = []
            for k in keys:
                vals = js.get(k, [])
                sample.append(vals[i] if i < len(vals) else 0.0)
            seq.append(sample)

        seq = np.array(seq, dtype=np.float32)
        return linear_interpolate(seq, self.micro_seq_length)

    def _load_vel(self, path):
        js = self._load_json(path)
        vals = js.get("longitudinal_velocity", [])
        seq = np.array(vals, dtype=np.float32).reshape(-1, 1)
        return linear_interpolate(seq, self.micro_seq_length)

    def _load_steer(self, path):
        js = self._load_json(path)
        vals = js.get("steering_tire_angle", [])
        seq = np.array(vals, dtype=np.float32).reshape(-1, 1)
        return linear_interpolate(seq, self.micro_seq_length)

    # -----------------------------
    # Main
    # -----------------------------
    def __getitem__(self, idx):
        T = self.temporal_window

        # Sequence index selection
        start = max(0, idx - T + 1)
        idxs = list(range(start, idx + 1))

        # padding for beginning
        if len(idxs) < T:
            pad = [0] * (T - len(idxs))
            idxs = pad + idxs

        imu_seq, vel_seq, steer_seq, gt_seq = [], [], [], []

        # =============================
        # 1) CURRENT PCD (only last frame)
        # =============================
        curr_fr = self.frame_dirs[idxs[-1]]
        pcd = self._load_pcd(os.path.join(curr_fr, "trailer_point.pcd"))  # (N,3)

        # =============================
        # 2) Temporal IMU/Vel/Steer
        # =============================
        for fi in idxs:
            fr = self.frame_dirs[fi]

            imu_seq.append(self._load_imu(os.path.join(fr, "vehicle_imu.json")))
            vel_seq.append(self._load_vel(os.path.join(fr, "vehicle_velocity.json")))
            steer_seq.append(self._load_steer(os.path.join(fr, "vehicle_steering.json")))

            # GT
            gt_json = self._load_json(os.path.join(fr, "gt_hitch_angle.json"))
            gt_deg = gt_json.get("gt_hitch_angle_deg", 0.0)
            gt_rad = np.deg2rad(gt_deg)
            gt_seq.append([np.cos(gt_rad), np.sin(gt_rad)])

        # Last frame target
        last_gt = torch.tensor(gt_seq[-1], dtype=torch.float32)  # (2,)

        imu_seq = np.stack(imu_seq, axis=0)          # (T, micro, 3)
        vel_seq = np.stack(vel_seq, axis=0)          # (T, micro, 1)
        steer_seq = np.stack(steer_seq, axis=0)      # (T, micro, 1)

        return {
            "pcd": torch.tensor(pcd, dtype=torch.float32),                 # (N,3)
            "imu": torch.from_numpy(imu_seq).float(),                      # (T,micro,3)
            "velocity": torch.from_numpy(vel_seq).float(),                 # (T,micro,1)
            "steering": torch.from_numpy(steer_seq).float(),               # (T,micro,1)
            "gt": last_gt,
        }

