import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

TRAILER_TYPES = {
    "charger": {"len": 1.8, "width": 1.45, "height": 1.5, "joint_to_trailer_x": -1.2},
    "dummy": {"len": 2.6, "width": 1.65, "height": 1.6, "joint_to_trailer_x": -2.35},
    "temporary": {"len": 1.8, "width": 1.45, "height": 1.2, "joint_to_trailer_x": -1.2},
}

def linear_interpolate(seq, target_len, path='none'):
    seq = np.asarray(seq)
    L = len(seq)

    if L == 0:
        print(path)
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

def points_to_bev(
    pts_xyz,
    x_range=(-4.0, 1.5),
    y_range=(-3.0, 3.0),
    z_range=(-2.0, 2.0),
    res=0.05,
    clip_count=10.0,
    joint_shift_x=0.8,
    normalize_xy=False,
    trailer_len=1.8,
    trailer_width=1.45,
):
    pts = pts_xyz.copy()

    # 1) joint-centered shift (LiDAR 0,0 / joint -0.8,0 => x += 0.8)
    pts[:, 0] += joint_shift_x

    # 2) type-aware scale normalization
    if normalize_xy:
        L = max(float(trailer_len), 1e-6)
        W = max(float(trailer_width), 1e-6)
        pts[:, 0] = pts[:, 0] / L
        pts[:, 1] = pts[:, 1] / W

    # ---- 아래부터는 기존 ROI filter / binning / 채널 생성 로직 그대로 ----
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    m = (
        (x >= x_min) & (x < x_max) &
        (y >= y_min) & (y < y_max) &
        (z >= z_min) & (z < z_max)
    )
    pts = pts[m]

    H = int(np.ceil((x_max - x_min) / res))
    W = int(np.ceil((y_max - y_min) / res))

    if pts.shape[0] == 0:
        return np.zeros((3, H, W), dtype=np.float32)

    ix = ((pts[:, 0] - x_min) / res).astype(np.int32)
    iy = ((pts[:, 1] - y_min) / res).astype(np.int32)
    ix = np.clip(ix, 0, H - 1)
    iy = np.clip(iy, 0, W - 1)

    count = np.zeros((H, W), dtype=np.float32)
    np.add.at(count, (ix, iy), 1.0)

    hmax = np.full((H, W), -np.inf, dtype=np.float32)
    for i in range(pts.shape[0]):
        xi, yi = ix[i], iy[i]
        zi = pts[i, 2]
        if zi > hmax[xi, yi]:
            hmax[xi, yi] = zi
    hmax[hmax == -np.inf] = z_min
    hmax = (hmax - z_min) / (z_max - z_min + 1e-6)
    hmax = np.clip(hmax, 0.0, 1.0)

    occ = np.clip(count, 0.0, clip_count) / clip_count
    dlog = np.log1p(count) / np.log1p(clip_count)

    bev = np.stack([occ, hmax, dlog], axis=0).astype(np.float32)
    bev = np.clip(bev, 0.0, 1.0)
    return bev

class HitchDataset(Dataset):
    def __init__(self,
                 root,
                 split_json,
                 split="train",
                 temporal_window=20,
                 micro_seq_length=10,
                 pcd_max_points=1000,
                 trailer_type="charger",
                 normalize_xy=False,
                 ):

        self.root = root
        self.temporal_window = temporal_window
        self.micro_seq_length = micro_seq_length
        self.pcd_max_points = pcd_max_points
        self.trailer_type = trailer_type
        self.normalize_xy = normalize_xy

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
        return linear_interpolate(seq, self.micro_seq_length, path)

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
        
        t = self.trailer_type   # "charger" / "dummy" / "temporary" 
        L = TRAILER_TYPES[t]["len"]
        W = TRAILER_TYPES[t]["width"]
        norm_xy = self.normalize_xy

        bev = points_to_bev(
            pcd,
            x_range=(-2.0, 0.5),
            y_range=(-2.5, 2.5),
            z_range=(-2.0, 2.0),
            res=0.033,
            clip_count=10.0,
            joint_shift_x=0.8,
            normalize_xy=norm_xy,
            trailer_len=L,
            trailer_width=W,
        )

        return {
            "pcd": torch.tensor(pcd, dtype=torch.float32),                 # (N,3)
            "bev": torch.from_numpy(bev),                                  # (3,H,W)
            "imu": torch.from_numpy(imu_seq).float(),                      # (T,micro,3)
            "velocity": torch.from_numpy(vel_seq).float(),                 # (T,micro,1)
            "steering": torch.from_numpy(steer_seq).float(),               # (T,micro,1)
            "gt": last_gt,
        }

