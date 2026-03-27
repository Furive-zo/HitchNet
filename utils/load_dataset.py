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
    x_range=(-2.0, 1.5),
    y_range=(-2.5, 2.5),
    z_range=(-2.0, 2.0),
    res=0.05,
    clip_count=10.0,
    joint_shift_x=0.8,
    use_hmax=True,
    use_dlog=True,
    occ_binary=False,
    add_observed_mask=False,
    observed_bins=360,
    observed_margin=0.0,
    add_xy=False,
    add_orient=False,
):
    pts = pts_xyz.copy()

    # 1) joint-centered shift (LiDAR 0,0 / joint -0.8,0 => x += 0.8)
    pts[:, 0] += joint_shift_x

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
        base_channels = [np.zeros((H, W), dtype=np.float32)]
        if use_hmax:
            base_channels.append(np.zeros((H, W), dtype=np.float32))
        if use_dlog:
            base_channels.append(np.zeros((H, W), dtype=np.float32))
        if add_observed_mask:
            base_channels.append(np.zeros((H, W), dtype=np.float32))
        base = np.stack(base_channels, axis=0).astype(np.float32)
        if add_xy:
            x_centers = x_min + (np.arange(H) + 0.5) * res
            y_centers = y_min + (np.arange(W) + 0.5) * res
            x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
            xy = np.stack([x_grid, y_grid], axis=0).astype(np.float32)
            base = np.concatenate([base, xy], axis=0)
        if add_orient:
            orient = np.zeros((2, H, W), dtype=np.float32)
            base = np.concatenate([base, orient], axis=0)
        return base

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

    if occ_binary:
        occ = (count > 0.0).astype(np.float32)
    else:
        occ = np.clip(count, 0.0, clip_count) / clip_count
    dlog = np.log1p(count) / np.log1p(clip_count)

    base_channels = [occ]
    if use_hmax:
        base_channels.append(hmax)
    if use_dlog:
        base_channels.append(dlog)
    if add_observed_mask:
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        bins = int(max(1, observed_bins))
        bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
        bin_idx = np.clip(np.digitize(phi, bin_edges) - 1, 0, bins - 1)
        rmax = np.zeros(bins, dtype=np.float32)
        np.maximum.at(rmax, bin_idx, r)

        x_centers = x_min + (np.arange(H) + 0.5) * res
        y_centers = y_min + (np.arange(W) + 0.5) * res
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
        phi_grid = np.arctan2(y_grid, x_grid)
        r_grid = np.sqrt(x_grid ** 2 + y_grid ** 2)
        grid_idx = np.clip(np.digitize(phi_grid, bin_edges) - 1, 0, bins - 1)
        rmax_grid = rmax[grid_idx]
        observed = (r_grid <= (rmax_grid + float(observed_margin))).astype(np.float32)
        base_channels.append(observed)
    bev = np.stack(base_channels, axis=0).astype(np.float32)
    bev = np.clip(bev, 0.0, 1.0)
    if add_xy:
        x_centers = x_min + (np.arange(H) + 0.5) * res
        y_centers = y_min + (np.arange(W) + 0.5) * res
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
        xy = np.stack([x_grid, y_grid], axis=0).astype(np.float32)
        bev = np.concatenate([bev, xy], axis=0)
    if add_orient:
        sum_x = np.zeros((H, W), dtype=np.float32)
        sum_y = np.zeros((H, W), dtype=np.float32)
        sum_xx = np.zeros((H, W), dtype=np.float32)
        sum_xy = np.zeros((H, W), dtype=np.float32)
        sum_yy = np.zeros((H, W), dtype=np.float32)

        np.add.at(sum_x, (ix, iy), pts[:, 0])
        np.add.at(sum_y, (ix, iy), pts[:, 1])
        np.add.at(sum_xx, (ix, iy), pts[:, 0] * pts[:, 0])
        np.add.at(sum_xy, (ix, iy), pts[:, 0] * pts[:, 1])
        np.add.at(sum_yy, (ix, iy), pts[:, 1] * pts[:, 1])

        mask = count > 1.0
        inv_n = np.zeros_like(count)
        inv_n[mask] = 1.0 / count[mask]
        mean_x = sum_x * inv_n
        mean_y = sum_y * inv_n
        cxx = sum_xx * inv_n - mean_x * mean_x
        cxy = sum_xy * inv_n - mean_x * mean_y
        cyy = sum_yy * inv_n - mean_y * mean_y

        angle = 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)
        orient = np.zeros((2, H, W), dtype=np.float32)
        orient[0] = np.cos(angle)
        orient[1] = np.sin(angle)
        orient[:, ~mask] = 0.0
        bev = np.concatenate([bev, orient], axis=0)
    return bev

class HitchDataset(Dataset):
    def __init__(self,
                 root,
                 split_json,
                 split="train",
                 temporal_window=20,
                 micro_seq_length=10,
                 trailer_type="charger",
                 normalize_xy=False,
                 bev_add_xy=False,
                 bev_add_orient=False,
                 bev_use_hmax=True,
                 bev_use_dlog=True,
                 occ_binary=False,
                 add_observed_mask=False,
                 observed_bins=360,
                 observed_margin=0.0,
                 aug_rotate_deg=0.0,
                 aug_rotate_prob=0.0,
                 aug_rotate_min_deg=None,
                 aug_rotate_max_deg=None,
                 aug_rotate_target_min_deg=None,
                 aug_rotate_target_max_deg=None,
                 occ_prob=0.0,
                 occ_x_thresh=-0.3,
                 occ_box_x=0.0,
                 occ_box_y=0.0,
                 trans_prob=0.0,
                 trans_max_m=0.0,
                 trans_dir="both",
                 joint_shift_x=0.8,
                 aug_vis=False,
                 centroid_mode="minmax",
                 dlog_range_norm=False,
                 dlog_range_norm_mode="center",
                 dlog_range_stats=None,
                 skip_temporal=False,
                 build_bev=True,
                 ):

        self.root = root
        self.split = split
        self.temporal_window = temporal_window
        self.micro_seq_length = micro_seq_length
        self.trailer_type = trailer_type
        self.normalize_xy = normalize_xy
        self.bev_add_xy = bev_add_xy
        self.bev_add_orient = bev_add_orient
        self.bev_use_hmax = bool(bev_use_hmax)
        self.bev_use_dlog = bool(bev_use_dlog)
        self.occ_binary = bool(occ_binary)
        self.add_observed_mask = bool(add_observed_mask)
        self.observed_bins = int(observed_bins)
        self.observed_margin = float(observed_margin)
        self.aug_rotate_deg = float(aug_rotate_deg)
        self.aug_rotate_prob = float(aug_rotate_prob)
        self.aug_rotate_min_deg = None if aug_rotate_min_deg is None else float(aug_rotate_min_deg)
        self.aug_rotate_max_deg = None if aug_rotate_max_deg is None else float(aug_rotate_max_deg)
        self.aug_rotate_target_min_deg = None if aug_rotate_target_min_deg is None else float(aug_rotate_target_min_deg)
        self.aug_rotate_target_max_deg = None if aug_rotate_target_max_deg is None else float(aug_rotate_target_max_deg)
        self.occ_prob = float(occ_prob)
        self.occ_x_thresh = float(occ_x_thresh)
        self.occ_box_x = float(occ_box_x)
        self.occ_box_y = float(occ_box_y)
        self.trans_prob = float(trans_prob)
        self.trans_max_m = float(trans_max_m)
        self.trans_dir = str(trans_dir or "both")
        self.joint_shift_x = float(joint_shift_x)
        self.aug_vis = bool(aug_vis)
        self.centroid_mode = str(centroid_mode or "minmax")
        self.dlog_range_norm = bool(dlog_range_norm)
        self.dlog_range_norm_mode = str(dlog_range_norm_mode or "center")
        self.dlog_range_stats = dlog_range_stats
        self.skip_temporal = bool(skip_temporal)
        self.build_bev = bool(build_bev)
        self.bev_x_range = (-4.0, 0.5)
        self.bev_y_range = (-4.0, 4.0)
        self.bev_z_range = (-2.0, 2.0)
        self.bev_res = 0.033
        self.bev_clip_count = 10.0
        self._range_bin_idx = None

        if self.dlog_range_norm and self.dlog_range_stats is not None:
            self._build_range_bin_idx()

        # Load frame directories
        with open(split_json) as f:
            seqs = json.load(f)[split]

        self.frame_dirs = []
        self.frame_domain_ids = []
        self.domain_name_to_id = {}
        for seq in seqs:
            seq_dir = os.path.join(root, seq)
            frames = sorted([d for d in os.listdir(seq_dir)
                             if d.startswith("frame_")])
            if seq not in self.domain_name_to_id:
                self.domain_name_to_id[seq] = len(self.domain_name_to_id)
            did = self.domain_name_to_id[seq]
            for fr in frames:
                self.frame_dirs.append(os.path.join(seq_dir, fr))
                self.frame_domain_ids.append(did)

        print(f"[INFO] HitchDataset split={split}, frames={len(self.frame_dirs)}")

    def _build_range_bin_idx(self):
        x_min, x_max = self.bev_x_range
        y_min, y_max = self.bev_y_range
        res = self.bev_res
        H = int(np.ceil((x_max - x_min) / res))
        W = int(np.ceil((y_max - y_min) / res))
        x_centers = x_min + (np.arange(H) + 0.5) * res
        y_centers = y_min + (np.arange(W) + 0.5) * res
        x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
        r = np.sqrt(x_grid ** 2 + y_grid ** 2)
        bin_size = float(self.dlog_range_stats["bin_size"])
        nbins = int(self.dlog_range_stats["nbins"])
        bin_idx = np.floor(r / bin_size).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, nbins - 1)
        self._range_bin_idx = bin_idx

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
        apply_rot = (
            self.split == "train"
            and self.aug_rotate_prob > 0.0
            and np.random.rand() < self.aug_rotate_prob
        )
        gt_ref_deg = None
        if apply_rot and (self.aug_rotate_target_min_deg is not None or self.aug_rotate_target_max_deg is not None):
            curr_fr = self.frame_dirs[idxs[-1]]
            gt_json = self._load_json(os.path.join(curr_fr, "gt_hitch_angle.json"))
            gt_ref_deg = float(gt_json.get("gt_hitch_angle_deg", 0.0))
        if apply_rot:
            rot_lo = None
            rot_hi = None
            if self.aug_rotate_min_deg is not None or self.aug_rotate_max_deg is not None:
                lo = self.aug_rotate_min_deg if self.aug_rotate_min_deg is not None else 0.0
                hi = self.aug_rotate_max_deg if self.aug_rotate_max_deg is not None else 0.0
                if hi < lo:
                    lo, hi = hi, lo
                rot_lo, rot_hi = lo, hi
            elif self.aug_rotate_deg > 0.0:
                rot_lo, rot_hi = -self.aug_rotate_deg, self.aug_rotate_deg

            if gt_ref_deg is not None:
                tgt_lo = self.aug_rotate_target_min_deg if self.aug_rotate_target_min_deg is not None else gt_ref_deg
                tgt_hi = self.aug_rotate_target_max_deg if self.aug_rotate_target_max_deg is not None else gt_ref_deg
                if tgt_hi < tgt_lo:
                    tgt_lo, tgt_hi = tgt_hi, tgt_lo
                tgt_rot_lo = tgt_lo - gt_ref_deg
                tgt_rot_hi = tgt_hi - gt_ref_deg
                if rot_lo is None or rot_hi is None:
                    rot_lo, rot_hi = tgt_rot_lo, tgt_rot_hi
                else:
                    rot_lo = max(rot_lo, tgt_rot_lo)
                    rot_hi = min(rot_hi, tgt_rot_hi)

            if rot_lo is None or rot_hi is None:
                rot_lo, rot_hi = 0.0, 0.0
            if rot_hi < rot_lo:
                rot_deg = 0.0
            else:
                rot_deg = np.random.uniform(rot_lo, rot_hi)
            rot_rad = np.deg2rad(rot_deg)
        else:
            rot_deg = 0.0
            rot_rad = 0.0

        # =============================
        # 1) CURRENT PCD (only last frame)
        # =============================
        curr_fr = self.frame_dirs[idxs[-1]]
        gt_json_curr = self._load_json(os.path.join(curr_fr, "gt_hitch_angle.json"))
        gt_deg_curr = float(gt_json_curr.get("gt_hitch_angle_deg", 0.0))
        gt_rad_raw = np.deg2rad(gt_deg_curr)
        gt_rad_curr = gt_rad_raw
        if rot_rad != 0.0:
            gt_rad_curr = gt_rad_curr + rot_rad

        pcd = self._load_pcd(os.path.join(curr_fr, "trailer_point.pcd"))  # (N,3)
        pcd_orig = pcd.copy() if self.aug_vis else None
        if rot_rad != 0.0:
            cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
            cx = -self.joint_shift_x
            cy = 0.0
            x = pcd[:, 0] - cx
            y = pcd[:, 1] - cy
            xr = cos_r * x - sin_r * y + cx
            yr = sin_r * x + cos_r * y + cy
            pcd[:, 0] = xr
            pcd[:, 1] = yr


        apply_occ = (
            self.split == "train"
            and self.occ_prob > 0.0
            and np.random.rand() < self.occ_prob
            and pcd.shape[0] > 0
        )
        occ_box = None
        occ_applied = False
        occ_rear_count = 0
        if apply_occ:
            occ_applied = True
            v = np.array([np.cos(gt_rad_curr), np.sin(gt_rad_curr)], dtype=np.float32)
            x_joint = pcd[:, 0] + self.joint_shift_x
            y_joint = pcd[:, 1]
            proj = x_joint * v[0] + y_joint * v[1]
            rear_mask = proj < self.occ_x_thresh
            occ_rear_count = int(np.sum(rear_mask))
            if np.any(rear_mask):
                if self.occ_box_x > 0.0 and self.occ_box_y > 0.0:
                    rear_x = x_joint[rear_mask]
                    rear_y = y_joint[rear_mask]
                    x_min = float(rear_x.min())
                    x_max = float(rear_x.max())
                    y_min = float(rear_y.min())
                    y_max = float(rear_y.max())
                    if x_max > x_min and y_max > y_min:
                        cx = np.random.uniform(x_min, x_max)
                        cy = np.random.uniform(y_min, y_max)
                    else:
                        cx = float(rear_x.mean())
                        cy = float(rear_y.mean())
                    hx = self.occ_box_x * 0.5
                    hy = self.occ_box_y * 0.5
                    in_box = (
                        (x_joint >= cx - hx) & (x_joint <= cx + hx) &
                        (y_joint >= cy - hy) & (y_joint <= cy + hy)
                    )
                    pcd = pcd[~in_box]
                    occ_box = (cx - self.joint_shift_x, cy, hx, hy)

        if (
            self.split == "train"
            and self.trans_prob > 0.0
            and self.trans_max_m > 0.0
            and np.random.rand() < self.trans_prob
        ):
            v = np.array([np.cos(gt_rad_curr), np.sin(gt_rad_curr)], dtype=np.float32)
            t = np.random.uniform(0.0, self.trans_max_m)
            if self.trans_dir == "forward":
                sign = 1.0
            elif self.trans_dir == "backward":
                sign = -1.0
            else:
                sign = 1.0 if np.random.rand() < 0.5 else -1.0
            pcd[:, 0] += sign * t * v[0]
            pcd[:, 1] += sign * t * v[1]

        # =============================
        # 2) Temporal IMU/Vel/Steer
        # =============================
        if self.skip_temporal:
            imu_seq = np.zeros((T, self.micro_seq_length, 3), dtype=np.float32)
            vel_seq = np.zeros((T, self.micro_seq_length, 1), dtype=np.float32)
            steer_seq = np.zeros((T, self.micro_seq_length, 1), dtype=np.float32)
            last_gt = torch.tensor([np.cos(gt_rad_curr), np.sin(gt_rad_curr)], dtype=torch.float32)
        else:
            for fi in idxs:
                fr = self.frame_dirs[fi]

                imu_seq.append(self._load_imu(os.path.join(fr, "vehicle_imu.json")))
                vel_seq.append(self._load_vel(os.path.join(fr, "vehicle_velocity.json")))
                steer_seq.append(self._load_steer(os.path.join(fr, "vehicle_steering.json")))

                # GT
                gt_json = self._load_json(os.path.join(fr, "gt_hitch_angle.json"))
                gt_deg = gt_json.get("gt_hitch_angle_deg", 0.0)
                gt_rad = np.deg2rad(gt_deg)
                if rot_rad != 0.0:
                    gt_rad = gt_rad + rot_rad
                gt_seq.append([np.cos(gt_rad), np.sin(gt_rad)])

            # Last frame target
            last_gt = torch.tensor(gt_seq[-1], dtype=torch.float32)  # (2,)
        if pcd_orig is not None:
            gt_orig = torch.tensor([np.cos(gt_rad_raw), np.sin(gt_rad_raw)], dtype=torch.float32)
        else:
            gt_orig = None

        theta_last = float(np.arctan2(last_gt[1].item(), last_gt[0].item()))
        theta_last_orig = theta_last - rot_rad

        if not self.skip_temporal:
            imu_seq = np.stack(imu_seq, axis=0)          # (T, micro, 3)
            vel_seq = np.stack(vel_seq, axis=0)          # (T, micro, 1)
            steer_seq = np.stack(steer_seq, axis=0)      # (T, micro, 1)
        
        t = self.trailer_type   # "charger" / "dummy" / "temporary" 
        L = TRAILER_TYPES[t]["len"]
        W = TRAILER_TYPES[t]["width"]
        norm_xy = self.normalize_xy
        shift_x = float(self.joint_shift_x)

        if norm_xy:
            S = max(float(L), float(W), 1e-6) / 2
            shift_x_before = shift_x
            pcd = pcd.copy()
            pcd[:, 0] = (pcd[:, 0] + shift_x) / S
            pcd[:, 1] = pcd[:, 1] / S
            if pcd_orig is not None:
                pcd_orig = pcd_orig.copy()
                pcd_orig[:, 0] = (pcd_orig[:, 0] + shift_x) / S
                pcd_orig[:, 1] = pcd_orig[:, 1] / S
            shift_x = 0.0
            if occ_box is not None:
                cx, cy, hx, hy = occ_box
                occ_box = ((cx + shift_x_before) / S, cy / S, hx / S, hy / S)

        if pcd.shape[0] > 0:
            if self.centroid_mode == "mean":
                centroid_xy = torch.tensor([float(pcd[:, 0].mean()), float(pcd[:, 1].mean())], dtype=torch.float32)
            else:
                x_min, x_max = float(pcd[:, 0].min()), float(pcd[:, 0].max())
                y_min, y_max = float(pcd[:, 1].min()), float(pcd[:, 1].max())
                centroid_xy = torch.tensor([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=torch.float32)
        else:
            centroid_xy = torch.zeros(2, dtype=torch.float32)

        bev = None
        bev_orig = None
        if self.build_bev:
            bev = points_to_bev(
                pcd,
                x_range=self.bev_x_range,
                y_range=self.bev_y_range,
                z_range=self.bev_z_range,
                res=self.bev_res,
                clip_count=self.bev_clip_count,
                joint_shift_x=shift_x,
                use_hmax=self.bev_use_hmax,
                use_dlog=self.bev_use_dlog,
                occ_binary=self.occ_binary,
                add_observed_mask=self.add_observed_mask,
                observed_bins=self.observed_bins,
                observed_margin=self.observed_margin,
                add_xy=self.bev_add_xy,
                add_orient=self.bev_add_orient,
            )
            if self.dlog_range_norm and self.dlog_range_stats is not None and self.bev_use_dlog:
                if self._range_bin_idx is None:
                    self._build_range_bin_idx()
                dlog_idx = 1 + (1 if self.bev_use_hmax else 0)
                mu = self.dlog_range_stats["mu"]
                sigma = self.dlog_range_stats["sigma"]
                idx = self._range_bin_idx
                mu_map = mu[idx]
                sig_map = sigma[idx]
                if self.dlog_range_norm_mode == "zscore":
                    bev[dlog_idx] = (bev[dlog_idx] - mu_map) / (sig_map + 1e-6)
                else:
                    bev[dlog_idx] = bev[dlog_idx] - mu_map
            if pcd_orig is not None:
                bev_orig = points_to_bev(
                    pcd_orig,
                    x_range=self.bev_x_range,
                    y_range=self.bev_y_range,
                    z_range=self.bev_z_range,
                    res=self.bev_res,
                    clip_count=self.bev_clip_count,
                    joint_shift_x=shift_x,
                    use_hmax=self.bev_use_hmax,
                    use_dlog=self.bev_use_dlog,
                    occ_binary=self.occ_binary,
                    add_observed_mask=self.add_observed_mask,
                    observed_bins=self.observed_bins,
                    observed_margin=self.observed_margin,
                    add_xy=self.bev_add_xy,
                    add_orient=self.bev_add_orient,
                )

        out = {
            "pcd": torch.tensor(pcd, dtype=torch.float32),                 # (N,3)
            "centroid_xy": centroid_xy,                                    # (2,)
            "imu": torch.from_numpy(imu_seq).float(),                      # (T,micro,3)
            "velocity": torch.from_numpy(vel_seq).float(),                 # (T,micro,1)
            "steering": torch.from_numpy(steer_seq).float(),               # (T,micro,1)
            "gt": last_gt,
            "domain_id": int(self.frame_domain_ids[idxs[-1]]),
            "aug_rot_deg": float(rot_deg),
            "joint_shift_x": float(shift_x),
            "occ_applied": bool(occ_applied),
            "occ_rear_count": int(occ_rear_count),
        }
        if bev is not None:
            out["bev"] = torch.from_numpy(bev)                              # (C,H,W)
        if occ_box is not None:
            out["occ_box"] = torch.tensor(occ_box, dtype=torch.float32)
        if pcd_orig is not None and bev_orig is not None and gt_orig is not None:
            out["bev_orig"] = torch.from_numpy(bev_orig)
            out["gt_orig"] = gt_orig
        return out
