import os
import json
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from typing import List, Union, Tuple


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def linear_interpolate(seq: Union[List[float], List[List[float]]], target_len: int):
    seq = np.array(seq)
    original_len = len(seq)
    if original_len == target_len:
        return seq

    original_idx = np.linspace(0, 1, original_len)
    target_idx = np.linspace(0, 1, target_len)

    if seq.ndim == 1:
        interpolated = np.interp(target_idx, original_idx, seq)
    else:
        interpolated = np.stack([
            np.interp(target_idx, original_idx, seq[:, d])
            for d in range(seq.shape[1])
        ], axis=-1)

    return interpolated.tolist()


class TrailerHitchSequenceDataset(Dataset):
    """
    - 시계열(차량 각속도/속도/조향각/가속도 등)과 포인트클라우드의 시퀀스를 서로 다른 길이/스트라이드로 슬라이딩
    - 두 모달리티 모두 같은 last_frame(= 타깃 프레임)에 정렬되어 끝난다.
    """
    def __init__(
        self,
        root_dir: str,
        signal_seq_len: int = 20,     # 시계열 시퀀스 길이 (S_sig)
        point_seq_len: int = 20,      # 포인트 시퀀스 길이 (S_pc)
        sensor_sample_len: int = 10,  # 각 프레임 내 시계열 샘플 길이 (L)
        num_points: int = 256,        # PCD 샘플 포인트 수 (N)
        mode: str = "train",
        signal_stride: int = 1,       # 시계열 프레임 간 간격
        point_stride: int = 1,        # 포인트 프레임 간 간격
    ):
        super().__init__()
        assert signal_seq_len >= 1 and point_seq_len >= 1
        assert signal_stride >= 1 and point_stride >= 1

        self.signal_seq_len = signal_seq_len
        self.point_seq_len = point_seq_len
        self.sensor_sample_len = sensor_sample_len
        self.num_points = num_points
        self.signal_stride = signal_stride
        self.point_stride = point_stride

        # 인덱스: (scene_path, last_frame_id)
        self.indices: List[Tuple[str, int]] = []

        mode_dir = os.path.join(root_dir, mode)
        for maneuver_type in sorted(os.listdir(mode_dir)):
            maneuver_path = os.path.join(mode_dir, maneuver_type)
            if not os.path.isdir(maneuver_path):
                continue

            for scene_name in sorted(os.listdir(maneuver_path)):
                scene_path = os.path.join(maneuver_path, scene_name)
                if not os.path.isdir(scene_path):
                    continue

                frame_dirs = sorted([f for f in os.listdir(scene_path) if f.startswith("frame_")])
                frame_ids = sorted([int(f.replace("frame_", "")) for f in frame_dirs])

                if len(frame_ids) == 0:
                    continue

                # 각 윈도우가 마지막 프레임에 맞춰 끝나므로,
                # last_frame 후보는 두 모달리티가 모두 충분한 과거 프레임을 갖는 프레임만 허용
                # 필요 과거 길이 = (len-1)*stride
                need_sig = (self.signal_seq_len - 1) * self.signal_stride
                need_pc  = (self.point_seq_len  - 1) * self.point_stride
                need_max = max(need_sig, need_pc)

                # frame_ids[i]를 last_frame으로 쓸 때 i >= need_max 여야 함
                for i in range(need_max, len(frame_ids)):
                    last_frame = frame_ids[i]
                    self.indices.append((scene_path, last_frame))

        # 빠른 조회를 위해 scene별 frame_ids 캐시
        self._scene_frames_cache = {}

    def __len__(self):
        return len(self.indices)

    def _get_scene_frames(self, scene_path: str):
        if scene_path not in self._scene_frames_cache:
            frame_dirs = sorted([f for f in os.listdir(scene_path) if f.startswith("frame_")])
            frame_ids = sorted([int(f.replace("frame_", "")) for f in frame_dirs])
            self._scene_frames_cache[scene_path] = frame_ids
        return self._scene_frames_cache[scene_path]

    def _gather_signal_seq(self, scene_path: str, last_frame: int):
        """
        last_frame을 끝으로 하는 길이 signal_seq_len의 프레임 인덱스 시퀀스를 stride 적용해 뽑고,
        각 프레임에서 [L, 5] 신호를 구성.
        """
        frame_ids = self._get_scene_frames(scene_path)
        # last_frame의 위치
        last_idx = frame_ids.index(last_frame)

        # 선택할 인덱스: last_idx - k*stride (k = 0..signal_seq_len-1) 역순 → 오름차순 정렬
        pick_indices = [last_idx - k * self.signal_stride for k in range(self.signal_seq_len)][::-1]
        pick_frames = [frame_ids[i] for i in pick_indices]

        signal_seq = []
        for fid in pick_frames:
            frame_dir = os.path.join(scene_path, f"frame_{fid:06d}")

            imu_data   = load_json(os.path.join(frame_dir, "vehicle_imu.json"))
            vel_data   = load_json(os.path.join(frame_dir, "velocity.json"))
            steer_data = load_json(os.path.join(frame_dir, "steering.json"))

            ang_z    = linear_interpolate(imu_data["angular_velocity_z"], self.sensor_sample_len)
            vel_long = linear_interpolate(vel_data["longitudinal_velocity"], self.sensor_sample_len)
            steer    = linear_interpolate(steer_data["steering_tire_angle"], self.sensor_sample_len)
            acc_x    = linear_interpolate(imu_data["linear_acceleration_x"], self.sensor_sample_len)
            acc_y    = linear_interpolate(imu_data["linear_acceleration_y"], self.sensor_sample_len)

            frame_signals = np.stack([ang_z, vel_long, steer, acc_x, acc_y], axis=-1)  # [L, 5]
            signal_seq.append(frame_signals)

        signal_tensor = torch.tensor(np.array(signal_seq), dtype=torch.float32)  # [S_sig, L, 5]
        return signal_tensor

    def _gather_point_seq(self, scene_path: str, last_frame: int):
        """
        last_frame을 끝으로 하는 길이 point_seq_len의 프레임 인덱스 시퀀스를 stride 적용해 뽑고,
        각 프레임에서 포인트클라우드 [N, 3]을 샘플링/패딩.
        """
        frame_ids = self._get_scene_frames(scene_path)
        last_idx = frame_ids.index(last_frame)

        pick_indices = [last_idx - k * self.point_stride for k in range(self.point_seq_len)][::-1]
        pick_frames = [frame_ids[i] for i in pick_indices]

        pc_seq = []
        for fid in pick_frames:
            frame_dir = os.path.join(scene_path, f"frame_{fid:06d}")
            pc_path = os.path.join(frame_dir, "trailer_point.pcd")

            pcd = o3d.io.read_point_cloud(pc_path)
            points = np.asarray(pcd.points, dtype=np.float32)

            if points.shape[0] > self.num_points:
                choose = np.random.choice(points.shape[0], self.num_points, replace=False)
                points = points[choose]
            elif points.shape[0] < self.num_points:
                pad = np.zeros((self.num_points - points.shape[0], 3), dtype=np.float32)
                points = np.concatenate([points, pad], axis=0)

            pc_seq.append(points)

        pc_tensor = torch.tensor(np.array(pc_seq), dtype=torch.float32)  # [S_pc, N, 3]
        return pc_tensor

    def __getitem__(self, idx):
        scene_path, last_frame = self.indices[idx]

        signal_tensor = self._gather_signal_seq(scene_path, last_frame)  # [S_sig, L, 5]
        pc_tensor     = self._gather_point_seq(scene_path, last_frame)   # [S_pc, N, 3]

        hitch_angle_path = os.path.join(scene_path, f"frame_{last_frame:06d}", "hitch_angle.json")
        hitch_angle = load_json(hitch_angle_path)["orientation_z"]

        return {
            "signals": signal_tensor,                # [S_sig, L, 5]
            "point_cloud_seq": pc_tensor,            # [S_pc, N, 3]
            "hitch_angle": torch.tensor(hitch_angle, dtype=torch.float32),
            "meta": (scene_path, last_frame)
        }


# ---- 예시 사용법 ----------------------------------------------------
# dataset = TrailerHitchMultiSeqDataset(
#     root_dir="/path/to/dataset",
#     mode="train",
#     signal_seq_len=30,    # 시계열 30프레임
#     point_seq_len=10,     # 포인트 10프레임
#     sensor_sample_len=10, # 각 프레임 내 L=10
#     num_points=512,
#     signal_stride=1,
#     point_stride=2        # 포인트는 2프레임 간격으로 더 성긴 샘플링
# )
# data = dataset[0]
# print(data["signals"].shape, data["point_cloud_seq"].shape, data["hitch_angle"])
