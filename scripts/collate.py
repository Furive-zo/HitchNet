import os
import torch
import open3d as o3d
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph

def load_pointcloud_from_pcd(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pointcloud file not found: {path}")
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points)  # shape: [N, 3]

def estimate_normals_open3d(xyz, k=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)  # [N, 3]
    return normals

def pca_centerline(xy):
    center = np.mean(xy, axis=0)
    centered = xy - center
    cov = np.cov(centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    direction = eig_vecs[:, np.argmax(eig_vals)]  # 가장 큰 분산 방향
    direction = direction / np.linalg.norm(direction)

    # 수직 방향 후보 2개
    perp1 = np.array([-direction[1], direction[0]])  # 직각 방향
    perp2 = np.array([ direction[1], -direction[0]])

    # 기울기 계산 (dy / dx) → 0에 가까운 쪽 선택
    def slope_abs(v):
        if abs(v[0]) < 1e-6:
            return float("inf")  # 수직 선에 가까운 경우
        return abs(v[1] / v[0])

    slope1 = slope_abs(perp1)
    slope2 = slope_abs(perp2)

    perp_dir = perp1 if slope1 < slope2 else perp2
    perp_dir = perp_dir / np.linalg.norm(perp_dir)

    return center, perp_dir

def get_distances_to_centerline(xy, center, direction):
    vec = xy - center[None, :]                      # [N, 2]
    proj = np.dot(vec, direction)[:, None] * direction[None, :]
    perp = vec - proj                               # [N, 2]
    distances = np.linalg.norm(perp, axis=1)        # [N]
    return distances

def estimate_normals_open3d(xyz, k=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals)
    return normals

def pointcloud_to_graph(xyz, k=8, batch_idx=None):
    xyz = np.asarray(xyz)                           # [N, 3]
    xy = xyz[:, :2]

    # 1. PCA 중심선 → 수직 거리
    center, direction = pca_centerline(xy)
    distances = get_distances_to_centerline(xy, center, direction)
    distances = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    # 2. 표면 노멀 벡터
    normals = estimate_normals_open3d(xyz)          # [N, 3]
    normals_xy = normals[:, :2]                     # [N, 2] → nx, ny
    normal_vec = torch.tensor(normals_xy, dtype=torch.float32)

    # 3. 위치
    pos = torch.tensor(xyz, dtype=torch.float32)    # [N, 3]

    # 4. 피처 조합: [x, y, z, nx, ny, dist]
    node_feature = torch.cat([pos, normal_vec, distances], dim=1)  # [N, 6]

    edge_index = knn_graph(pos, k=k)
    data = Data(x=node_feature, pos=pos, edge_index=edge_index)
    if batch_idx is not None:
        data.batch = batch_idx
    return data


def collate_fn(batch):
    angular = torch.stack([b["angular"] for b in batch])
    vel = torch.stack([b["vel"] for b in batch])
    steer = torch.stack([b["steer"] for b in batch])
    target = torch.stack([b["hitch_angle"] for b in batch])

    graphs = []
    for i, b in enumerate(batch):
        xyz = load_pointcloud_from_pcd(b["pointcloud_path"])
        num_points = xyz.shape[0]
        batch_idx = torch.full((num_points,), i, dtype=torch.long)
        graphs.append(pointcloud_to_graph(xyz, k=8, batch_idx=batch_idx))

    graph_batch = Batch.from_data_list(graphs)

    return {
        "angular": angular,
        "vel": vel,
        "steer": steer,
        "graph": graph_batch,
        "hitch_angle": target
    }

def collate_fn_pointnet(batch_list):
    pcs = []
    targets = []

    for sample in batch_list:
        pcd = o3d.io.read_point_cloud(sample["pointcloud_path"])
        points = np.asarray(pcd.points, dtype=np.float32)

        if points.shape[0] != 256:
            # 샘플링 or 패딩
            if points.shape[0] > 256:
                idx = np.random.choice(points.shape[0], 256, replace=False)
                points = points[idx]
            else:
                pad = np.zeros((256 - points.shape[0], 3), dtype=np.float32)
                points = np.concatenate([points, pad], axis=0)

        pcs.append(torch.tensor(points, dtype=torch.float32))  # [256, 3]
        targets.append(sample["hitch_angle"])

    pcs = torch.stack(pcs, dim=0)           # [B, 256, 3]
    targets = torch.stack(targets, dim=0)   # [B]
    return {"point_cloud": pcs, "hitch_angle": targets}

def pointcloud_to_graph_pos_only(xyz, k=8, batch_idx=None):
    xyz = np.asarray(xyz)
    pos = torch.tensor(xyz, dtype=torch.float32)  # [N, 3]
    edge_index = knn_graph(pos, k=k)
    data = Data(x=pos, pos=pos, edge_index=edge_index)

    if batch_idx is not None:
        data.batch = batch_idx 

    return data

def collate_fn_graph(batch_list):
    graphs = []
    targets = []

    for i, b in enumerate(batch_list):
        xyz = load_pointcloud_from_pcd(b["pointcloud_path"])
        num_points = xyz.shape[0]
        batch_idx = torch.full((num_points,), i, dtype=torch.long)  # 각 포인트의 배치 인덱스
        graph = pointcloud_to_graph_pos_only(xyz, k=8, batch_idx=batch_idx)
        graphs.append(graph)
        targets.append(b["hitch_angle"])

    graph_batch = Batch.from_data_list(graphs)
    targets = torch.stack(targets, dim=0)

    return {
        "graph": graph_batch,         # PyG Batch, x: [N, 3], edge_index: [2, E]
        "hitch_angle": targets        # [B]
    }

def pointcloud_to_topview(points, grid_size=0.1, height=10.0, width=10.0):
    x_max = width / 2
    y_max = height / 2
    x_min = -x_max
    y_min = -y_max

    H = int(height / grid_size)
    W = int(width / grid_size)

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    )
    filtered = points[mask]

    if filtered.shape[0] == 0:
        return torch.zeros(1, H, W, dtype=torch.float32)

    x_idx = ((filtered[:, 0] - x_min) / grid_size).astype(int)
    y_idx = ((filtered[:, 1] - y_min) / grid_size).astype(int)

    top_view = np.zeros((H, W), dtype=np.float32)
    for x, y, z in zip(x_idx, y_idx, filtered[:, 2]):
        if 0 <= y < H and 0 <= x < W:
            if z > top_view[y, x]:
                top_view[y, x] = z

    return torch.tensor(top_view, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

def collate_fn_bev(batch_list, grid_size=0.1, height=10.0, width=10.0):
    images = []
    targets = []

    for sample in batch_list:
        pcd = o3d.io.read_point_cloud(sample["pointcloud_path"])
        points = np.asarray(pcd.points, dtype=np.float32)
        bev_img = pointcloud_to_topview(points, grid_size, height, width)  # [1, H, W]
        images.append(bev_img)
        targets.append(sample["hitch_angle"])

    images = torch.stack(images, dim=0)         # [B, 1, H, W]
    targets = torch.stack(targets, dim=0)       # [B]

    return {"image": images, "hitch_angle": targets}
