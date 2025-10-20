#!/usr/bin/env python3
import os
import json
import argparse
import rclpy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from autoware_auto_vehicle_msgs.msg import SteeringReport, VelocityReport
import open3d as o3d
import numpy as np
from sensor_msgs_py import point_cloud2
from collections import defaultdict
from builtin_interfaces.msg import Time

# =============================================================================
# ROI 파라미터
ROI_MIN_X = -3.0
ROI_MAX_X = -1.0
ROI_MIN_Y = -1.5
ROI_MAX_Y =  1.5
ROI_MIN_Z = 0.3 
ROI_MAX_Z =  1.0  
# =============================================================================

DEBUG = False
FULL_CONVERT = False

def dbg(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def to_sec(t: Time) -> float:
    return t.sec + t.nanosec * 1e-9

def get_time(msg):
    if hasattr(msg, 'header'):
        return msg.header.stamp
    if hasattr(msg, 'stamp'):
        return msg.stamp
    raise AttributeError(f"No timestamp in {type(msg)}")

def ros_to_dict(msg):
    result = {}
    for slot in msg.__slots__:
        val = getattr(msg, slot)
        if hasattr(val, '__slots__'):
            result[slot] = ros_to_dict(val)
        elif isinstance(val, (list, tuple)):
            result[slot] = [ros_to_dict(v) if hasattr(v, '__slots__') else v for v in val]
        else:
            result[slot] = val
    return result

# PointCloud2 → Open3D PointCloud
def pc2_to_o3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    pts = []
    for p in point_cloud2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
        pts.append([float(p[0]), float(p[1]), float(p[2])])
    pcd = o3d.geometry.PointCloud()
    if pts:
        pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    return pcd

# ROI 필터링: x, y, z 축 모두 고려
def filter_trailer(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return pcd
    mask = np.logical_and.reduce([
        pts[:,0] > ROI_MIN_X, pts[:,0] < ROI_MAX_X,
        pts[:,1] > ROI_MIN_Y, pts[:,1] < ROI_MAX_Y,
        pts[:,2] > ROI_MIN_Z, pts[:,2] < ROI_MAX_Z
    ])
    trailer = o3d.geometry.PointCloud()
    if mask.any():
        trailer.points = o3d.utility.Vector3dVector(pts[mask])
    return trailer

# JSON 저장
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    dbg(f"Saved JSON {path}")

# PCD 저장
def save_pcd(pcd: o3d.geometry.PointCloud, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)
    dbg(f"Saved PCD {path} ({len(pcd.points)} pts)")

# 필드 시계열 추출
def extract_series(msgs, fields, t0, t1):
    series = defaultdict(list)
    for m in msgs:
        ts = to_sec(get_time(m))
        if t0 < ts <= t1:
            for fld in fields:
                v = m
                for attr in fld.split('.'):
                    v = getattr(v, attr)
                series[fld.replace('.', '_')].append(v)
    return series

# 메인: 데이터셋 생성
def generate_dataset(bag, output):
    rclpy.init()
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=bag, storage_id='sqlite3'),
        ConverterOptions('', '')
    )
    topics = {
        '/gt_hitch_angle': PoseStamped,
        '/vehicle/imu': Imu,
        '/trailer/imu': Imu,
        '/vehicle/steering_angle': SteeringReport,
        '/vehicle/velocity': VelocityReport,
        '/vehicle/localization': Odometry,
        '/trailer/localizaton': Odometry,
        '/sensing/lidar/top/rectified/pointcloud_ex': PointCloud2
    }
    data = {t: [] for t in topics}
    while reader.has_next():
        topic, buf, _ = reader.read_next()
        if topic in data:
            data[topic].append(deserialize_message(buf, topics[topic]))

    lidar = data['/sensing/lidar/top/rectified/pointcloud_ex']
    gt    = data['/gt_hitch_angle']
    imu_fields = [
        'linear_acceleration.x','linear_acceleration.y','linear_acceleration.z',
        'angular_velocity.x','angular_velocity.y','angular_velocity.z'
    ]
    total = len(lidar)
    print(f"[INFO] Frames to process: {total}")

    for i in range(1, total):
        t0 = to_sec(get_time(lidar[i-1]))
        t1 = to_sec(get_time(lidar[i]))
        fd = os.path.join(output, f"frame_{i:06d}")
        print(f"[INFO] Frame {i}/{total}")

        # 1) GT 히치각
        closest_gt = min(gt, key=lambda m: abs(to_sec(get_time(m)) - t1))
        save_json({'orientation_z': closest_gt.pose.orientation.z}, os.path.join(fd, 'hitch_angle.json'))

        # 2) vehicle_imu
        vi = extract_series(data['/vehicle/imu'], imu_fields, t0, t1)
        save_json(vi, os.path.join(fd, 'vehicle_imu.json'))

        # 3) steering
        st = extract_series(data['/vehicle/steering_angle'], ['steering_tire_angle'], t0, t1)
        save_json(st, os.path.join(fd, 'steering.json'))

        # 4) velocity (longitudinal)
        vl = extract_series(data['/vehicle/velocity'], ['longitudinal_velocity'], t0, t1)
        save_json(vl, os.path.join(fd, 'velocity.json'))

        # 5) trailer_point (ROI 포함 z축)
        raw = pc2_to_o3d(lidar[i])
        tr = filter_trailer(raw)
        save_pcd(tr, os.path.join(fd, 'trailer_point.pcd'))

        if FULL_CONVERT:
            # trailer_imu
            ti = extract_series(data['/trailer/imu'], imu_fields, t0, t1)
            save_json(ti, os.path.join(fd, 'trailer_imu.json'))
            # vehicle_odom & trailer_odom (quaternion)
            for name, tp in [('vehicle_odom','/vehicle/localization'), ('trailer_odom','/trailer/localizaton')]:
                quats = []
                for m in data[tp]:
                    ts = to_sec(get_time(m))
                    if t0 < ts <= t1:
                        q = m.pose.pose.orientation
                        quats.append({'x':q.x,'y':q.y,'z':q.z,'w':q.w})
                save_json({'orientation': quats}, os.path.join(fd, f"{name}.json"))
            # raw lidar
            save_pcd(raw, os.path.join(fd, 'lidar_raw.pcd'))

    rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bag')
    parser.add_argument('output')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--full-convert', action='store_true')
    args = parser.parse_args()
    DEBUG = args.debug
    FULL_CONVERT = args.full_convert
    generate_dataset(args.bag, args.output)
