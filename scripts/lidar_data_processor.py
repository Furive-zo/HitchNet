#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from autoware_auto_vehicle_msgs.msg import VelocityReport, SteeringReport
from geometry_msgs.msg import PoseStamped
from sensor_msgs_py.point_cloud2 import read_points
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import numpy as np
import torch
from message_filters import ApproximateTimeSynchronizer, Subscriber
from message_filters import SimpleFilter
from collections import deque
from nav_msgs.msg import Odometry
import os 

SCENARIO_TYPE = {
    1: 'dw',
    2: 'dw_up',
    3: 'reverse',
    4: 'uturn_l',
    5: 'uturn_r',
    6: 'lc'
}

SCENARIO_NAME = f'temporal_buffer/edge/{SCENARIO_TYPE[3]}_7'

class LidarDataProcessor(Node):
    def __init__(self):
        super().__init__('lidar_data_processor')

        # Subscribers for velocity, steering, angular, ground truth, and lidar point cloud
        self.label_sub = Subscriber(self, PoseStamped, '/gt_hitch_angle')
        self.lidar_sub = Subscriber(self, PointCloud2, '/sensing/lidar/concatenated/pointcloud')
        self.angular_sub = self.create_subscription(Imu, '/vehicle/imu_tmp', self.angular_callback, 10)
        self.odometry_sub = self.create_subscription(Odometry, '/vehicle/localization', self.odometry_callback, 10)
        self.velocity_sub = self.create_subscription(VelocityReport, '/vehicle/velocity', self.velocity_callback, 10)
        self.steering_sub = self.create_subscription(SteeringReport, '/vehicle/steering_angle', self.steering_callback, 10)

        # Synchronizer for all data
        self.sync = ApproximateTimeSynchronizer(
            [self.label_sub, self.lidar_sub],
            queue_size=10,
            slop=0.01
        )
        self.sync.registerCallback(self.sync_callback)

        self.data = []
        self.velocity_buffer = deque(maxlen=10)
        self.steering_buffer = deque(maxlen=10)
        self.angular_buffer = deque(maxlen=10)

    def velocity_callback(self, velocity):
        self.velocity_buffer.append([velocity.longitudinal_velocity])

    def steering_callback(self, steering):
        self.steering_buffer.append([steering.steering_tire_angle])
    
    def angular_callback(self, angular):
        self.angular_buffer.append([angular.angular_velocity.z])
    
    def odometry_callback(self, odometry):
        self.angular_buffer.append([odometry.twist.twist.angular.z])

    def sync_callback(self, label_msg, lidar_msg):
        print(f"callback sync {len(self.velocity_buffer), len(self.steering_buffer), len(self.angular_buffer)}")
        if len(self.velocity_buffer) == 10\
            and len(self.steering_buffer) == 10\
            and len(self.angular_buffer) == 10:
            # Extract velocity, steering, and label
            label = label_msg.pose.orientation.z
            timestamp = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec * 1e-9

            # Convert PointCloud2 to NumPy array
            try:
                point_list = list(read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True))
                if not point_list:
                    self.get_logger().warning("PointCloud2 message contains no valid points.")
                    return  # Skip this callback if no points are available

                # Convert to structured NumPy array
                point_cloud = np.array(
                    [(p[0], p[1], p[2]) for p in point_list],
                    dtype=np.float32
                )
            except Exception as e:
                self.get_logger().error(f"Error processing PointCloud2 message: {e}")
                return

            # Convert point cloud to graph
            graph = self.create_graph_from_point_cloud(point_cloud)

            # Store synchronized data
            self.data.append({
                "velocity": list(self.velocity_buffer),
                "steering": list(self.steering_buffer),
                "angular" : list(self.angular_buffer),
                "label": label,
                "timestamp": timestamp,
                "lidar_graph": graph
            })
            self.get_logger().info(f"\n === timestamp: {timestamp} ===\n Label: {label}")

    def create_graph_from_point_cloud(self, point_cloud, k=10):
        """
        Convert a point cloud into a graph using k-NN, and compute an additional edge attribute:
        the 2D top-view angle between points.
        
        Args:
            point_cloud (array-like): shape [N, F] (여기서 F>=3, x, y, z ...)
            k (int): 각 점에 대해 찾을 최근접 이웃의 수

        Returns:
            Data: PyG Data 객체, with fields:
                - x: 노드 feature, shape [N, F]
                - edge_index: 연결 정보를 담은 텐서, shape [2, E]
                - edge_attr: 각 edge의 top-view angle (radians), shape [E, 1]
        """
        tree = cKDTree(point_cloud)
        edges = []
        edge_attrs = []
        for i, point in enumerate(point_cloud):
            # 자기 자신 포함해 k+1개를 찾고, 자기 자신은 제외합니다.
            _, indices = tree.query(point, k=k+1)
            for idx in indices[1:]:
                edges.append([i, idx])
                # 탑뷰 각도 계산: x, y 좌표만 사용 (index 0: x, index 1: y)
                dx = point_cloud[idx][0] - point_cloud[i][0]
                dy = point_cloud[idx][1] - point_cloud[i][1]
                # arctan2를 통해 각도 계산 (radians)
                angle = np.arctan2(dy, dx)
                edge_attrs.append(angle)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape: [2, E]
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)   # shape: [E, 1]
        x = torch.tensor(point_cloud, dtype=torch.float)  # shape: [N, F]
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def save_to_pt(self, filepath=f'dataset/gat/{SCENARIO_NAME}'):
        """
        Save synchronized data to a PyTorch .pt file.
        """
        filename = f'{filepath}/synchronized_data.pt'
        os.makedirs(filepath, exist_ok=True)

        if not self.data:
            self.get_logger().warning("No data to save.")
            return

        torch.save(self.data, filename)
        self.get_logger().info(f"Data saved to {filename}")

    def shutdown_callback(self):
        """
        Save data when the node is shutting down.
        """
        self.save_to_pt()


def main(args=None):
    rclpy.init(args=args)
    node = LidarDataProcessor()
    node.get_logger().info(f'Scenario name : {SCENARIO_NAME}')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.shutdown_callback()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
