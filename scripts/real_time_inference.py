#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry
from autoware_auto_vehicle_msgs.msg import VelocityReport, SteeringReport
from geometry_msgs.msg import PoseStamped
from sensor_msgs_py.point_cloud2 import read_points
from scipy.spatial import cKDTree
from torch_geometric.data import Data
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hitchnet.hitchnet.models import GATSpatialTemporal
from collections import deque
import time
from std_msgs.msg import Float32

RAD_2_DEG = 360.0/np.pi
DEG_2_RAD = np.pi/360.0

class RealTimeInference(Node):
    def __init__(self):
        super().__init__('real_time_inference')

        # Subscribers using message_filters
        self.pointcloud_sub = self.create_subscription(PointCloud2, '/vehicle/cropped/pointcloud', self.pointcloud_callback, 10)
        self.odometry_sub = self.create_subscription(Odometry, '/vehicle/localization', self.odometry_callback, 10)
        self.velocity_sub = self.create_subscription(VelocityReport, '/vehicle/velocity', self.velocity_callback, 10)
        self.steering_sub = self.create_subscription(SteeringReport, '/vehicle/steering_angle', self.steering_callback, 10)

        # Publisher
        self.hitch_angle_pub = self.create_publisher(PoseStamped, '/predicted_hitch_angle', 10)
        self.gat_hitch_angle_pub = self.create_publisher(PoseStamped, '/predicted_hitch_angle_before_filter', 10)
        self.process_time_pub = self.create_publisher(Float32, '/gnn_process_time', 10)

        # Load model (replace with your actual model path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model('src/st_gnn/model/gat_tcn_3/best_gat_st_gnn_model.pth') # best_gat_st_gnn_model, gat_st_gnn_model

        self.pred_hitch = 0.0
        self.error_sum = 0.0
        self.error_max = 0.0
        self.count = 0
        self.velocity_buffer = deque(maxlen=10)
        self.steering_buffer = deque(maxlen=10)
        self.angular_buffer = deque(maxlen=10)
        self.start_time = time.time()
        self.end_time = time.time()

        # 예측 결과를 스무딩하기 위한 가중이동평균 버퍼 (최대 5개 값 저장)
        self.prediction_buffer = deque(maxlen=5)

    def velocity_callback(self, velocity):
        self.velocity_buffer.append([velocity.longitudinal_velocity])

    def steering_callback(self, steering):
        self.steering_buffer.append([steering.steering_tire_angle])
    
    def odometry_callback(self, odometry):
        self.angular_buffer.append([odometry.twist.twist.angular.z])

    def load_model(self, model_path):
        # Load the model
        model = GATSpatialTemporal(
            graph_input_dim=3,
            graph_hidden_dim=64, 
            temporal_dim=64,
            num_heads=3,         
            output_dim=1
        ).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.get_logger().info(f"Model loaded from {model_path}")
        return model

    def pointcloud_callback(self, lidar_msg):
        self.start_time = time.time()
        # 충분한 temporal 데이터를 모았는지 확인
        if len(self.velocity_buffer) == 10 and len(self.steering_buffer) == 10 and len(self.angular_buffer) == 10:

            # PointCloud2 메시지를 NumPy 배열로 변환
            try:
                point_list = list(read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True))
                if not point_list:
                    self.get_logger().warning("PointCloud2 message contains no valid points.")
                    return  # 유효한 포인트가 없으면 건너뜁니다.

                point_cloud = np.array(
                    [(p[0], p[1], p[2]) for p in point_list],
                    dtype=np.float32
                )
            except Exception as e:
                self.get_logger().error(f"Error processing PointCloud2 message: {e}")
                return

            velocity = list(self.velocity_buffer)
            steering = list(self.steering_buffer)
            angular = list(self.angular_buffer)
            graph = self.create_graph_from_point_cloud(point_cloud)

            velocity = torch.tensor(velocity, dtype=torch.float32).unsqueeze(0).to(self.device)
            steering = torch.tensor(steering, dtype=torch.float32).unsqueeze(0).to(self.device)
            angular = torch.tensor(angular, dtype=torch.float32).unsqueeze(0).to(self.device)
            graph = graph.to(self.device)

            # Predict hitch angle
            with torch.no_grad():
                prediction = self.model(graph, velocity, steering, angular, graph.batch)
                hitch_angle = prediction.item()
                self.end_time = time.time()
                process_time = (self.end_time - self.start_time) * 1000
                self.get_logger().info(f'Predict step time: {process_time:.3f} ms')

            # 가중이동평균 필터 적용
            self.prediction_buffer.append(hitch_angle)
            # 최대 버퍼 길이가 5라고 가정할 때, 고정 가중치 배열 정의
            custom_weights = np.array([1.0, 2.0, 5.0, 7.0, 15.0])
            # 예측 버퍼에 저장된 값의 개수에 맞게 최신 n개의 가중치를 사용 (오래된 값부터 가장 낮은 가중치)
            weights = custom_weights[-len(self.prediction_buffer):]
            smoothed_hitch = sum(w * p for w, p in zip(weights, self.prediction_buffer)) / weights.sum()

            # Publish filtered (smoothed) hitch angle
            pose = PoseStamped()
            pose.header = lidar_msg.header
            pose.pose.orientation.z = smoothed_hitch
            self.pred_hitch = smoothed_hitch
            self.hitch_angle_pub.publish(pose)

            pose.pose.orientation.z = hitch_angle
            self.gat_hitch_angle_pub.publish(pose)

            ps = Float32()
            ps.data = process_time
            self.process_time_pub.publish(ps)

    def create_graph_from_point_cloud(self, point_cloud, k=10):
        """Convert a point cloud into a graph using k-NN."""
        tree = cKDTree(point_cloud)
        edges = []
        num_nodes = point_cloud.shape[0]
        for i, point in enumerate(point_cloud):
            _, indices = tree.query(point, k=min(k + 1, num_nodes))
            for idx in indices[1:]:
                if idx < num_nodes: 
                    edges.append([i, idx])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(point_cloud, dtype=torch.float)  # PyTorch tensor
        return Data(x=x, edge_index=edge_index)

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeInference()  # Create an instance of the RealTimeInference node

    try:
        rclpy.spin(node)  # ROS2 spin to keep the node running
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down RealTimeInference node...")
    finally:
        rclpy.shutdown()  # Clean up and shut down ROS2

if __name__ == '__main__':
    main()
