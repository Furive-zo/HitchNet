#!/usr/bin/env python3
import rclpy
import math
import time
from rclpy.node import Node
import numpy as np
from scipy.integrate import solve_ivp
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from autoware_auto_vehicle_msgs.msg import VelocityReport, SteeringReport
from std_msgs.msg import Float32MultiArray, Float32
from scipy.spatial.transform import Rotation as R
from message_filters import Subscriber, ApproximateTimeSynchronizer

TIME_STEP = 0.01

class TrailerHitchEKF(Node):
    def __init__(self):
        super().__init__('ekf_hitch_angle')

        # Initialize the state (hitch angle and rate of change)
        self.state = np.array([[0.0], [0.0]])  # [theta_hitch, theta_dot_hitch]
        self.P = np.eye(2) * 0.001  # State covariance matrix

        # Process and measurement noise
        self.Q = np.diag([0.001, 0.001])  # Process noise covariance
        self.R_imu = np.eye(1) * 0.05   # Measurement noise from IMU
        self.R_lidar = 0.05  # Measurement noise from LiDAR

        # Kinematic model parameters
        self.L1 = 1.145 #2.70 #1.145 Length for tractor wheel base 
        self.L2 = 1.804 #1.774 #1.804 # Length for trailer wheel base 
        self.e = 0.704 #0.884 #0.704  # Length between hitch point and tractor base link
        self.delta = 0.0  # Steering angle
        self.v1 = 0.0  # Tractor forward velocity
        self.start_time = time.time()
        self.end_time = time.time()
        self.hitch_1 = 0.0
        self.lidar_stamp = PoseStamped()
        
        # ROS2 Subscribers
        self.trailer_imu_sub = Subscriber(self, Imu, '/sensing/imu/xsens/imu_raw')
        self.tractor_imu_sub = Subscriber(self, Odometry, '/localization/kinematic_state')
        self.lidar_hitch_angle_sub = self.create_subscription(PoseStamped, '/lidar_hitch_angle', self.lidar_hitch_angle_callback, 10)
        self.gnn_hitch_angle_sub = self.create_subscription(PoseStamped, '/predicted_hitch_angle', self.gnn_hitch_angle_callback, 10)
        self.vel_sub = self.create_subscription(VelocityReport, '/vehicle/velocity', self.velocity_callback, 10)
        self.steer_sub = self.create_subscription(SteeringReport, '/vehicle/steering_angle', self.steering_callback, 10)

        # ROS2 Publisher
        self.hitch_angle_pub = self.create_publisher(PoseStamped, '/ekf_hitch_angle', 10)
        self.kin_hitch_angle_pub = self.create_publisher(PoseStamped, '/kin_hitch_angle', 10)

        # Timer for periodic updates (prediction step)
        self.timer = self.create_timer(TIME_STEP, self.predict_step)
        # Approximate time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.trailer_imu_sub, self.tractor_imu_sub],
            queue_size=10,
            slop=0.1 # Allowable time difference (seconds)
        )
        self.sync.registerCallback(self.sync_callback)

    def sync_callback(self, trailer_msg, tractor_msg):
        self.get_logger().info(f"Received synchronized messages:\n"
                               f"Trailer timestamp: {trailer_msg.header.stamp.sec}.{trailer_msg.header.stamp.nanosec}\n"
                               f"Tractor timestamp: {tractor_msg.header.stamp.sec}.{tractor_msg.header.stamp.nanosec}")
        # Process synchronized messages here
        trailer_R = [trailer_msg.orientation.x, trailer_msg.orientation.y, trailer_msg.orientation.z, trailer_msg.orientation.w]
        trailer_theta = self.quaternion_to_yaw(trailer_R)
        trailer_cov = trailer_msg.orientation_covariance[8]

        tractor_R = [tractor_msg.pose.pose.orientation.x, tractor_msg.pose.pose.orientation.y, tractor_msg.pose.pose.orientation.z, tractor_msg.pose.pose.orientation.w]
        tractor_theta = self.quaternion_to_yaw(tractor_R)
        tractor_cov = tractor_msg.pose.covariance[35]

        hitch_imu = trailer_theta - tractor_theta
        R_imu = np.array([[trailer_cov + tractor_cov]])

        # Update step with IMU data
        self.update_step(np.array([[hitch_imu]]), R_imu)

    def predict_step(self):
        self.start_time = time.time()
        # Prediction step using kinematic model
        theta_hitch = self.state[0, 0]
        theta_kin = self.hitch_1 
        # self.get_logger().info(f"Received theta_hitch:{theta_hitch}")
        
        # 보정항 (e*cos(theta)/L2 - 1): ekf
        w1 = self.v1 * np.tan(self.delta) / self.L1 * ((self.e * np.cos(theta_hitch) / self.L2) - 1)
        w2 = self.v1 * np.sin(theta_hitch) / self.L2

        # 보정항 (e*cos(theta)/L2 - 1): kin
        kin_w1 = self.v1 * np.tan(self.delta) / self.L1 * ((self.e * np.cos(theta_kin) / self.L2) - 1)
        kin_w2 = self.v1 * np.sin(theta_kin) / self.L2

        theta_dot_hitch = w1 - w2      
        theta_dot_kin_hitch = kin_w1 - kin_w2
        # State transition
        self.hitch_1 += theta_dot_kin_hitch * TIME_STEP
        self.state[0, 0] += theta_dot_hitch * TIME_STEP  # Update angle with time step
        self.state[1, 0] = theta_dot_hitch
        
        # Jacobian for linearizing the model (simple example)
        F = np.eye(2)
        F[0, 1] = TIME_STEP  # Time step for updating angle with rate

        # Update covariance
        self.update_dynamic_noise()
        self.P = F @ self.P @ F.T + self.Q

    def update_dynamic_noise(self):
        # 속도에 따른 프로세스 노이즈 조정
        base_Q = 0.001
        Q_factor = 0.01  # 속도에 따른 증가량 (튜닝 필요)
        if self.v1 > 0.0:
            new_Q_value = base_Q + Q_factor * abs(self.v1)
        else:
            new_Q_value = 1.0
        self.Q = np.diag([new_Q_value, new_Q_value])
        
        # 조향각에 따른 관측 노이즈 조정 (IMU의 경우)
        base_R = 0.05
        R_factor = 0.005  # 조향각에 따른 증가량 (튜닝 필요)
        new_R_value = base_R - R_factor * abs(self.delta)
        self.R_lidar = new_R_value

    def update_step(self, measurement, R):
        # Kalman gain
        H = np.array([[1, 0]])  # Measurement matrix (only angle)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state with measurement
        y = measurement - H @ self.state
        self.state += K @ y

        # Update covariance
        self.P = (np.eye(2) - K @ H) @ self.P

    def velocity_callback(self, msg):
        self.v1 = msg.longitudinal_velocity

    def steering_callback(self, msg):
        self.delta = msg.steering_tire_angle

    def lidar_hitch_angle_callback(self, msg):
        # Extract hitch angle and covariance from LiDAR
        theta_lidar = msg.pose.pose.orientation.z
        covariance_lidar = msg.pose.covariance[0]
        # print(f"theta: {theta_lidar}, cov: {covariance_lidar}")

        R_lidar = np.array([[covariance_lidar]])
        # self.lidar_stamp.header = msg.header
        
        # Update step with LiDAR data
        self.update_step(np.array([[theta_lidar]]), R_lidar)

    def gnn_hitch_angle_callback(self, msg):
        # Extract hitch angle and covariance from LiDAR
        theta_gnn = msg.pose.orientation.z
        # print(f"theta: {theta_lidar}, cov: {covariance_lidar}")

        R_lidar = np.array([[self.R_lidar]])
        self.lidar_stamp.header = msg.header
        
        # Update step with LiDAR data
        self.update_step(np.array([[theta_gnn]]), R_lidar)
        self.publish_hitch_angle()

    def publish_hitch_angle(self):
        # Prepare message with estimated hitch angle
        msg = PoseStamped()
        msg.header = self.lidar_stamp.header
        msg.pose.orientation.z = self.state[0, 0]

        msg2 = PoseStamped()
        msg2.header = self.lidar_stamp.header
        msg2.pose.orientation.z = self.hitch_1

        self.hitch_angle_pub.publish(msg)
        self.kin_hitch_angle_pub.publish(msg2)
        self.end_time = time.time()
        self.get_logger().info(f'Predict step time: {self.end_time - self.start_time:.6f} sec')  # 처리 시간 출력

    def quaternion_to_yaw(self, q):
        # Convert quaternion to a Rotation object
        rotation = R.from_quat(q)
        
        # Extract yaw (rotation around Z-axis) from the quaternion
        yaw = rotation.as_euler('zyx', degrees=False)[0]
        
        return yaw

def main(args=None):
    rclpy.init(args=args)
    ekf_node = TrailerHitchEKF()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
