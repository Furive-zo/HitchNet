# models/__init__.py
from .builder import build_hitchnet
# from .pointnet_lstm import PointNetLSTM
# from .imu_gru import IMUGRU
# from .kinematic_imu_ukf import KinematicIMUUKF


def build_model(model_cfg):
    name = model_cfg["name"]

    if name == "HitchNet":
        return build_hitchnet(model_cfg)
    # elif name == "PointNet-LSTM":
    #     return PointNetLSTM(model_cfg)
    # elif name == "IMU-GRU":
    #     return IMUGRU(model_cfg)
    # elif name == "Kinematic-IMU-UKF":
    #     return KinematicIMUUKF(model_cfg)
    else:
        raise ValueError(f"Unknown model name: {name}")
