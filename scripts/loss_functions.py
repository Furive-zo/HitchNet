import torch
import torch.nn.functional as F
import numpy as np

def weighted_loss(predictions, targets, weight_factor=5.0):
    """ 각도 크기에 비례한 가중 RMSE """
    weights = 1.0 + weight_factor * torch.abs(targets)
    diff = predictions - targets
    mse = torch.mean(weights * diff ** 2)
    rmse = torch.sqrt(mse)
    return rmse

def angular_loss_cosine(predictions, targets):
    """ 코사인 기반 각도 유사성 RMSE """
    pred_vec = torch.stack([torch.cos(predictions), torch.sin(predictions)], dim=-1)
    target_vec = torch.stack([torch.cos(targets), torch.sin(targets)], dim=-1)
    mse = F.mse_loss(pred_vec, target_vec)
    rmse = torch.sqrt(mse)
    return rmse
    
def combined_loss(predictions, targets, weight_factor=3.0, alpha=0.7, beta=0.3):
    rmse = weighted_loss(predictions, targets, weight_factor)
    angular_distance = angular_loss_cosine(predictions, targets)

    # print(f"rmse: {rmse}, angular_distance: {angular_distance}, rate_loss: {rate_loss}, stop_penalty: {stop_penalty}")

    total_loss = rmse + alpha * angular_distance * beta

    # RMSE (deg) 계산
    rmse_rad = torch.sqrt(torch.mean((predictions - targets) ** 2))
    rmse_deg = np.rad2deg(rmse_rad.cpu().item())

    return total_loss, rmse_deg
