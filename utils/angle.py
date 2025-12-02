import torch
import numpy as np

def wrap_rad_torch(x):
    pi = torch.tensor(np.pi, device=x.device, dtype=x.dtype)
    return torch.remainder(x + pi, 2 * pi) - pi

def wrap_deg_torch(x):
    return torch.remainder(x + 180.0, 360.0) - 180.0
