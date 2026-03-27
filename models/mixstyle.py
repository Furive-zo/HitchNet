import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """
    MixStyle (source-domain generalization regularizer).
    Applies instance-level feature statistic mixing during training.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x):
        if (not self.training) or self.p <= 0.0:
            return x
        if torch.rand(1, device=x.device) > self.p:
            return x

        b = x.size(0)
        if b < 2:
            return x

        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig

        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lmda = self.beta.sample((b, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_mix = mu * lmda + mu2 * (1.0 - lmda)
        sig_mix = sig * lmda + sig2 * (1.0 - lmda)

        return x_norm * sig_mix + mu_mix
