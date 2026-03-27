# models/rule_based.py
import torch
import torch.nn as nn


class _RuleBase(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def _extract_points(self, batch):
        pcd = batch["pcd"][:, :, :2]  # (B,N,2)
        mask = batch.get("pcd_mask")
        out = []
        for b in range(pcd.shape[0]):
            pb = pcd[b]
            if mask is not None:
                pb = pb[mask[b] > 0]
            out.append(pb)
        return out

    def _fallback_theta(self, pb, dtype, device):
        if pb.shape[0] == 0:
            return torch.tensor(0.0, dtype=dtype, device=device)
        c = pb.mean(dim=0)
        return torch.atan2(-c[1], -c[0])

    @staticmethod
    def _wrap_pi(theta):
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    def _dir_to_theta(self, v, c):
        # Resolve 180-deg ambiguity with centroid direction.
        if torch.dot(v, c) < 0:
            v = -v
        # hitch-angle convention: direction (-1,0) -> angle 0
        theta = torch.atan2(v[1], v[0]) - torch.pi
        return self._wrap_pi(theta)

    def _stack_sincos(self, thetas, dtype, device):
        if len(thetas) == 0:
            return torch.zeros((0, 2), dtype=dtype, device=device)
        th = torch.stack(thetas, dim=0)
        out = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
        n = torch.norm(out, dim=-1, keepdim=True).clamp_min(self.eps)
        return out / n


class RuleBasedCentroid(_RuleBase):
    """
    Rule-based baseline: predict hitch angle from centroid direction.
    Uses pcd (optionally masked) and returns (cos, sin).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__(eps=eps)

    def forward(self, batch):
        pcd = batch["pcd"]  # (B,N,3)
        mask = batch.get("pcd_mask")  # (B,N) or None
        if mask is not None:
            m = mask.to(pcd.dtype)
            masked = pcd * m.unsqueeze(-1)
            x = masked[:, :, 0]
            y = masked[:, :, 1]
            neg_inf = torch.full_like(x, -1e9)
            pos_inf = torch.full_like(x, 1e9)
            x_min = torch.where(m > 0, x, pos_inf).min(dim=1).values
            x_max = torch.where(m > 0, x, neg_inf).max(dim=1).values
            y_min = torch.where(m > 0, y, pos_inf).min(dim=1).values
            y_max = torch.where(m > 0, y, neg_inf).max(dim=1).values
            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5
        else:
            x = pcd[:, :, 0]
            y = pcd[:, :, 1]
            cx = (x.min(dim=1).values + x.max(dim=1).values) * 0.5
            cy = (y.min(dim=1).values + y.max(dim=1).values) * 0.5

        # Flip direction to match hitch-angle convention (avoid 180° offset)
        angle = torch.atan2(-cy, -cx)
        out = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        # normalize for safety
        n = torch.norm(out, dim=-1, keepdim=True).clamp_min(self.eps)
        return out / n


class RuleBasedPCA(_RuleBase):
    """
    Rule-based PCA baseline:
    1) compute centered 2D covariance
    2) take principal axis (largest eigenvector)
    3) resolve 180-deg ambiguity using centroid direction
    4) map axis to hitch-angle convention and return (cos, sin)
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__(eps=eps)

    def forward(self, batch):
        pts_list = self._extract_points(batch)
        dtype = batch["pcd"].dtype
        device = batch["pcd"].device
        thetas = []
        for pb in pts_list:
            if pb.shape[0] < 2:
                thetas.append(self._fallback_theta(pb, dtype, device))
                continue
            c = pb.mean(dim=0)
            xc = pb - c
            cov = (xc.t() @ xc) / max(pb.shape[0] - 1, 1)
            _evals, evecs = torch.linalg.eigh(cov)
            v_major = evecs[:, -1]
            v_minor = evecs[:, 0]

            # Choose axis (major/minor) better aligned with hitch->centroid direction.
            # This avoids frequent 90-deg failure when partial/occluded points
            # make the lateral axis dominate as PCA major axis.
            cn = torch.norm(c)
            if cn < self.eps:
                v = v_major
            else:
                c_dir = c / cn
                s_major = torch.abs(torch.dot(v_major, c_dir))
                s_minor = torch.abs(torch.dot(v_minor, c_dir))
                v = v_major if s_major >= s_minor else v_minor

            nv = torch.norm(v).clamp_min(self.eps)
            v = v / nv
            thetas.append(self._dir_to_theta(v, c))
        return self._stack_sincos(thetas, dtype=dtype, device=device)


class RuleBasedOLS(_RuleBase):
    """
    Rule-based OLS baseline:
    1) fit y = beta*x (+ optional inverse form x = beta2*y for stability)
    2) convert slope to direction vector
    3) resolve sign with centroid direction
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__(eps=eps)

    def forward(self, batch):
        pts_list = self._extract_points(batch)
        dtype = batch["pcd"].dtype
        device = batch["pcd"].device
        thetas = []
        for pb in pts_list:
            if pb.shape[0] < 2:
                thetas.append(self._fallback_theta(pb, dtype, device))
                continue
            c = pb.mean(dim=0)
            x = pb[:, 0]
            y = pb[:, 1]

            # y = b*x (through origin in hitch-centered frame)
            den1 = torch.sum(x * x).clamp_min(self.eps)
            b1 = torch.sum(x * y) / den1
            y_hat = b1 * x
            mse1 = torch.mean((y - y_hat) ** 2)

            # x = b2*y
            den2 = torch.sum(y * y).clamp_min(self.eps)
            b2 = torch.sum(x * y) / den2
            x_hat = b2 * y
            mse2 = torch.mean((x - x_hat) ** 2)

            if mse1 <= mse2:
                v = torch.stack([
                    torch.tensor(1.0, dtype=dtype, device=device),
                    b1.to(dtype=dtype),
                ])
            else:
                v = torch.stack([
                    b2.to(dtype=dtype),
                    torch.tensor(1.0, dtype=dtype, device=device),
                ])
            v = v / torch.norm(v).clamp_min(self.eps)
            thetas.append(self._dir_to_theta(v, c))
        return self._stack_sincos(thetas, dtype=dtype, device=device)


class RuleBasedMLE(_RuleBase):
    """
    Rule-based MLE-style baseline:
    - Build a synthetic reference cloud along canonical hitch axis (-x).
    - Search theta that minimizes log det of covariance of
      X(theta) = [X_ref, R(-theta) X_obs], which approximates best alignment.
    """

    def __init__(self, eps: float = 1e-6, theta_min_deg: float = -100.0, theta_max_deg: float = 100.0, theta_step_deg: float = 1.0):
        super().__init__(eps=eps)
        self.theta_min_deg = float(theta_min_deg)
        self.theta_max_deg = float(theta_max_deg)
        self.theta_step_deg = float(theta_step_deg)

    def _objective(self, x_ref, x_obs, theta_rad):
        c = torch.cos(theta_rad)
        s = torch.sin(theta_rad)
        r = torch.stack([torch.stack([c, s]), torch.stack([-s, c])])  # R(-theta)
        x_rot = (r @ x_obs.t()).t()
        x = torch.cat([x_ref, x_rot], dim=0)
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.t() @ x) / max(x.shape[0] - 1, 1)
        cov = cov + self.eps * torch.eye(2, device=x.device, dtype=x.dtype)
        sign, logabsdet = torch.linalg.slogdet(cov)
        # invalid SPD fallback
        if sign <= 0:
            return torch.tensor(1e9, device=x.device, dtype=x.dtype)
        return logabsdet

    def forward(self, batch):
        pts_list = self._extract_points(batch)
        dtype = batch["pcd"].dtype
        device = batch["pcd"].device
        thetas = []
        theta_grid = torch.arange(
            self.theta_min_deg,
            self.theta_max_deg + 1e-6,
            self.theta_step_deg,
            device=device,
            dtype=dtype,
        ) * torch.pi / 180.0

        for pb in pts_list:
            if pb.shape[0] < 2:
                thetas.append(self._fallback_theta(pb, dtype, device))
                continue

            # synthetic reference along canonical axis (-x)
            r = torch.norm(pb, dim=1)
            x_ref = torch.stack([-r, torch.zeros_like(r)], dim=1)

            best_val = None
            best_theta = None
            for th in theta_grid:
                v = self._objective(x_ref, pb, th)
                if best_val is None or v < best_val:
                    best_val = v
                    best_theta = th

            if best_theta is None:
                thetas.append(self._fallback_theta(pb, dtype, device))
            else:
                thetas.append(self._wrap_pi(best_theta))
        return self._stack_sincos(thetas, dtype=dtype, device=device)
