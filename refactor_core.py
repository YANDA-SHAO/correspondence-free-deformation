from pathlib import Path
from textwrap import dedent

ROOT = Path(".")
SRC = ROOT / "src"

# backup old entry scripts
for name in ["train.py", "eval.py"]:
    p = SRC / name
    if p.exists():
        backup = SRC / f"{name}.before_modular.bak"
        backup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

# 1. model
(SRC / "models" / "joint_uw_model.py").write_text(dedent(r'''
import torch
import torch.nn as nn


class JointUWProbabilisticMLP(nn.Module):
    def __init__(
        self,
        y_dim: int,
        U_dim: int,
        K_obs: int,
        N_vertices: int,
        z_dim: int = 32,
        hidden_dim: int = 256,
        logvar_min: float = -8.0,
        logvar_max: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.y_dim = y_dim
        self.U_dim = U_dim
        self.K_obs = K_obs
        self.N_vertices = N_vertices
        self.z_dim = z_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.z_mu_head = nn.Linear(hidden_dim, z_dim)
        self.z_logvar_head = nn.Linear(hidden_dim, z_dim)

        self.decoder_backbone = nn.Sequential(
            nn.Linear(y_dim + z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.U_mu_head = nn.Linear(hidden_dim, U_dim)
        self.U_logvar_head = nn.Linear(hidden_dim, U_dim)
        self.W_head = nn.Linear(hidden_dim, K_obs * N_vertices)

    def reparameterize(self, mu, logvar, sample_z=True):
        if not sample_z:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, y_norm, sample_z=True):
        h = self.encoder(y_norm)
        z_mu = self.z_mu_head(h)
        z_logvar = self.z_logvar_head(h).clamp(min=-12.0, max=8.0)

        z = self.reparameterize(z_mu, z_logvar, sample_z=sample_z)
        hd = self.decoder_backbone(torch.cat([y_norm, z], dim=-1))

        U_mu = self.U_mu_head(hd)
        U_logvar = self.U_logvar_head(hd).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        W_logits = self.W_head(hd).view(-1, self.K_obs, self.N_vertices)

        return U_mu, U_logvar, W_logits, z_mu, z_logvar
''').strip() + "\n", encoding="utf-8")

# 2. projection
(SRC / "geometry" / "projection.py").write_text(dedent(r'''
import torch


def torch_perspective_project(X_world, Kmat, R, t):
    """
    X_world: [B, K, 3]
    Kmat:    [B, 3, 3]
    R:       [B, 3, 3]
    t:       [B, 3]

    Returns:
        uv:    [B, K, 2]
        X_cam: [B, K, 3]
    """
    X_cam = torch.einsum("bij,bkj->bki", R, X_world) + t[:, None, :]

    z = X_cam[..., 2:3].clamp_min(1e-6)
    x_norm = X_cam[..., 0:1] / z
    y_norm = X_cam[..., 1:2] / z

    fx = Kmat[:, 0, 0].view(-1, 1, 1)
    fy = Kmat[:, 1, 1].view(-1, 1, 1)
    cx = Kmat[:, 0, 2].view(-1, 1, 1)
    cy = Kmat[:, 1, 2].view(-1, 1, 1)

    u = fx * x_norm + cx
    v = fy * y_norm + cy
    uv = torch.cat([u, v], dim=-1)

    return uv, X_cam
''').strip() + "\n", encoding="utf-8")

# 3. observation
(SRC / "geometry" / "observation.py").write_text(dedent(r'''
from typing import Tuple

import torch
import torch.nn.functional as F

from geometry.projection import torch_perspective_project


def observation_operator_softW(
    U_phys: torch.Tensor,
    W_logits: torch.Tensor,
    X0: torch.Tensor,
    Kmat: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    normalize_y: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    U_phys:   [B, N, 3]
    W_logits: [B, K, N]
    X0:       [B, N, 3]

    Returns:
        y_hat: [B, K, 2]
        W:     [B, K, N]
    """
    W = F.softmax(W_logits, dim=-1)

    X_surface = torch.einsum("bkn,bnd->bkd", W, X0)
    U_surface = torch.einsum("bkn,bnd->bkd", W, U_phys)

    uv0, _ = torch_perspective_project(X_surface, Kmat, R, t)
    uvt, _ = torch_perspective_project(X_surface + U_surface, Kmat, R, t)

    y_pixel_hat = uvt - uv0

    if normalize_y:
        fx = Kmat[:, 0, 0].view(-1, 1, 1)
        return y_pixel_hat / fx, W

    return y_pixel_hat, W


def observation_operator_gtW(
    U_phys: torch.Tensor,
    X0: torch.Tensor,
    sparse_idx: torch.Tensor,
    sparse_val: torch.Tensor,
    Kmat: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    normalize_y: bool = True,
) -> torch.Tensor:
    """
    Uses sparse barycentric ground-truth W.
    sparse_idx: [B, K, 3]
    sparse_val: [B, K, 3]
    """
    _, Kobs, _ = sparse_idx.shape

    idx_exp = sparse_idx.long().unsqueeze(-1).expand(-1, -1, -1, 3)
    X0_exp = X0[:, None, :, :].expand(-1, Kobs, -1, -1)
    U_exp = U_phys[:, None, :, :].expand(-1, Kobs, -1, -1)

    X_tri = torch.gather(X0_exp, dim=2, index=idx_exp)
    U_tri = torch.gather(U_exp, dim=2, index=idx_exp)

    w = sparse_val.unsqueeze(-1)
    X_surface = torch.sum(w * X_tri, dim=2)
    U_surface = torch.sum(w * U_tri, dim=2)

    uv0, _ = torch_perspective_project(X_surface, Kmat, R, t)
    uvt, _ = torch_perspective_project(X_surface + U_surface, Kmat, R, t)

    y_pixel_hat = uvt - uv0

    if normalize_y:
        fx = Kmat[:, 0, 0].view(-1, 1, 1)
        return y_pixel_hat / fx

    return y_pixel_hat
''').strip() + "\n", encoding="utf-8")

# 4. losses
(SRC / "losses" / "loss_core.py").write_text(dedent(r'''
import torch
import torch.nn.functional as F


def mse_loss(target, pred):
    return torch.mean((target - pred) ** 2)


def gaussian_nll_loss(target, mu, logvar):
    return 0.5 * torch.mean(
        logvar + (target - mu).pow(2) / torch.exp(logvar)
    )


def kl_loss(z_mu, z_logvar):
    return -0.5 * torch.mean(
        1.0 + z_logvar - z_mu.pow(2) - z_logvar.exp()
    )


def soft_ce_loss_from_sparse(logits, sparse_idx, sparse_val):
    """
    logits:     [B, K, N]
    sparse_idx: [B, K, 3]
    sparse_val: [B, K, 3]
    """
    logp = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(logp, dim=-1, index=sparse_idx.long())
    ce = -(sparse_val * gathered).sum(dim=-1)
    return ce.mean()
''').strip() + "\n", encoding="utf-8")

# 5. direct model
(SRC / "models" / "direct_model.py").write_text(dedent(r'''
import torch.nn as nn


class DirectModel(nn.Module):
    """
    Direct baseline: y -> U.
    No latent correspondence W.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, y):
        return self.net(y)
''').strip() + "\n", encoding="utf-8")

print("Done.")
print("Backups created:")
print("  src/train.py.before_modular.bak")
print("  src/eval.py.before_modular.bak")
print("Updated:")
print("  src/models/joint_uw_model.py")
print("  src/models/direct_model.py")
print("  src/geometry/projection.py")
print("  src/geometry/observation.py")
print("  src/losses/loss_core.py")