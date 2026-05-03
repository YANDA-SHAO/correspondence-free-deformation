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
