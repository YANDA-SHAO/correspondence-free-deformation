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
