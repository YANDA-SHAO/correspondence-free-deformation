# src/eval.py
"""
Final evaluation script for joint deformation + soft-correspondence model.

Designed for the final "real-like" setting:

Model input:
    y_t only, from sparse 2D tracking trajectories.

Model output:
    U_pred     [T,N,3]
    U_std      [T,N,3]
    W_pred     [T,K,N]
    y_hat      [T,K,2] obtained through differentiable observation operator

Important principle:
    observed_idx is NOT used.
    obs_face_vertices / obs_barycentric / W_gt are NOT used as model input.

For synthetic benchmark only, if U_seq and W_gt exist, this script computes:
    - U reconstruction errors
    - observation-space consistency y_hat vs y
    - W correspondence accuracy / CE
    - uncertainty coverage
    - confidence filtering metrics

For real deployment, minimum required files are:
    y.npy
    X0.npy
    camera_K.npy
    camera_R.npy
    camera_t.npy

If U_seq.npy or W_gt files are missing, evaluation still runs and saves predictions,
but skips GT metrics.
"""

import os
import csv
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_v2_surface", "test")
DEFAULT_WEIGHT_PATH = os.path.join(PROJECT_ROOT, "outputs", "weights", "joint_UW_surface_best.pth")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
DEFAULT_PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_sample_dirs(data_dir):
    data_dir = os.path.abspath(data_dir)

    if os.path.exists(os.path.join(data_dir, "y.npy")):
        return [data_dir]

    sample_dirs = sorted(glob.glob(os.path.join(data_dir, "sample_*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]

    if len(sample_dirs) == 0:
        raise ValueError(
            f"No valid sample found in {data_dir}. "
            f"Pass either one sample folder or a folder containing sample_* folders."
        )

    return sample_dirs


def sample_name_from_dir(sample_dir):
    return os.path.basename(os.path.normpath(sample_dir))


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_normalization(weight_path):
    norm_path = os.path.join(os.path.dirname(os.path.abspath(weight_path)), "normalization.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")

    norm = np.load(norm_path)
    return {
        "path": norm_path,
        "y_mean": norm["y_mean"].astype(np.float32),
        "y_std": norm["y_std"].astype(np.float32),
        "U_mean": norm["U_mean"].astype(np.float32),
        "U_std": norm["U_std"].astype(np.float32),
    }


def coverage_rate(abs_error, std, k):
    return float(np.mean(abs_error <= k * std))


def gaussian_nll_numpy(target, mu, std, eps=1e-8):
    var = std ** 2 + eps
    return float(np.mean(0.5 * (np.log(var) + (target - mu) ** 2 / var)))


def rms_calibration_scale(abs_error, std, eps=1e-12):
    """
    Post-hoc one-number uncertainty calibration.
    alpha > 1 means the predicted uncertainty is under-estimated.
    alpha < 1 means the predicted uncertainty is over-estimated.
    """
    err2 = np.mean(np.asarray(abs_error, dtype=np.float64) ** 2)
    std2 = np.mean(np.asarray(std, dtype=np.float64) ** 2) + eps
    return float(np.sqrt(err2 / std2))


def confidence_filtering_metrics(vertex_error, vertex_std, keep_rates):
    e = vertex_error.reshape(-1)
    s = vertex_std.reshape(-1)

    order = np.argsort(s)
    total = len(order)
    out = {}

    for rate in keep_rates:
        n_keep = max(1, int(round(total * rate)))
        idx = order[:n_keep]
        key = int(round(rate * 100))
        out[f"keep_{key:02d}_mean_error"] = float(np.mean(e[idx]))
        out[f"keep_{key:02d}_median_error"] = float(np.median(e[idx]))
        out[f"keep_{key:02d}_mean_std"] = float(np.mean(s[idx]))

    return out


# ============================================================
# Model definition: must match train.py
# ============================================================

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
        self.hidden_dim = hidden_dim
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

        decoder_in = y_dim + z_dim

        self.decoder_backbone = nn.Sequential(
            nn.Linear(decoder_in, hidden_dim),
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
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y_norm, sample_z=True):
        h = self.encoder(y_norm)

        z_mu = self.z_mu_head(h)
        z_logvar = self.z_logvar_head(h).clamp(min=-12.0, max=8.0)

        z = self.reparameterize(z_mu, z_logvar, sample_z=sample_z)

        dec_in = torch.cat([y_norm, z], dim=-1)
        hd = self.decoder_backbone(dec_in)

        U_mu = self.U_mu_head(hd)
        raw_logvar = self.U_logvar_head(hd)
        U_logvar = torch.clamp(raw_logvar, min=self.logvar_min, max=self.logvar_max)

        W_logits = self.W_head(hd).view(-1, self.K_obs, self.N_vertices)

        return U_mu, U_logvar, W_logits, z_mu, z_logvar


# ============================================================
# Observation operator
# ============================================================

def torch_perspective_project(X_world, Kmat, R, t):
    """
    X_world: [B,K,3]
    Kmat:    [B,3,3]
    R:       [B,3,3]
    t:       [B,3]
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


def observation_operator_softW(U_phys, W_logits, X0, Kmat, R, t, normalize_y=True):
    """
    U_phys:   [B,N,3]
    W_logits: [B,K,N]
    X0:       [B,N,3]

    Returns:
        y_hat: [B,K,2]
        W:     [B,K,N]
    """
    W = F.softmax(W_logits, dim=-1)

    X_surface = torch.einsum("bkn,bnd->bkd", W, X0)
    U_surface = torch.einsum("bkn,bnd->bkd", W, U_phys)

    uv0, _ = torch_perspective_project(X_surface, Kmat, R, t)
    uvt, _ = torch_perspective_project(X_surface + U_surface, Kmat, R, t)

    y_pixel_hat = uvt - uv0

    if normalize_y:
        fx = Kmat[:, 0, 0].view(-1, 1, 1)
        y_hat = y_pixel_hat / fx
    else:
        y_hat = y_pixel_hat

    return y_hat, W


def sparse_W_metrics(W_logits_np, W_idx, W_val):
    """
    W_logits_np: [T,K,N]
    W_idx:       [K,3]
    W_val:       [K,3]
    """
    T, K, N = W_logits_np.shape

    # Numerically stable log-softmax in numpy
    x = W_logits_np - np.max(W_logits_np, axis=-1, keepdims=True)
    logp = x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True) + 1e-12)

    ce_list = []
    top1_list = []
    gt_top_local = np.argmax(W_val, axis=-1)           # [K]
    gt_top = W_idx[np.arange(K), gt_top_local]         # [K]

    pred_top = np.argmax(W_logits_np, axis=-1)         # [T,K]

    for t in range(T):
        gathered = logp[t, np.arange(K)[:, None], W_idx]  # [K,3]
        ce = -np.sum(W_val * gathered, axis=-1)           # [K]
        ce_list.append(np.mean(ce))
        top1_list.append(np.mean(pred_top[t] == gt_top))

    return {
        "W_ce": float(np.mean(ce_list)),
        "W_top1_acc": float(np.mean(top1_list)),
    }


# ============================================================
# Loading model
# ============================================================

def load_model(weight_path, y_dim, U_dim, K_obs, N_vertices, device):
    ckpt = torch.load(weight_path, map_location=device)

    model = JointUWProbabilisticMLP(
        y_dim=y_dim,
        U_dim=U_dim,
        K_obs=K_obs,
        N_vertices=N_vertices,
        z_dim=ckpt.get("z_dim", 32),
        hidden_dim=ckpt.get("hidden_dim", 256),
        logvar_min=ckpt.get("logvar_min", -8.0),
        logvar_max=ckpt.get("logvar_max", 2.0),
        dropout=0.0,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "epoch": ckpt.get("epoch", None),
        "best_metric": ckpt.get("best_metric", None),
        "model_type": ckpt.get("model_type", "JointUWProbabilisticMLP"),
        "z_dim": ckpt.get("z_dim", 32),
        "hidden_dim": ckpt.get("hidden_dim", 256),
        "logvar_min": ckpt.get("logvar_min", -8.0),
        "logvar_max": ckpt.get("logvar_max", 2.0),
        "lambda_obs": ckpt.get("lambda_obs", None),
        "lambda_U": ckpt.get("lambda_U", None),
        "lambda_W": ckpt.get("lambda_W", None),
        "beta_kl": ckpt.get("beta_kl", None),
    }

    return model, meta


# ============================================================
# Prediction
# ============================================================

def predict_sequence_real_like(
    model,
    y,
    X0,
    camera_K,
    camera_R,
    camera_t,
    norm,
    device,
    batch_size=64,
    mc_samples=8,
    normalize_y_for_projection=True,
    save_W=True,
    return_samples=False,
):
    """
    Real-like inference:
        input to model is y only.

    Geometry/camera are used only by the observation operator after prediction.

    Returns:
        U_pred_seq: [T,N,3]
        U_std_seq:  [T,N,3]
        y_hat_seq:  [T,K,2]
        W_seq:      [T,K,N] if save_W else None
        W_entropy:  [T,K]
    """
    T, K_obs, _ = y.shape
    N = X0.shape[0]
    U_dim = 3 * N

    y_flat = y.reshape(T, -1).astype(np.float32)

    y_mean = norm["y_mean"]
    y_std = norm["y_std"]
    U_mean = norm["U_mean"]
    U_std = norm["U_std"]

    if y_flat.shape[1] != y_mean.shape[1]:
        raise ValueError(
            f"y_dim mismatch. y has {y_flat.shape[1]}, normalization has {y_mean.shape[1]}."
        )

    y_norm = (y_flat - y_mean) / y_std

    X0_b = torch.from_numpy(X0.astype(np.float32)).to(device)
    K_b = torch.from_numpy(camera_K.astype(np.float32)).to(device)
    R_b = torch.from_numpy(camera_R.astype(np.float32)).to(device)
    t_b = torch.from_numpy(camera_t.astype(np.float32)).to(device)

    U_mean_t = torch.from_numpy(U_mean.astype(np.float32)).to(device)
    U_std_t = torch.from_numpy(U_std.astype(np.float32)).to(device)

    all_mu_samples = []
    all_var_samples = []
    all_yhat_samples = []
    all_W_samples = []
    all_W_entropy_samples = []

    with torch.no_grad():
        for s in range(mc_samples):
            mu_batches = []
            var_batches = []
            yhat_batches = []
            W_batches = []
            Went_batches = []

            for i in range(0, T, batch_size):
                y_batch_np = y_norm[i:i + batch_size]
                B = y_batch_np.shape[0]

                y_batch = torch.from_numpy(y_batch_np.astype(np.float32)).to(device)

                U_mu_norm, U_logvar_norm, W_logits, _, _ = model(y_batch, sample_z=True)

                U_mu_phys_flat = U_mu_norm * U_std_t + U_mean_t
                U_var_phys_flat = torch.exp(U_logvar_norm) * (U_std_t ** 2)

                U_mu_phys = U_mu_phys_flat.view(B, N, 3)

                X0_batch = X0_b.unsqueeze(0).expand(B, -1, -1)
                K_batch = K_b.unsqueeze(0).expand(B, -1, -1)
                R_batch = R_b.unsqueeze(0).expand(B, -1, -1)
                t_batch = t_b.unsqueeze(0).expand(B, -1)

                y_hat, W = observation_operator_softW(
                    U_phys=U_mu_phys,
                    W_logits=W_logits,
                    X0=X0_batch,
                    Kmat=K_batch,
                    R=R_batch,
                    t=t_batch,
                    normalize_y=normalize_y_for_projection,
                )

                W_clamped = torch.clamp(W, min=1e-12)
                W_entropy = -torch.sum(W_clamped * torch.log(W_clamped), dim=-1)

                mu_batches.append(U_mu_phys_flat.cpu().numpy().astype(np.float32))
                var_batches.append(U_var_phys_flat.cpu().numpy().astype(np.float32))
                yhat_batches.append(y_hat.cpu().numpy().astype(np.float32))
                Went_batches.append(W_entropy.cpu().numpy().astype(np.float32))

                if save_W:
                    W_batches.append(W.cpu().numpy().astype(np.float32))

            all_mu_samples.append(np.concatenate(mu_batches, axis=0))
            all_var_samples.append(np.concatenate(var_batches, axis=0))
            all_yhat_samples.append(np.concatenate(yhat_batches, axis=0))
            all_W_entropy_samples.append(np.concatenate(Went_batches, axis=0))

            if save_W:
                all_W_samples.append(np.concatenate(W_batches, axis=0))

    mu_samples = np.stack(all_mu_samples, axis=0)       # [S,T,U_dim]
    var_samples = np.stack(all_var_samples, axis=0)     # [S,T,U_dim]

    mean_flat = np.mean(mu_samples, axis=0)
    total_var_flat = np.mean(var_samples, axis=0) + np.var(mu_samples, axis=0)
    std_flat = np.sqrt(np.maximum(total_var_flat, 1e-12))

    U_pred_seq = mean_flat.reshape(T, N, 3).astype(np.float32)
    U_std_seq = std_flat.reshape(T, N, 3).astype(np.float32)

    y_hat_seq = np.mean(np.stack(all_yhat_samples, axis=0), axis=0).astype(np.float32)
    W_entropy_seq = np.mean(np.stack(all_W_entropy_samples, axis=0), axis=0).astype(np.float32)

    if save_W:
        W_seq = np.mean(np.stack(all_W_samples, axis=0), axis=0).astype(np.float32)
    else:
        W_seq = None

    # Posterior variance decomposition:
    # total variance = E_z[var(U|y,z)] + var_z(E[U|y,z])
    posterior_extra = None
    if return_samples:
        aleatoric_var_flat = np.mean(var_samples, axis=0)
        epistemic_var_flat = np.var(mu_samples, axis=0)
        posterior_extra = {
            "U_mu_samples": mu_samples.reshape(mc_samples, T, N, 3).astype(np.float32),
            "U_aleatoric_std_seq": np.sqrt(np.maximum(aleatoric_var_flat, 1e-12)).reshape(T, N, 3).astype(np.float32),
            "U_epistemic_std_seq": np.sqrt(np.maximum(epistemic_var_flat, 1e-12)).reshape(T, N, 3).astype(np.float32),
        }

    return U_pred_seq, U_std_seq, y_hat_seq, W_seq, W_entropy_seq, posterior_extra


# ============================================================
# Evaluate one sample
# ============================================================

def evaluate_one_sample(sample_dir, model, norm, device, args):
    sample_name = sample_name_from_dir(sample_dir)

    y = np.load(os.path.join(sample_dir, "y.npy")).astype(np.float32)
    X0 = np.load(os.path.join(sample_dir, "X0.npy")).astype(np.float32)
    camera_K = np.load(os.path.join(sample_dir, "camera_K.npy")).astype(np.float32)
    camera_R = np.load(os.path.join(sample_dir, "camera_R.npy")).astype(np.float32)
    camera_t = np.load(os.path.join(sample_dir, "camera_t.npy")).astype(np.float32)

    if args.max_frames is not None:
        y = y[:args.max_frames]

    T, K_obs, _ = y.shape
    N = X0.shape[0]

    U_pred_seq, U_std_seq, y_hat_seq, W_seq, W_entropy_seq, posterior_extra = predict_sequence_real_like(
        model=model,
        y=y,
        X0=X0,
        camera_K=camera_K,
        camera_R=camera_R,
        camera_t=camera_t,
        norm=norm,
        device=device,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        normalize_y_for_projection=not args.no_normalize_y_for_projection,
        save_W=args.save_W,
        return_samples=args.save_samples,
    )

    metrics = {
        "sample": sample_name,
        "T": T,
        "N": N,
        "K": K_obs,
    }

    # Observation consistency is always available.
    y_error = y_hat_seq - y
    metrics["obs_mse"] = float(np.mean(y_error ** 2))
    metrics["obs_rmse"] = float(np.sqrt(metrics["obs_mse"]))
    metrics["obs_mae"] = float(np.mean(np.abs(y_error)))
    metrics["obs_corr"] = safe_corr(y_hat_seq, y)
    metrics["W_entropy_mean"] = float(np.mean(W_entropy_seq))
    metrics["W_entropy_median"] = float(np.median(W_entropy_seq))
    metrics["W_entropy_max"] = float(np.max(W_entropy_seq))

    # U metrics only if GT exists.
    U_path = os.path.join(sample_dir, "U_seq.npy")
    has_U_gt = os.path.exists(U_path)
    if has_U_gt:
        U_seq = np.load(U_path).astype(np.float32)
        U_seq = U_seq[:T]

        error_seq = U_pred_seq - U_seq
        abs_error_seq = np.abs(error_seq)

        mse = float(np.mean(error_seq ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_error_seq))
        rel_l2 = float(np.linalg.norm(error_seq.reshape(-1)) / (np.linalg.norm(U_seq.reshape(-1)) + 1e-12))

        vertex_error = np.linalg.norm(error_seq, axis=2)
        vertex_std = np.linalg.norm(U_std_seq, axis=2)

        axis_rmse = np.sqrt(np.mean(error_seq ** 2, axis=(0, 1)))
        axis_mae = np.mean(abs_error_seq, axis=(0, 1))

        disp_axis = args.disp_axis
        amp_per_vertex = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
        curve_vertex_idx = int(np.argmax(amp_per_vertex))

        gt_curve = U_seq[:, curve_vertex_idx, disp_axis]
        pred_curve = U_pred_seq[:, curve_vertex_idx, disp_axis]
        std_curve = U_std_seq[:, curve_vertex_idx, disp_axis]
        curve_err = pred_curve - gt_curve

        metrics.update({
            "rmse": rmse,
            "mae": mae,
            "relative_l2": rel_l2,
            "axis_rmse_x": float(axis_rmse[0]),
            "axis_rmse_y": float(axis_rmse[1]),
            "axis_rmse_z": float(axis_rmse[2]),
            "axis_mae_x": float(axis_mae[0]),
            "axis_mae_y": float(axis_mae[1]),
            "axis_mae_z": float(axis_mae[2]),
            "gt_max_disp": float(np.max(np.linalg.norm(U_seq, axis=2))),
            "pred_max_disp": float(np.max(np.linalg.norm(U_pred_seq, axis=2))),
            "vertex_mean_error": float(np.mean(vertex_error)),
            "vertex_median_error": float(np.median(vertex_error)),
            "vertex_max_error": float(np.max(vertex_error)),
            "mean_vertex_std": float(np.mean(vertex_std)),
            "median_vertex_std": float(np.median(vertex_std)),
            "max_vertex_std": float(np.max(vertex_std)),
            "coverage_1std_component": coverage_rate(abs_error_seq, U_std_seq, 1.0),
            "coverage_2std_component": coverage_rate(abs_error_seq, U_std_seq, 2.0),
            "coverage_3std_component": coverage_rate(abs_error_seq, U_std_seq, 3.0),
            "coverage_1std_vertex": coverage_rate(vertex_error, vertex_std, 1.0),
            "coverage_2std_vertex": coverage_rate(vertex_error, vertex_std, 2.0),
            "coverage_3std_vertex": coverage_rate(vertex_error, vertex_std, 3.0),
            "calibration_scale_component": rms_calibration_scale(abs_error_seq, U_std_seq),
            "calibration_scale_vertex": rms_calibration_scale(vertex_error, vertex_std),
            "error_std_corr_vertex": safe_corr(vertex_error, vertex_std),
            "error_std_corr_component": safe_corr(abs_error_seq, U_std_seq),
            "nll_physical": gaussian_nll_numpy(U_seq, U_pred_seq, U_std_seq),
            "curve_vertex_idx": curve_vertex_idx,
            "curve_rmse": float(np.sqrt(np.mean(curve_err ** 2))),
            "curve_corr": safe_corr(gt_curve, pred_curve),
            "curve_gt_max_abs": float(np.max(np.abs(gt_curve))),
            "curve_pred_max_abs": float(np.max(np.abs(pred_curve))),
            "curve_mean_std": float(np.mean(np.abs(std_curve))),
        })

        # Calibrated coverage. This only reports calibration quality; it does not change prediction.
        alpha_c = metrics["calibration_scale_component"]
        alpha_v = metrics["calibration_scale_vertex"]
        metrics["coverage_1std_component_cal"] = coverage_rate(abs_error_seq, alpha_c * U_std_seq, 1.0)
        metrics["coverage_2std_component_cal"] = coverage_rate(abs_error_seq, alpha_c * U_std_seq, 2.0)
        metrics["coverage_3std_component_cal"] = coverage_rate(abs_error_seq, alpha_c * U_std_seq, 3.0)
        metrics["coverage_1std_vertex_cal"] = coverage_rate(vertex_error, alpha_v * vertex_std, 1.0)
        metrics["coverage_2std_vertex_cal"] = coverage_rate(vertex_error, alpha_v * vertex_std, 2.0)
        metrics["coverage_3std_vertex_cal"] = coverage_rate(vertex_error, alpha_v * vertex_std, 3.0)

        if posterior_extra is not None:
            epi = np.linalg.norm(posterior_extra["U_epistemic_std_seq"], axis=2)
            ale = np.linalg.norm(posterior_extra["U_aleatoric_std_seq"], axis=2)
            metrics["mean_epistemic_std"] = float(np.mean(epi))
            metrics["mean_aleatoric_std"] = float(np.mean(ale))
            metrics["error_epistemic_corr_vertex"] = safe_corr(vertex_error, epi)
            metrics["error_aleatoric_corr_vertex"] = safe_corr(vertex_error, ale)

        metrics.update(confidence_filtering_metrics(vertex_error, vertex_std, args.keep_rates))
    else:
        error_seq = None
        vertex_error = None
        U_seq = None

    # W metrics only if synthetic W_gt exists.
    W_idx_path = os.path.join(sample_dir, "W_gt_sparse_indices.npy")
    W_val_path = os.path.join(sample_dir, "W_gt_sparse_values.npy")
    if args.save_W and W_seq is not None and os.path.exists(W_idx_path) and os.path.exists(W_val_path):
        W_idx = np.load(W_idx_path).astype(np.int64)
        W_val = np.load(W_val_path).astype(np.float32)

        # Convert W_seq to logits-like by log(W) for CE metric.
        # This is equivalent to evaluating soft CE on predicted probabilities.
        W_logits_np = np.log(np.clip(W_seq, 1e-12, 1.0)).astype(np.float32)
        metrics.update(sparse_W_metrics(W_logits_np, W_idx, W_val))

        gt_top_local = np.argmax(W_val, axis=-1)
        gt_top = W_idx[np.arange(K_obs), gt_top_local]
        pred_top = np.argmax(W_seq, axis=-1)  # [T,K]
        metrics["W_top1_acc_from_prob"] = float(np.mean(pred_top == gt_top[None, :]))

    # Save predictions.
    if args.save_predictions:
        out_dir = os.path.join(args.pred_dir, sample_name)
        ensure_dir(out_dir)

        np.save(os.path.join(out_dir, "U_pred_seq.npy"), U_pred_seq.astype(np.float32))
        np.save(os.path.join(out_dir, "U_std_seq.npy"), U_std_seq.astype(np.float32))
        np.save(os.path.join(out_dir, "y_hat_seq.npy"), y_hat_seq.astype(np.float32))
        np.save(os.path.join(out_dir, "y_input.npy"), y.astype(np.float32))
        np.save(os.path.join(out_dir, "W_entropy_seq.npy"), W_entropy_seq.astype(np.float32))

        if posterior_extra is not None:
            np.save(os.path.join(out_dir, "U_mu_samples.npy"), posterior_extra["U_mu_samples"].astype(np.float32))
            np.save(os.path.join(out_dir, "U_epistemic_std_seq.npy"), posterior_extra["U_epistemic_std_seq"].astype(np.float32))
            np.save(os.path.join(out_dir, "U_aleatoric_std_seq.npy"), posterior_extra["U_aleatoric_std_seq"].astype(np.float32))

        if W_seq is not None:
            np.save(os.path.join(out_dir, "W_pred_seq.npy"), W_seq.astype(np.float32))
            np.save(os.path.join(out_dir, "W_pred_mean.npy"), np.mean(W_seq, axis=0).astype(np.float32))
            np.save(os.path.join(out_dir, "W_pred_top_idx_seq.npy"), np.argmax(W_seq, axis=-1).astype(np.int64))

        if has_U_gt:
            np.save(os.path.join(out_dir, "error_seq.npy"), error_seq.astype(np.float32))
            np.save(os.path.join(out_dir, "vertex_error_seq.npy"), vertex_error.astype(np.float32))
            np.save(os.path.join(out_dir, "curve_gt.npy"), gt_curve.astype(np.float32))
            np.save(os.path.join(out_dir, "curve_pred.npy"), pred_curve.astype(np.float32))
            np.save(os.path.join(out_dir, "curve_std.npy"), std_curve.astype(np.float32))
            np.save(os.path.join(out_dir, "curve_vertex_idx.npy"), np.array([curve_vertex_idx], dtype=np.int64))

    return metrics


# ============================================================
# CSV helpers
# ============================================================

def save_csv(path, rows):
    if len(rows) == 0:
        return
    ensure_dir(os.path.dirname(path))

    # union of keys
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows):
    summary = {}
    keys = set()
    for r in rows:
        keys.update(r.keys())

    for k in sorted(keys):
        if k == "sample":
            continue
        vals = []
        for r in rows:
            v = r.get(k, None)
            if isinstance(v, (int, float, np.integer, np.floating)):
                if not np.isnan(v):
                    vals.append(float(v))
        if len(vals) > 0:
            summary[k + "_mean"] = float(np.mean(vals))
            summary[k + "_std"] = float(np.std(vals))

    return summary


def save_summary_csv(path, summary):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])


# ============================================================
# Main
# ============================================================

def evaluate(args):
    ensure_dir(args.report_dir)
    if args.save_predictions:
        ensure_dir(args.pred_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Using device:", device)
    print("data_dir:", os.path.abspath(args.data_dir))
    print("weight_path:", os.path.abspath(args.weight_path))

    sample_dirs = find_sample_dirs(args.data_dir)

    if args.sample_name is not None:
        sample_dirs = [d for d in sample_dirs if sample_name_from_dir(d) == args.sample_name]
        if len(sample_dirs) == 0:
            raise ValueError(f"sample_name={args.sample_name} not found under {args.data_dir}")

    if not args.eval_all and len(sample_dirs) > 1:
        print("WARNING: multiple samples found but --eval_all not set.")
        print("Evaluating only first sample. Use --eval_all for formal testing.")
        sample_dirs = sample_dirs[:1]

    print("Number of samples:", len(sample_dirs))

    norm = load_normalization(args.weight_path)
    print("normalization:", norm["path"])

    # Determine dimensions from first sample.
    y0 = np.load(os.path.join(sample_dirs[0], "y.npy")).astype(np.float32)
    X0 = np.load(os.path.join(sample_dirs[0], "X0.npy")).astype(np.float32)

    y_dim = y0.reshape(y0.shape[0], -1).shape[1]
    K_obs = y0.shape[1]
    N_vertices = X0.shape[0]
    U_dim = 3 * N_vertices

    model, meta = load_model(
        args.weight_path,
        y_dim=y_dim,
        U_dim=U_dim,
        K_obs=K_obs,
        N_vertices=N_vertices,
        device=device,
    )

    print("model meta:", meta)
    print("mc_samples:", args.mc_samples)
    print("real-like input: y only")
    print("save_predictions:", args.save_predictions)
    print("save_W:", args.save_W)
    print("save_samples:", args.save_samples)

    rows = []
    for i, sample_dir in enumerate(sample_dirs):
        metrics = evaluate_one_sample(sample_dir, model, norm, device, args)
        rows.append(metrics)

        msg = (
            f"[{i+1:03d}/{len(sample_dirs):03d}] {metrics['sample']} | "
            f"ObsRMSE={metrics.get('obs_rmse', np.nan):.6e} | "
            f"ObsCorr={metrics.get('obs_corr', np.nan):.6f} | "
            f"Went={metrics.get('W_entropy_mean', np.nan):.3f}"
        )

        if "relative_l2" in metrics:
            msg += (
                f" | RMSE={metrics['rmse']:.6e}"
                f" | RelL2={metrics['relative_l2']:.6e}"
                f" | CurveCorr={metrics['curve_corr']:.6f}"
                f" | ErrStdCorr={metrics['error_std_corr_vertex']:.6f}"
            )

        if "W_top1_acc" in metrics:
            msg += f" | Wacc={metrics['W_top1_acc']:.4f}"

        print(msg)

    per_sample_path = os.path.join(args.report_dir, "eval_joint_UW_per_sample_metrics.csv")
    save_csv(per_sample_path, rows)

    summary = summarize_metrics(rows)
    summary_path = os.path.join(args.report_dir, "eval_joint_UW_summary_metrics.csv")
    save_summary_csv(summary_path, summary)

    print("\n========== Evaluation Summary ==========")
    print("num_samples:", len(rows))

    for key in [
        "obs_rmse",
        "obs_corr",
        "rmse",
        "mae",
        "relative_l2",
        "curve_corr",
        "error_std_corr_vertex",
        "coverage_1std_component",
        "coverage_2std_component",
        "coverage_2std_component_cal",
        "calibration_scale_component",
        "calibration_scale_vertex",
        "mean_vertex_std",
        "mean_epistemic_std",
        "mean_aleatoric_std",
        "W_ce",
        "W_top1_acc",
        "W_entropy_mean",
    ]:
        m = key + "_mean"
        s = key + "_std"
        if m in summary:
            print(f"{key:28s}: {summary[m]:.6e} ± {summary.get(s, 0.0):.6e}")

    print("\nSaved per-sample metrics to:", per_sample_path)
    print("Saved summary metrics to:", summary_path)

    if args.save_predictions:
        print("Saved predictions to:", os.path.abspath(args.pred_dir))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--weight_path", type=str, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)

    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--save_predictions", action="store_true")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument("--disp_axis", type=int, default=1)
    parser.add_argument("--keep_rates", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])

    parser.add_argument("--save_W", action="store_true", help="Save W_pred_seq.npy. Can be large but ok for this dataset.")
    parser.add_argument("--save_samples", action="store_true", help="Save posterior U samples and epistemic/aleatoric std maps.")
    parser.add_argument("--no_normalize_y_for_projection", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
