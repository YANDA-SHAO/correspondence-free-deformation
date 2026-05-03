# src/visualize.py
"""
Final visualization script for joint deformation + soft-correspondence model.

Designed to work with predictions saved by final eval.py:

    python src\eval.py --data_dir data\dataset_v2_surface\test --eval_all --mc_samples 32 --save_predictions --save_samples --save_W

This script visualizes:
1. GT vs Pred displacement vibration curve
2. Multiple confident vertex curves
3. 3D mesh displacement / uncertainty / error
4. Observation-space consistency: y input vs y_hat
5. W correspondence heat / top predicted vertices
6. Error vs uncertainty scatter
7. Confidence filtering curve
8. Dataset-level summary plots

Real-like principle:
    The model input is y only.
    W_gt and U_gt are only used for evaluation/visualization if available.
"""

import os
import glob
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_v2_surface", "test")
DEFAULT_PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
DEFAULT_FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sample_name_from_dir(sample_dir):
    return os.path.basename(os.path.normpath(sample_dir))


def find_sample_dirs(data_dir):
    data_dir = os.path.abspath(data_dir)

    if os.path.exists(os.path.join(data_dir, "y.npy")):
        return [data_dir]

    sample_dirs = sorted(glob.glob(os.path.join(data_dir, "sample_*")))
    sample_dirs = [d for d in sample_dirs if os.path.isdir(d)]

    if len(sample_dirs) == 0:
        raise ValueError(f"No sample found in {data_dir}")

    return sample_dirs


def resolve_pred_dir(pred_root, sample_dir):
    sample_name = sample_name_from_dir(sample_dir)
    candidate = os.path.join(pred_root, sample_name)

    if os.path.exists(os.path.join(candidate, "U_pred_seq.npy")):
        return candidate

    if os.path.exists(os.path.join(pred_root, "U_pred_seq.npy")):
        return pred_root

    raise FileNotFoundError(
        f"Cannot find predictions for {sample_name}.\n"
        f"Expected {candidate}\\U_pred_seq.npy\n"
        f"Run eval.py with --save_predictions first."
    )


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def maybe_load(path, dtype=None):
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def load_sample_and_prediction(sample_dir, pred_root):
    pred_dir = resolve_pred_dir(pred_root, sample_dir)

    data = {}
    data["sample_name"] = sample_name_from_dir(sample_dir)
    data["sample_dir"] = sample_dir
    data["pred_dir"] = pred_dir

    # Required sample files
    data["X0"] = np.load(os.path.join(sample_dir, "X0.npy")).astype(np.float32)
    data["y"] = np.load(os.path.join(sample_dir, "y.npy")).astype(np.float32)
    data["camera_K"] = np.load(os.path.join(sample_dir, "camera_K.npy")).astype(np.float32)
    data["camera_R"] = np.load(os.path.join(sample_dir, "camera_R.npy")).astype(np.float32)
    data["camera_t"] = np.load(os.path.join(sample_dir, "camera_t.npy")).astype(np.float32)

    # Optional synthetic GT files
    data["U_seq"] = maybe_load(os.path.join(sample_dir, "U_seq.npy"), np.float32)
    data["faces"] = maybe_load(os.path.join(sample_dir, "faces.npy"), np.int64)
    data["obs_face_vertices"] = maybe_load(os.path.join(sample_dir, "obs_face_vertices.npy"), np.int64)
    data["obs_barycentric"] = maybe_load(os.path.join(sample_dir, "obs_barycentric.npy"), np.float32)
    data["W_idx"] = maybe_load(os.path.join(sample_dir, "W_gt_sparse_indices.npy"), np.int64)
    data["W_val"] = maybe_load(os.path.join(sample_dir, "W_gt_sparse_values.npy"), np.float32)
    data["track_uv0"] = maybe_load(os.path.join(sample_dir, "track_uv0.npy"), np.float32)
    data["track_uv_seq"] = maybe_load(os.path.join(sample_dir, "track_uv_seq.npy"), np.float32)

    # Prediction files
    data["U_pred_seq"] = np.load(os.path.join(pred_dir, "U_pred_seq.npy")).astype(np.float32)
    data["U_std_seq"] = np.load(os.path.join(pred_dir, "U_std_seq.npy")).astype(np.float32)
    data["y_hat_seq"] = maybe_load(os.path.join(pred_dir, "y_hat_seq.npy"), np.float32)
    data["W_pred_seq"] = maybe_load(os.path.join(pred_dir, "W_pred_seq.npy"), np.float32)
    data["W_pred_mean"] = maybe_load(os.path.join(pred_dir, "W_pred_mean.npy"), np.float32)
    data["W_entropy_seq"] = maybe_load(os.path.join(pred_dir, "W_entropy_seq.npy"), np.float32)
    data["error_seq"] = maybe_load(os.path.join(pred_dir, "error_seq.npy"), np.float32)
    data["vertex_error_seq"] = maybe_load(os.path.join(pred_dir, "vertex_error_seq.npy"), np.float32)

    return data


def vertex_std_score(U_std_seq, mode="mean"):
    std_mag = np.linalg.norm(U_std_seq, axis=2)
    if mode == "mean":
        return np.mean(std_mag, axis=0)
    if mode == "max":
        return np.max(std_mag, axis=0)
    raise ValueError(mode)


def get_topk_mask(U_std_seq, ratio=1.0):
    N = U_std_seq.shape[1]
    if ratio >= 1.0:
        return np.ones(N, dtype=bool)

    k = max(1, int(round(ratio * N)))
    score = vertex_std_score(U_std_seq, mode="mean")
    order = np.argsort(score)
    mask = np.zeros(N, dtype=bool)
    mask[order[:k]] = True
    return mask


def suffix_top(top_ratio):
    if top_ratio >= 1.0:
        return ""
    return f"_top{int(round(top_ratio * 100)):02d}pct"
def _reshape_U_samples(U_samples, T, N):
    """
    Accepts posterior samples saved as either [S,T,N,3] or [S,T,3N].
    Returns [S,T,N,3].
    """
    if U_samples is None:
        return None
    if U_samples.ndim == 4:
        return U_samples
    if U_samples.ndim == 3:
        S, T0, D = U_samples.shape
        if T0 != T:
            raise ValueError(f"U_mu_samples T mismatch: samples T={T0}, expected T={T}")
        if D != 3 * N:
            raise ValueError(f"U_mu_samples last dim mismatch: got {D}, expected {3*N}")
        return U_samples.reshape(S, T, N, 3)
    raise ValueError(f"Unsupported U_mu_samples shape: {U_samples.shape}")


def plot_posterior_samples(data, fig_dir, disp_axis=1, num_samples=5):
    """Visualize posterior deformation samples for one representative vertex."""
    U_samples = data.get("U_mu_samples", None)
    U_mean = data["U_pred_seq"]
    U_std = data["U_std_seq"]

    if U_samples is None:
        print("Skip posterior samples: U_mu_samples.npy not found.")
        print("Expected file:", os.path.join(data["pred_dir"], "U_mu_samples.npy"))
        print("Run eval.py with: --save_predictions --save_samples")
        return

    sample_name = data["sample_name"]
    T, N, _ = U_mean.shape
    U_samples = _reshape_U_samples(U_samples, T=T, N=N)

    U_seq = data["U_seq"]
    if U_seq is not None:
        amp = np.max(np.abs(U_seq[:T, :, disp_axis]), axis=0)
    else:
        amp = np.max(np.abs(U_mean[:, :, disp_axis]), axis=0)

    idx = int(np.argmax(amp))
    n_show = min(int(num_samples), U_samples.shape[0])

    plt.figure(figsize=(10, 4))

    for s in range(n_show):
        label = "posterior samples" if s == 0 else None
        plt.plot(U_samples[s, :, idx, disp_axis], alpha=0.35, linewidth=1.0, label=label)

    plt.plot(U_mean[:, idx, disp_axis], "k--", linewidth=2.0, label="posterior mean")

    if U_seq is not None:
        plt.plot(U_seq[:T, idx, disp_axis], linewidth=2.0, label="GT")

    std = np.abs(U_std[:, idx, disp_axis])
    x = np.arange(T)
    plt.fill_between(
        x,
        U_mean[:, idx, disp_axis] - 2 * std,
        U_mean[:, idx, disp_axis] + 2 * std,
        alpha=0.20,
        label="mean ±2 std",
    )

    plt.xlabel("Frame")
    plt.ylabel(f"Displacement axis {disp_axis}")
    plt.title(f"{sample_name}: posterior samples | vertex {idx} | S={U_samples.shape[0]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, "posterior_samples.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def label_top(top_ratio):
    if top_ratio >= 1.0:
        return "all vertices"
    return f"top {int(round(top_ratio * 100))}% lowest-std vertices"


# ============================================================
# Plot 1: single vibration curve
# ============================================================

def plot_vibration_curve(data, fig_dir, disp_axis=1, top_ratio=1.0):
    sample_name = data["sample_name"]
    U_seq = data["U_seq"]
    U_pred_seq = data["U_pred_seq"]
    U_std_seq = data["U_std_seq"]

    if U_seq is None:
        print("Skip vibration_curve: U_seq.npy not available.")
        return

    mask = get_topk_mask(U_std_seq, top_ratio)
    valid = np.where(mask)[0]

    amp = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
    idx = int(valid[np.argmax(amp[valid])])

    gt = U_seq[:, idx, disp_axis]
    pred = U_pred_seq[:, idx, disp_axis]
    std = np.abs(U_std_seq[:, idx, disp_axis])

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    corr = safe_corr(gt, pred)

    plt.figure(figsize=(10, 4))
    plt.plot(gt, label="GT")
    plt.plot(pred, "--", label="Pred mean")

    if np.max(std) > 0:
        x = np.arange(len(gt))
        plt.fill_between(x, pred - 2 * std, pred + 2 * std, alpha=0.25, label="Pred ±2 std")

    plt.xlabel("Frame")
    plt.ylabel(f"Displacement axis {disp_axis}")
    plt.title(f"{sample_name}: vibration curve | {label_top(top_ratio)} | vertex {idx} | RMSE={rmse:.2e}, Corr={corr:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, f"vibration_curve{suffix_top(top_ratio)}.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Plot 2: multiple confident vertex curves
# ============================================================

def plot_multiple_vertex_curves(data, fig_dir, disp_axis=1, top_ratio=0.05, num_vertices=8):
    sample_name = data["sample_name"]
    U_seq = data["U_seq"]
    U_pred_seq = data["U_pred_seq"]
    U_std_seq = data["U_std_seq"]

    if U_seq is None:
        print("Skip multiple curves: U_seq.npy not available.")
        return

    mask = get_topk_mask(U_std_seq, top_ratio)
    valid = np.where(mask)[0]

    amp = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
    valid_sorted = valid[np.argsort(amp[valid])]

    if len(valid_sorted) <= num_vertices:
        chosen = valid_sorted
    else:
        pos = np.linspace(0, len(valid_sorted) - 1, num_vertices).astype(int)
        chosen = valid_sorted[pos]

    print(f"\n========== Multiple Vertex Curves: {sample_name} ({label_top(top_ratio)}) ==========")
    print(f"{'vertex':>8} | {'GT max':>12} | {'Pred max':>12} | {'RMSE':>12} | {'Corr':>10}")
    print("-" * 68)

    plt.figure(figsize=(12, max(4, 2.0 * len(chosen))))

    for i, idx in enumerate(chosen):
        gt = U_seq[:, idx, disp_axis]
        pred = U_pred_seq[:, idx, disp_axis]
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        corr = safe_corr(gt, pred)

        print(f"{idx:8d} | {np.max(np.abs(gt)):12.6e} | {np.max(np.abs(pred)):12.6e} | {rmse:12.6e} | {corr:10.6f}")

        plt.subplot(len(chosen), 1, i + 1)
        plt.plot(gt, label="GT")
        plt.plot(pred, "--", label="Pred")
        plt.grid(True)
        plt.ylabel("Disp")
        plt.title(f"Vertex {idx} | RMSE={rmse:.2e}, Corr={corr:.3f}")

        if i == 0:
            plt.legend()

    plt.xlabel("Frame")
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, f"multiple_vertex_curves{suffix_top(top_ratio)}.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Plot 3: observation-space consistency
# ============================================================

def plot_observation_consistency(data, fig_dir, obs_axis=1):
    sample_name = data["sample_name"]
    y = data["y"]
    y_hat = data["y_hat_seq"]

    if y_hat is None:
        print("Skip observation consistency: y_hat_seq.npy not available.")
        return

    T, K, _ = y.shape

    # Main summary curve: mean absolute motion over all tracks.
    y_amp = np.mean(np.abs(y), axis=(1, 2))
    yhat_amp = np.mean(np.abs(y_hat), axis=(1, 2))
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    corr = safe_corr(y_hat, y)

    plt.figure(figsize=(10, 4))
    plt.plot(y_amp, label="Input y mean abs")
    plt.plot(yhat_amp, "--", label="Projected y_hat mean abs")
    plt.xlabel("Frame")
    plt.ylabel("Mean abs 2D displacement")
    plt.title(f"{sample_name}: observation consistency | RMSE={rmse:.2e}, Corr={corr:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, "observation_consistency_mean_abs.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

    # Per-track plot, limited to avoid giant figure.
    max_tracks = min(K, 20)
    plt.figure(figsize=(12, max(4, 1.2 * max_tracks)))

    for k in range(max_tracks):
        plt.subplot(max_tracks, 1, k + 1)
        plt.plot(y[:, k, obs_axis], label="y" if k == 0 else None)
        plt.plot(y_hat[:, k, obs_axis], "--", label="y_hat" if k == 0 else None)
        plt.ylabel(f"P{k}")
        plt.grid(True)
        if k == 0:
            plt.legend()

    plt.xlabel("Frame")
    plt.suptitle(f"{sample_name}: per-track y vs y_hat, axis {obs_axis}")
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, "observation_consistency_tracks.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Plot 4: 3D uncertainty / error
# ============================================================

def plot_frame_3d(data, fig_dir, t_eval=0, top_ratio=1.0):
    sample_name = data["sample_name"]
    X0 = data["X0"]
    U_pred_seq = data["U_pred_seq"]
    U_std_seq = data["U_std_seq"]
    U_seq = data["U_seq"]

    T, N, _ = U_pred_seq.shape
    t_eval = int(np.clip(t_eval, 0, T - 1))

    mask = get_topk_mask(U_std_seq, top_ratio)
    valid = np.where(mask)[0]

    U_pred = U_pred_seq[t_eval]
    U_std = U_std_seq[t_eval]
    X_pred = X0 + U_pred

    pred_mag = np.linalg.norm(U_pred, axis=1)
    std_mag = np.linalg.norm(U_std, axis=1)

    if U_seq is not None:
        U_gt = U_seq[t_eval]
        X_gt = X0 + U_gt
        gt_mag = np.linalg.norm(U_gt, axis=1)
        err = np.linalg.norm(U_pred - U_gt, axis=1)
    else:
        X_gt = None
        gt_mag = None
        err = None

    ncols = 4 if U_seq is not None else 2
    fig = plt.figure(figsize=(5.5 * ncols, 5))

    if U_seq is not None:
        ax1 = fig.add_subplot(1, ncols, 1, projection="3d")
        p1 = ax1.scatter(X_gt[valid, 0], X_gt[valid, 1], X_gt[valid, 2], c=gt_mag[valid], s=8)
        ax1.set_title("GT displacement")
        fig.colorbar(p1, ax=ax1, shrink=0.6)

        ax2 = fig.add_subplot(1, ncols, 2, projection="3d")
        p2 = ax2.scatter(X_pred[valid, 0], X_pred[valid, 1], X_pred[valid, 2], c=pred_mag[valid], s=8)
        ax2.set_title("Pred mean")
        fig.colorbar(p2, ax=ax2, shrink=0.6)

        ax3 = fig.add_subplot(1, ncols, 3, projection="3d")
        p3 = ax3.scatter(X0[valid, 0], X0[valid, 1], X0[valid, 2], c=std_mag[valid], s=8)
        ax3.set_title("Pred std")
        fig.colorbar(p3, ax=ax3, shrink=0.6)

        ax4 = fig.add_subplot(1, ncols, 4, projection="3d")
        p4 = ax4.scatter(X0[valid, 0], X0[valid, 1], X0[valid, 2], c=err[valid], s=8)
        ax4.set_title("Error")
        fig.colorbar(p4, ax=ax4, shrink=0.6)

        axes = [ax1, ax2, ax3, ax4]

    else:
        ax1 = fig.add_subplot(1, ncols, 1, projection="3d")
        p1 = ax1.scatter(X_pred[valid, 0], X_pred[valid, 1], X_pred[valid, 2], c=pred_mag[valid], s=8)
        ax1.set_title("Pred mean")
        fig.colorbar(p1, ax=ax1, shrink=0.6)

        ax2 = fig.add_subplot(1, ncols, 2, projection="3d")
        p2 = ax2.scatter(X0[valid, 0], X0[valid, 1], X0[valid, 2], c=std_mag[valid], s=8)
        ax2.set_title("Pred std")
        fig.colorbar(p2, ax=ax2, shrink=0.6)

        axes = [ax1, ax2]

    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-60)

    plt.suptitle(f"{sample_name}: frame {t_eval} | {label_top(top_ratio)}")
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, f"frame_{t_eval:04d}_3d{suffix_top(top_ratio)}.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Plot 5: W correspondence visualization
# ============================================================

def plot_W_correspondence(data, fig_dir, track_id=0, t_eval=0, top_m=20):
    sample_name = data["sample_name"]
    X0 = data["X0"]
    W_seq = data["W_pred_seq"]
    W_mean = data["W_pred_mean"]
    W_idx = data["W_idx"]
    W_val = data["W_val"]

    if W_seq is None and W_mean is None:
        print("Skip W correspondence: W_pred_seq.npy or W_pred_mean.npy not available. Use eval.py --save_W.")
        return

    if W_seq is not None:
        T = W_seq.shape[0]
        t_eval = int(np.clip(t_eval, 0, T - 1))
        W = W_seq[t_eval]
    else:
        W = W_mean

    K, N = W.shape
    track_id = int(np.clip(track_id, 0, K - 1))

    w = W[track_id]
    top_idx = np.argsort(w)[::-1][:top_m]

    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1, projection="3d")
    p = ax1.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c=w, s=5)
    ax1.scatter(X0[top_idx, 0], X0[top_idx, 1], X0[top_idx, 2], s=35, marker="x")
    ax1.set_title(f"Predicted W distribution\ntrack {track_id}, frame {t_eval}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=20, azim=-60)
    plt.colorbar(p, ax=ax1, shrink=0.6)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(np.arange(top_m), w[top_idx])
    ax2.set_xlabel("Top predicted vertex rank")
    ax2.set_ylabel("Predicted correspondence probability")
    ax2.set_title("Top W probabilities")
    ax2.grid(True, axis="y")

    # Mark GT barycentric vertices if available.
    if W_idx is not None and W_val is not None:
        gt_vertices = W_idx[track_id]
        gt_weights = W_val[track_id]
        print(f"\n========== W correspondence: {sample_name} ==========")
        print("track:", track_id)
        print("GT vertices:", gt_vertices.tolist())
        print("GT barycentric:", gt_weights.tolist())
        print("Top predicted vertices:", top_idx[:10].tolist())
        print("Top predicted probabilities:", w[top_idx[:10]].tolist())

    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, f"W_correspondence_track{track_id:02d}_frame{t_eval:04d}.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_W_entropy(data, fig_dir):
    sample_name = data["sample_name"]
    Went = data["W_entropy_seq"]

    if Went is None:
        print("Skip W entropy: W_entropy_seq.npy not available.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(np.mean(Went, axis=1), label="mean W entropy over tracks")
    plt.fill_between(
        np.arange(Went.shape[0]),
        np.min(Went, axis=1),
        np.max(Went, axis=1),
        alpha=0.25,
        label="min-max entropy",
    )
    plt.xlabel("Frame")
    plt.ylabel("W entropy")
    plt.title(f"{sample_name}: correspondence uncertainty over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, "W_entropy_over_time.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Plot 6: error vs std + confidence filtering
# ============================================================

def plot_error_std_scatter(data, fig_dir, top_ratio=1.0):
    sample_name = data["sample_name"]
    U_seq = data["U_seq"]
    U_pred_seq = data["U_pred_seq"]
    U_std_seq = data["U_std_seq"]

    if U_seq is None:
        print("Skip error-vs-std: U_seq.npy not available.")
        return

    mask = get_topk_mask(U_std_seq, top_ratio)
    valid = np.where(mask)[0]

    vertex_error = np.linalg.norm(U_pred_seq[:, valid, :] - U_seq[:, valid, :], axis=2).reshape(-1)
    vertex_std = np.linalg.norm(U_std_seq[:, valid, :], axis=2).reshape(-1)

    corr = safe_corr(vertex_std, vertex_error)

    max_points = 50000
    n = len(vertex_error)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        x = vertex_std[idx]
        y = vertex_error[idx]
    else:
        x = vertex_std
        y = vertex_error

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=3, alpha=0.25)
    plt.xlabel("Predicted vertex std magnitude")
    plt.ylabel("Vertex error magnitude")
    plt.title(f"{sample_name}: error vs std | {label_top(top_ratio)} | corr={corr:.3f}")
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, f"error_vs_std{suffix_top(top_ratio)}.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_confidence_filtering(data, fig_dir):
    sample_name = data["sample_name"]
    U_seq = data["U_seq"]
    U_pred_seq = data["U_pred_seq"]
    U_std_seq = data["U_std_seq"]

    if U_seq is None:
        print("Skip confidence filtering: U_seq.npy not available.")
        return

    vertex_error = np.linalg.norm(U_pred_seq - U_seq, axis=2)  # [T,N]
    std_score = vertex_std_score(U_std_seq, mode="mean")       # [N]
    order = np.argsort(std_score)

    keep_ratios = np.linspace(0.05, 1.0, 20)
    mean_errors = []

    print(f"\n========== Confidence Filtering: {sample_name} ==========")
    for r in [0.05, 0.1, 0.2, 0.5, 1.0]:
        k = max(1, int(round(r * U_seq.shape[1])))
        keep = order[:k]
        err = float(np.mean(vertex_error[:, keep]))
        print(f"Top {int(r*100):3d}% lowest-std vertices | mean error = {err:.6e}")

    for r in keep_ratios:
        k = max(1, int(round(r * U_seq.shape[1])))
        keep = order[:k]
        mean_errors.append(float(np.mean(vertex_error[:, keep])))

    plt.figure(figsize=(7, 4))
    plt.plot(keep_ratios * 100, mean_errors, marker="o")
    plt.xlabel("Kept vertices with lowest predicted std (%)")
    plt.ylabel("Mean vertex error")
    plt.title(f"{sample_name}: confidence filtering")
    plt.grid(True)
    plt.tight_layout()

    out = os.path.join(fig_dir, sample_name, "confidence_filtering.png")
    ensure_dir(os.path.dirname(out))
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# Dataset summary plots
# ============================================================

def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row, key):
    try:
        return float(row[key])
    except Exception:
        return np.nan


def plot_dataset_summary(report_dir, fig_dir):
    path = os.path.join(report_dir, "eval_joint_UW_per_sample_metrics.csv")
    rows = read_csv_rows(path)

    if len(rows) == 0:
        print("No dataset summary csv found:", path)
        return

    def arr(key):
        return np.array([to_float(r, key) for r in rows], dtype=np.float64)

    x = np.arange(len(rows))

    metrics_to_bar = [
        ("obs_rmse", "Observation RMSE"),
        ("relative_l2", "Relative L2"),
        ("curve_corr", "Curve correlation"),
        ("W_top1_acc", "W top-1 accuracy"),
        ("error_std_corr_vertex", "Error-std correlation"),
        ("W_entropy_mean", "Mean W entropy"),
    ]

    ensure_dir(fig_dir)

    for key, title in metrics_to_bar:
        vals = arr(key)
        if np.all(np.isnan(vals)):
            continue

        plt.figure(figsize=(12, 4))
        plt.bar(x, vals)
        plt.xlabel("Test sample")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, axis="y")
        plt.tight_layout()

        out = os.path.join(fig_dir, f"dataset_{key}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)

    # Scatter: W accuracy vs relative L2
    wacc = arr("W_top1_acc")
    rel = arr("relative_l2")
    if not np.all(np.isnan(wacc)) and not np.all(np.isnan(rel)):
        plt.figure(figsize=(6, 5))
        plt.scatter(wacc, rel, s=35)
        plt.xlabel("W top-1 accuracy")
        plt.ylabel("Relative L2")
        plt.title("Correspondence accuracy vs reconstruction error")
        plt.grid(True)
        plt.tight_layout()

        out = os.path.join(fig_dir, "dataset_Wacc_vs_RelL2.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)

    # Scatter: observation RMSE vs U error
    obs = arr("obs_rmse")
    if not np.all(np.isnan(obs)) and not np.all(np.isnan(rel)):
        plt.figure(figsize=(6, 5))
        plt.scatter(obs, rel, s=35)
        plt.xlabel("Observation RMSE")
        plt.ylabel("Relative L2")
        plt.title("Observation consistency vs reconstruction error")
        plt.grid(True)
        plt.tight_layout()

        out = os.path.join(fig_dir, "dataset_ObsRMSE_vs_RelL2.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print("Saved:", out)


# ============================================================
# Main
# ============================================================

def visualize_one_sample(sample_dir, args):
    data = load_sample_and_prediction(sample_dir, args.pred_dir)
    data["U_mu_samples"] = maybe_load(os.path.join(args.pred_dir, "U_mu_samples.npy"), np.float32)

    if args.plot in ["all", "curve"]:
        plot_vibration_curve(data, args.fig_dir, args.disp_axis, args.top_ratio)

    if args.plot in ["all", "multi"]:
        plot_multiple_vertex_curves(data, args.fig_dir, args.disp_axis, args.top_ratio, args.num_vertices)

    if args.plot in ["all", "obs"]:
        plot_observation_consistency(data, args.fig_dir, args.obs_axis)

    if args.plot in ["all", "frame3d"]:
        plot_frame_3d(data, args.fig_dir, args.t_eval, args.top_ratio)

    if args.plot in ["all", "W"]:
        plot_W_correspondence(data, args.fig_dir, args.track_id, args.t_eval, args.top_m)
        plot_W_entropy(data, args.fig_dir)

    if args.plot in ["all", "scatter"]:
        plot_error_std_scatter(data, args.fig_dir, args.top_ratio)

    if args.plot in ["all", "filter"]:
        plot_confidence_filtering(data, args.fig_dir)

    if args.plot in ["all", "posterior"]:
        plot_posterior_samples(data, args.fig_dir, args.disp_axis, args.num_posterior_samples)


def visualize(args):
    ensure_dir(args.fig_dir)

    if args.dataset_summary:
        plot_dataset_summary(args.report_dir, args.fig_dir)
        return

    sample_dirs = find_sample_dirs(args.data_dir)

    if args.sample_name is not None:
        sample_dirs = [d for d in sample_dirs if sample_name_from_dir(d) == args.sample_name]
        if len(sample_dirs) == 0:
            raise ValueError(f"sample_name={args.sample_name} not found under {args.data_dir}")

    if len(sample_dirs) > 1 and not args.visualize_all:
        print("Multiple samples found. Visualizing only the first sample.")
        print("Use --sample_name sample_XXXX or --visualize_all.")
        sample_dirs = sample_dirs[:1]

    print("plot:", args.plot)
    print("top_ratio:", args.top_ratio)

    for d in sample_dirs:
        print("\nVisualizing:", d)
        visualize_one_sample(d, args)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)
    parser.add_argument("--fig_dir", type=str, default=DEFAULT_FIG_DIR)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)

    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["all", "curve", "multi", "obs", "frame3d", "W", "scatter", "filter", "posterior"],
    )

    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--visualize_all", action="store_true")
    parser.add_argument("--dataset_summary", action="store_true")

    parser.add_argument("--disp_axis", type=int, default=1)
    parser.add_argument("--obs_axis", type=int, default=1)
    parser.add_argument("--t_eval", type=int, default=0)

    parser.add_argument("--top_ratio", type=float, default=1.0)
    parser.add_argument("--num_vertices", type=int, default=8)

    parser.add_argument("--track_id", type=int, default=0)
    parser.add_argument("--top_m", type=int, default=20)

    parser.add_argument("--num_posterior_samples", type=int, default=5)

    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
