# src/eval_dataset.py

import os
import csv
import glob
import argparse
import numpy as np
import torch

from model import ProbabilisticSingleFrameYtoU


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_v1", "test")
DEFAULT_WEIGHT_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "weights", "prob_single_frame_y_to_Ut_best.pth"
)
DEFAULT_PRED_ROOT = os.path.join(PROJECT_ROOT, "outputs", "predictions", "test")
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


def load_model(weight_path, y_dim, U_dim, device):
    ckpt = torch.load(weight_path, map_location=device)

    if ckpt.get("y_dim", y_dim) != y_dim:
        raise ValueError(f"y_dim mismatch: checkpoint={ckpt.get('y_dim')} current={y_dim}")
    if ckpt.get("U_dim", U_dim) != U_dim:
        raise ValueError(f"U_dim mismatch: checkpoint={ckpt.get('U_dim')} current={U_dim}")

    model = ProbabilisticSingleFrameYtoU(
        y_dim=y_dim,
        U_dim=U_dim,
        z_dim=ckpt.get("z_dim", 32),
        hidden_dim=ckpt.get("hidden_dim", 256),
        logvar_min=ckpt.get("logvar_min", -20.0),
        logvar_max=ckpt.get("logvar_max", -10.0),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_normalization(weight_path):
    norm_path = os.path.join(os.path.dirname(weight_path), "normalization.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")

    norm = np.load(norm_path)
    return {
        "path": norm_path,
        "y_mean": norm["y_mean"],
        "y_std": norm["y_std"],
        "U_mean": norm["U_mean"],
        "U_std": norm["U_std"],
    }


def find_sample_dirs(data_dir):
    sample_dirs = sorted(glob.glob(os.path.join(data_dir, "sample_*")))
    if len(sample_dirs) == 0:
        raise ValueError(
            f"No sample_* folders found in {data_dir}. "
            "For fair testing, data_dir should be data/dataset_v1/test, not one old sample folder."
        )
    return sample_dirs


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def infer_one_sample(sample_dir, model, norm, device, args):
    y_path = os.path.join(sample_dir, "y.npy")
    U_path = os.path.join(sample_dir, "U_seq.npy")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing file: {y_path}")
    if not os.path.exists(U_path):
        raise FileNotFoundError(f"Missing file: {U_path}")

    y = np.load(y_path).astype(np.float32)          # [T,K,2]
    U_seq = np.load(U_path).astype(np.float32)      # [T,N,3]

    T, K, _ = y.shape
    N = U_seq.shape[1]

    y_flat = y.reshape(T, -1)

    if y_flat.shape[1] != norm["y_mean"].shape[1]:
        raise ValueError(
            f"K/y_dim mismatch in {sample_dir}: y_dim={y_flat.shape[1]}, "
            f"normalization y_dim={norm['y_mean'].shape[1]}"
        )

    y_norm = (y_flat - norm["y_mean"]) / norm["y_std"]

    U_pred_batches = []
    U_std_batches = []

    with torch.no_grad():
        for start in range(0, T, args.batch_size):
            end = start + args.batch_size
            y_batch = torch.from_numpy(y_norm[start:end].astype(np.float32)).to(device)

            U_mu_flat, U_logvar_flat, _, _ = model(y_batch, sample_z=False)

            U_mu_norm = U_mu_flat.cpu().numpy()
            U_mu = U_mu_norm * norm["U_std"] + norm["U_mean"]
            U_mu = U_mu.reshape(-1, N, 3)

            U_logvar = U_logvar_flat.cpu().numpy()
            U_std_norm = np.sqrt(np.exp(U_logvar))
            U_std = U_std_norm * norm["U_std"]
            U_std = U_std.reshape(-1, N, 3)

            U_pred_batches.append(U_mu)
            U_std_batches.append(U_std)

    U_pred_seq = np.concatenate(U_pred_batches, axis=0).astype(np.float32)
    U_std_seq = np.concatenate(U_std_batches, axis=0).astype(np.float32)
    error_seq = U_pred_seq - U_seq

    mse = float(np.mean(error_seq ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error_seq)))
    rel_l2 = float(
        np.linalg.norm(error_seq.reshape(-1)) /
        (np.linalg.norm(U_seq.reshape(-1)) + 1e-12)
    )

    axis = args.disp_axis
    gt_amp_per_vertex = np.max(np.abs(U_seq[:, :, axis]), axis=0)
    vertex_idx = int(np.argmax(gt_amp_per_vertex))

    gt_curve = U_seq[:, vertex_idx, axis]
    pred_curve = U_pred_seq[:, vertex_idx, axis]
    curve_rmse = float(np.sqrt(np.mean((pred_curve - gt_curve) ** 2)))
    curve_corr = safe_corr(gt_curve, pred_curve)

    gt_mag = np.linalg.norm(U_seq, axis=2)
    pred_mag = np.linalg.norm(U_pred_seq, axis=2)
    vertex_error = np.linalg.norm(error_seq, axis=2)

    free_end_idx = int(np.argmax(np.max(gt_mag, axis=0)))
    free_end_rmse = float(np.sqrt(np.mean(np.sum(error_seq[:, free_end_idx, :] ** 2, axis=1))))

    if args.save_predictions:
        sample_name = os.path.basename(sample_dir)
        save_dir = os.path.join(args.pred_root, sample_name)
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "U_pred_seq.npy"), U_pred_seq)
        np.save(os.path.join(save_dir, "U_std_seq.npy"), U_std_seq)
        np.save(os.path.join(save_dir, "error_seq.npy"), error_seq.astype(np.float32))
        np.save(os.path.join(save_dir, "curve_gt.npy"), gt_curve.astype(np.float32))
        np.save(os.path.join(save_dir, "curve_pred.npy"), pred_curve.astype(np.float32))
        np.save(os.path.join(save_dir, "curve_vertex_idx.npy"), np.array([vertex_idx], dtype=np.int64))

    return {
        "sample": os.path.basename(sample_dir),
        "T": T,
        "N": N,
        "K": K,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "relative_l2": rel_l2,
        "curve_vertex_idx": vertex_idx,
        "curve_rmse": curve_rmse,
        "curve_corr": curve_corr,
        "free_end_idx": free_end_idx,
        "free_end_rmse": free_end_rmse,
        "gt_max_disp": float(np.max(gt_mag)),
        "pred_max_disp": float(np.max(pred_mag)),
        "mean_vertex_error": float(np.mean(vertex_error)),
        "max_vertex_error": float(np.max(vertex_error)),
    }


def write_csv(rows, path):
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows):
    numeric_keys = [
        "mse", "rmse", "mae", "relative_l2", "curve_rmse", "curve_corr",
        "free_end_rmse", "gt_max_disp", "pred_max_disp", "mean_vertex_error", "max_vertex_error"
    ]

    summary = {"num_samples": len(rows)}
    for k in numeric_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[f"mean_{k}"] = np.nan
            summary[f"std_{k}"] = np.nan
        else:
            summary[f"mean_{k}"] = float(np.mean(vals))
            summary[f"std_{k}"] = float(np.std(vals))
    return summary


def main(args):
    os.makedirs(args.report_dir, exist_ok=True)
    if args.save_predictions:
        os.makedirs(args.pred_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)
    print("Test data_dir:", os.path.abspath(args.data_dir))
    print("Weight path:", os.path.abspath(args.weight_path))

    sample_dirs = find_sample_dirs(args.data_dir)
    print("Number of test samples:", len(sample_dirs))

    norm = load_normalization(args.weight_path)
    print("Normalization:", norm["path"])

    # Determine dimensions from the first sample.
    y0 = np.load(os.path.join(sample_dirs[0], "y.npy")).astype(np.float32)
    U0 = np.load(os.path.join(sample_dirs[0], "U_seq.npy")).astype(np.float32)
    y_dim = y0.reshape(y0.shape[0], -1).shape[1]
    U_dim = U0.shape[1] * 3

    model = load_model(args.weight_path, y_dim, U_dim, device)

    rows = []
    for i, sample_dir in enumerate(sample_dirs):
        row = infer_one_sample(sample_dir, model, norm, device, args)
        rows.append(row)
        print(
            f"[{i+1:03d}/{len(sample_dirs):03d}] {row['sample']} | "
            f"RMSE={row['rmse']:.6e} | "
            f"RelL2={row['relative_l2']:.6e} | "
            f"CurveCorr={row['curve_corr']:.6f} | "
            f"PredMax={row['pred_max_disp']:.6e} | GTMax={row['gt_max_disp']:.6e}"
        )

    per_sample_csv = os.path.join(args.report_dir, "test_per_sample_metrics.csv")
    write_csv(rows, per_sample_csv)

    summary = summarize(rows)
    summary_csv = os.path.join(args.report_dir, "test_summary_metrics.csv")
    write_csv([summary], summary_csv)

    print("\n========== Fair Test Summary: All Test Samples ==========")
    print("num_samples:", summary["num_samples"])
    print(f"Mean RMSE        : {summary['mean_rmse']:.6e} ± {summary['std_rmse']:.6e}")
    print(f"Mean MAE         : {summary['mean_mae']:.6e} ± {summary['std_mae']:.6e}")
    print(f"Mean Relative L2 : {summary['mean_relative_l2']:.6e} ± {summary['std_relative_l2']:.6e}")
    print(f"Mean Curve Corr  : {summary['mean_curve_corr']:.6e} ± {summary['std_curve_corr']:.6e}")
    print(f"Mean FreeEnd RMSE: {summary['mean_free_end_rmse']:.6e} ± {summary['std_free_end_rmse']:.6e}")
    print(f"Mean Pred Max    : {summary['mean_pred_max_disp']:.6e}")
    print(f"Mean GT Max      : {summary['mean_gt_max_disp']:.6e}")
    print("\nSaved per-sample metrics to:", per_sample_csv)
    print("Saved summary metrics to:", summary_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--weight_path", type=str, default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--pred_root", type=str, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--report_dir", type=str, default=DEFAULT_REPORT_DIR)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--disp_axis", type=int, default=1)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    main(args)
