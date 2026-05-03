# src/visualize.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_v1", "test", "sample_0000")
DEFAULT_PRED_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")
DEFAULT_FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_vibration_curve(data_dir, pred_dir, fig_dir, disp_axis=1):
    U_seq = np.load(os.path.join(data_dir, "U_seq.npy"))
    U_pred_seq = np.load(os.path.join(pred_dir, "U_pred_seq.npy"))
    U_std_seq = np.load(os.path.join(pred_dir, "U_std_seq.npy"))

    amp = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
    idx = int(np.argmax(amp))

    gt = U_seq[:, idx, disp_axis]
    pred = U_pred_seq[:, idx, disp_axis]
    std = U_std_seq[:, idx, disp_axis]

    err = pred - gt
    rmse = np.sqrt(np.mean(err ** 2))

    if np.std(gt) > 1e-12 and np.std(pred) > 1e-12:
        corr = np.corrcoef(gt, pred)[0, 1]
    else:
        corr = np.nan

    plt.figure(figsize=(10, 4))
    plt.plot(gt, label="GT")
    plt.plot(pred, "--", label="Pred")
    plt.fill_between(
        np.arange(len(gt)),
        pred - 2.0 * std,
        pred + 2.0 * std,
        alpha=0.25,
        label="Pred ±2 std",
    )
    plt.xlabel("Frame")
    plt.ylabel(f"Displacement axis {disp_axis}")
    plt.title(f"Vibration curve at vertex {idx} | RMSE={rmse:.2e}, Corr={corr:.5f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "vibration_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


def plot_multiple_vertex_curves(data_dir, pred_dir, fig_dir, num_points=8, disp_axis=1):
    U_seq = np.load(os.path.join(data_dir, "U_seq.npy"))
    U_pred_seq = np.load(os.path.join(pred_dir, "U_pred_seq.npy"))

    T, N, _ = U_seq.shape

    amp = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
    sorted_idx = np.argsort(amp)
    chosen_idx = sorted_idx[np.linspace(0, N - 1, num_points).astype(int)]

    print("\n========== Multiple Vertex Curve Errors ==========")
    print(
        f"{'vertex':>8} | "
        f"{'GT max':>12} | "
        f"{'Pred max':>12} | "
        f"{'RMSE':>12} | "
        f"{'MAE':>12} | "
        f"{'Max err':>12} | "
        f"{'Corr':>10}"
    )
    print("-" * 90)

    plt.figure(figsize=(12, 2.2 * num_points))

    for i, idx in enumerate(chosen_idx):
        gt = U_seq[:, idx, disp_axis]
        pred = U_pred_seq[:, idx, disp_axis]

        err = pred - gt
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(np.abs(err))
        max_err = np.max(np.abs(err))
        gt_max = np.max(np.abs(gt))
        pred_max = np.max(np.abs(pred))

        if np.std(gt) > 1e-12 and np.std(pred) > 1e-12:
            corr = np.corrcoef(gt, pred)[0, 1]
        else:
            corr = np.nan

        print(
            f"{idx:8d} | "
            f"{gt_max:12.6e} | "
            f"{pred_max:12.6e} | "
            f"{rmse:12.6e} | "
            f"{mae:12.6e} | "
            f"{max_err:12.6e} | "
            f"{corr:10.6f}"
        )

        plt.subplot(num_points, 1, i + 1)
        plt.plot(gt, label="GT")
        plt.plot(pred, "--", label="Pred")
        plt.ylabel("Disp")
        plt.title(f"Vertex {idx} | RMSE={rmse:.2e}, Corr={corr:.4f}")
        plt.grid(True)

        if i == 0:
            plt.legend()

    plt.xlabel("Frame")
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "multiple_vertex_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


def plot_observed_2d_curves(data_dir, fig_dir, obs_axis=1):
    y = np.load(os.path.join(data_dir, "y.npy"))

    T, K, _ = y.shape

    plt.figure(figsize=(12, 1.6 * K))

    for k in range(K):
        curve = y[:, k, obs_axis]

        plt.subplot(K, 1, k + 1)
        plt.plot(curve)
        plt.ylabel(f"P{k}")
        plt.grid(True)

    plt.xlabel("Frame")
    plt.suptitle(f"Observed sparse 2D displacement curves, axis {obs_axis}")
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "observed_2d_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


def plot_mode_amplitude(data_dir, pred_dir, fig_dir, disp_axis=1):
    U_seq = np.load(os.path.join(data_dir, "U_seq.npy"))
    U_pred_seq = np.load(os.path.join(pred_dir, "U_pred_seq.npy"))

    gt_amp = np.max(np.abs(U_seq[:, :, disp_axis]), axis=0)
    pred_amp = np.max(np.abs(U_pred_seq[:, :, disp_axis]), axis=0)

    order = np.argsort(gt_amp)

    plt.figure(figsize=(10, 4))
    plt.plot(gt_amp[order], label="GT amplitude")
    plt.plot(pred_amp[order], "--", label="Pred amplitude")
    plt.xlabel("Vertex ordered by GT amplitude")
    plt.ylabel(f"Max abs displacement axis {disp_axis}")
    plt.title("Mode amplitude along beam")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "mode_amplitude.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


def plot_frame_3d_scatter(data_dir, pred_dir, fig_dir, t_eval=0):
    X0 = np.load(os.path.join(data_dir, "X0.npy"))
    U_seq = np.load(os.path.join(data_dir, "U_seq.npy"))
    U_pred_seq = np.load(os.path.join(pred_dir, "U_pred_seq.npy"))

    U_t = U_seq[t_eval]
    U_pred_t = U_pred_seq[t_eval]

    X_gt = X0 + U_t
    X_pred = X0 + U_pred_t

    gt_mag = np.linalg.norm(U_t, axis=1)
    pred_mag = np.linalg.norm(U_pred_t, axis=1)
    err = np.linalg.norm(U_pred_t - U_t, axis=1)

    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    p1 = ax1.scatter(X_gt[:, 0], X_gt[:, 1], X_gt[:, 2], c=gt_mag, s=4)
    ax1.set_title(f"GT displacement, frame {t_eval}")
    fig.colorbar(p1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(132, projection="3d")
    p2 = ax2.scatter(X_pred[:, 0], X_pred[:, 1], X_pred[:, 2], c=pred_mag, s=4)
    ax2.set_title(f"Pred displacement, frame {t_eval}")
    fig.colorbar(p2, ax=ax2, shrink=0.6)

    ax3 = fig.add_subplot(133, projection="3d")
    p3 = ax3.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c=err, s=4)
    ax3.set_title("Vertex error")
    fig.colorbar(p3, ax=ax3, shrink=0.6)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=20, azim=-60)

    plt.tight_layout()

    save_path = os.path.join(fig_dir, f"frame_{t_eval:04d}_3d_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print("Saved:", save_path)


def visualize(args):
    ensure_dir(args.fig_dir)

    if args.plot in ["all", "curve"]:
        plot_vibration_curve(args.data_dir, args.pred_dir, args.fig_dir, args.disp_axis)

    if args.plot in ["all", "multi"]:
        plot_multiple_vertex_curves(
            args.data_dir,
            args.pred_dir,
            args.fig_dir,
            args.num_points,
            args.disp_axis,
        )

    if args.plot in ["all", "observed"]:
        plot_observed_2d_curves(args.data_dir, args.fig_dir, args.obs_axis)

    if args.plot in ["all", "amplitude"]:
        plot_mode_amplitude(args.data_dir, args.pred_dir, args.fig_dir, args.disp_axis)

    if args.plot in ["all", "frame3d"]:
        plot_frame_3d_scatter(args.data_dir, args.pred_dir, args.fig_dir, args.t_eval)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pred_dir", type=str, default=DEFAULT_PRED_DIR)
    parser.add_argument("--fig_dir", type=str, default=DEFAULT_FIG_DIR)

    parser.add_argument(
        "--plot",
        type=str,
        default="all",
        choices=["all", "curve", "multi", "observed", "amplitude", "frame3d"],
    )

    parser.add_argument("--disp_axis", type=int, default=1)
    parser.add_argument("--obs_axis", type=int, default=1)
    parser.add_argument("--num_points", type=int, default=8)
    parser.add_argument("--t_eval", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())