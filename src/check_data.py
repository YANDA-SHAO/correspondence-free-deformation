# src/check_data.py

import os
import json
import argparse
import numpy as np


def describe_array(name, arr):
    print(f"\n========== {name} ==========")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    print("min:", np.min(arr))
    print("max:", np.max(arr))
    print("mean:", np.mean(arr))
    print("std:", np.std(arr))
    print("mean abs:", np.mean(np.abs(arr)))
    print("max abs:", np.max(np.abs(arr)))
    print("has nan:", np.isnan(arr).any())
    print("has inf:", np.isinf(arr).any())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    print("DATA_DIR:", os.path.abspath(DATA_DIR))

    files = [
        "X0.npy",
        "U.npy",
        "U_seq.npy",
        "y.npy",
        "y_pixel.npy",
        "observed_idx.npy",
        "camera_K.npy",
        "camera_R.npy",
        "camera_t.npy",
        "amp_t.npy",
        "phi_1.npy",
        "meta.json",
    ]

    print("\n========== File Check ==========")
    missing_files = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        exists = os.path.exists(path)
        print(f"{f:20s} exists: {exists}")
        if not exists:
            missing_files.append(f)

    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")

    X0 = np.load(os.path.join(DATA_DIR, "X0.npy"))
    U = np.load(os.path.join(DATA_DIR, "U.npy"))
    U_seq = np.load(os.path.join(DATA_DIR, "U_seq.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    y_pixel = np.load(os.path.join(DATA_DIR, "y_pixel.npy"))
    observed_idx = np.load(os.path.join(DATA_DIR, "observed_idx.npy"))
    amp_t = np.load(os.path.join(DATA_DIR, "amp_t.npy"))

    describe_array("X0", X0)
    describe_array("U", U)
    describe_array("U_seq", U_seq)
    describe_array("y", y)
    describe_array("y_pixel", y_pixel)
    describe_array("amp_t", amp_t)

    print("\n========== Shape Check ==========")
    T, N, D = U_seq.shape
    K = y.shape[1]

    print("N vertices:", N)
    print("T frames:", T)
    print("D displacement dim:", D)
    print("K observed points:", K)

    assert X0.shape == (N, 3), f"X0 shape mismatch: {X0.shape}"
    assert U.shape == (N, 3), f"U shape mismatch: {U.shape}"
    assert y.shape[0] == T, f"y T mismatch: {y.shape[0]} vs {T}"
    assert y.shape[2] == 2, f"y should be [T,K,2], got {y.shape}"
    assert observed_idx.shape[0] == K, f"observed_idx K mismatch: {observed_idx.shape[0]} vs {K}"

    print("Shape check: PASSED")

    print("\n========== Observed Points ==========")
    print("observed_idx shape:", observed_idx.shape)
    print("observed_idx min:", observed_idx.min())
    print("observed_idx max:", observed_idx.max())
    print("unique observed:", len(np.unique(observed_idx)))
    print("first 20 observed_idx:", observed_idx[:20])

    print("\n========== Displacement Magnitude ==========")
    U_mag = np.linalg.norm(U, axis=1)
    U_seq_mag = np.linalg.norm(U_seq, axis=2)

    print("U max disp:", U_mag.max())
    print("U mean disp:", U_mag.mean())
    print("U median disp:", np.median(U_mag))
    print("U_seq max disp:", U_seq_mag.max())
    print("U_seq mean disp:", U_seq_mag.mean())

    print("\n========== y Signal Strength ==========")
    print("y max abs:", np.max(np.abs(y)))
    print("y mean abs:", np.mean(np.abs(y)))
    print("y_pixel max abs:", np.max(np.abs(y_pixel)))
    print("y_pixel mean abs:", np.mean(np.abs(y_pixel)))

    if np.max(np.abs(y)) < 1e-4:
        print("WARNING: y is very small. Consider using --no_normalize_y.")
    if np.mean(np.abs(y)) < 1e-5:
        print("WARNING: mean abs y is extremely small.")

    print("\n========== Input-output Relation Check ==========")
    y_frame_amp = np.mean(np.abs(y), axis=(1, 2))
    U_frame_amp = np.mean(np.abs(U_seq), axis=(1, 2))

    corr = np.corrcoef(y_frame_amp, U_frame_amp)[0, 1]
    print("corr(mean_abs_y_per_frame, mean_abs_U_per_frame):", corr)

    print("y_frame_amp min/max:", y_frame_amp.min(), y_frame_amp.max())
    print("U_frame_amp min/max:", U_frame_amp.min(), U_frame_amp.max())

    print("\n========== Camera Check ==========")
    for name in ["camera_K.npy", "camera_R.npy", "camera_t.npy"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            arr = np.load(path)
            print(f"\n{name}:")
            print(arr)

    meta_path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        print("\n========== Meta ==========")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        keys = [
            "split",
            "sample_index",
            "projection",
            "y_unit",
            "T",
            "K_obs",
            "amplitude",
            "noise_std",
            "frequency_hz",
            "frequency_scale",
            "phase",
            "camera_intrinsics",
            "camera_eye",
            "camera_target",
        ]

        for k in keys:
            print(f"{k}: {meta.get(k)}")


if __name__ == "__main__":
    main()