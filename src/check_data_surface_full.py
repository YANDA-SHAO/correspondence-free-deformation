# src/check_data.py
"""
Comprehensive data checker for dataset_v2_surface.

It checks:
1. Required files
2. Array shapes / dtype / nan / inf / magnitude
3. Surface correspondence:
   - obs_face_vertices shape
   - obs_barycentric shape
   - barycentric weights sum to 1
   - barycentric weights are non-negative
   - face vertices match faces[obs_face_id]
   - W_gt sparse consistency
   - optional W_gt_dense consistency
4. Projection consistency:
   - recompute surface points
   - recompute interpolated displacement
   - project with camera
   - compare recomputed y_clean / y_pixel with saved files
5. Input-output relation:
   - corr(mean_abs_y_per_frame, mean_abs_U_per_frame)
6. Camera depth validity
7. Meta information
"""

import os
import json
import argparse
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_v2_surface", "train", "sample_0000")


# ============================================================
# Basic utilities
# ============================================================

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
    print("has nan:", np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else False)
    print("has inf:", np.isinf(arr).any() if np.issubdtype(arr.dtype, np.floating) else False)


def safe_corr(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or b.size < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def load_required(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def perspective_project(X_world, K, R, t):
    X_cam = np.einsum("ij,...j->...i", R, X_world) + t
    z = X_cam[..., 2:3]
    z_safe = np.maximum(z, 1e-6)
    x_norm = X_cam[..., 0:1] / z_safe
    y_norm = X_cam[..., 1:2] / z_safe
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    uv = np.concatenate([u, v], axis=-1)
    return uv.astype(np.float32), X_cam.astype(np.float32)


def check_close(name, a, b, atol=1e-5, rtol=1e-5):
    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    print(f"{name}: ok={ok} | max diff={max_diff:.6e} | mean diff={mean_diff:.6e}")
    return ok, max_diff, mean_diff


# ============================================================
# Main checks
# ============================================================

def check_one_sample(args):
    data_dir = os.path.abspath(args.data_dir)
    print("DATA_DIR:", data_dir)

    required_files = [
        "X0.npy",
        "faces.npy",
        "U.npy",
        "U_seq.npy",
        "y.npy",
        "y_clean.npy",
        "y_pixel.npy",
        "track_uv0.npy",
        "track_uv_seq.npy",
        "obs_face_id.npy",
        "obs_face_vertices.npy",
        "obs_barycentric.npy",
        "W_gt_sparse_indices.npy",
        "W_gt_sparse_values.npy",
        "observed_idx.npy",
        "camera_K.npy",
        "camera_R.npy",
        "camera_t.npy",
        "camera_eye.npy",
        "camera_target.npy",
        "amp_t.npy",
        "phi_1.npy",
        "meta.json",
    ]

    optional_files = [
        "W_gt_dense.npy",
        "candidate_free_end_face_id.npy",
        "split_info.json",
        "deformed_mesh.obj",
    ]

    print("\n========== File Check ==========")
    for f in required_files:
        p = os.path.join(data_dir, f)
        print(f"{f:30s} exists: {os.path.exists(p)}")

    print("\n========== Optional File Check ==========")
    for f in optional_files:
        p = os.path.join(data_dir, f)
        print(f"{f:30s} exists: {os.path.exists(p)}")

    # Load core files
    X0 = load_required(data_dir, "X0.npy").astype(np.float32)
    faces = load_required(data_dir, "faces.npy").astype(np.int64)
    U = load_required(data_dir, "U.npy").astype(np.float32)
    U_seq = load_required(data_dir, "U_seq.npy").astype(np.float32)
    y = load_required(data_dir, "y.npy").astype(np.float32)
    y_clean = load_required(data_dir, "y_clean.npy").astype(np.float32)
    y_pixel = load_required(data_dir, "y_pixel.npy").astype(np.float32)
    track_uv0 = load_required(data_dir, "track_uv0.npy").astype(np.float32)
    track_uv_seq = load_required(data_dir, "track_uv_seq.npy").astype(np.float32)

    obs_face_id = load_required(data_dir, "obs_face_id.npy").astype(np.int64)
    obs_face_vertices = load_required(data_dir, "obs_face_vertices.npy").astype(np.int64)
    obs_barycentric = load_required(data_dir, "obs_barycentric.npy").astype(np.float32)
    W_idx = load_required(data_dir, "W_gt_sparse_indices.npy").astype(np.int64)
    W_val = load_required(data_dir, "W_gt_sparse_values.npy").astype(np.float32)
    observed_idx = load_required(data_dir, "observed_idx.npy").astype(np.int64)

    Kmat = load_required(data_dir, "camera_K.npy").astype(np.float32)
    R = load_required(data_dir, "camera_R.npy").astype(np.float32)
    t = load_required(data_dir, "camera_t.npy").astype(np.float32)
    camera_eye = load_required(data_dir, "camera_eye.npy").astype(np.float32)
    camera_target = load_required(data_dir, "camera_target.npy").astype(np.float32)
    amp_t = load_required(data_dir, "amp_t.npy").astype(np.float32)
    phi_1 = load_required(data_dir, "phi_1.npy").astype(np.float32)

    # Descriptions
    for name, arr in [
        ("X0", X0),
        ("faces", faces),
        ("U", U),
        ("U_seq", U_seq),
        ("y", y),
        ("y_clean", y_clean),
        ("y_pixel", y_pixel),
        ("track_uv0", track_uv0),
        ("track_uv_seq", track_uv_seq),
        ("obs_face_id", obs_face_id),
        ("obs_face_vertices", obs_face_vertices),
        ("obs_barycentric", obs_barycentric),
        ("W_gt_sparse_indices", W_idx),
        ("W_gt_sparse_values", W_val),
        ("observed_idx_compat", observed_idx),
        ("camera_K", Kmat),
        ("camera_R", R),
        ("camera_t", t),
        ("camera_eye", camera_eye),
        ("camera_target", camera_target),
        ("amp_t", amp_t),
        ("phi_1", phi_1),
    ]:
        describe_array(name, arr)

    print("\n========== Shape Check ==========")
    T, N, D = U_seq.shape
    K_obs = y.shape[1]

    shape_ok = True
    checks = [
        ("X0", X0.shape == (N, 3)),
        ("U", U.shape == (N, 3)),
        ("U_seq", U_seq.ndim == 3 and U_seq.shape[1:] == (N, 3)),
        ("y", y.ndim == 3 and y.shape[0] == T and y.shape[2] == 2),
        ("y_clean", y_clean.shape == y.shape),
        ("y_pixel", y_pixel.shape == y.shape),
        ("track_uv0", track_uv0.shape == (K_obs, 2)),
        ("track_uv_seq", track_uv_seq.shape == (T, K_obs, 2)),
        ("faces", faces.ndim == 2 and faces.shape[1] == 3),
        ("obs_face_id", obs_face_id.shape == (K_obs,)),
        ("obs_face_vertices", obs_face_vertices.shape == (K_obs, 3)),
        ("obs_barycentric", obs_barycentric.shape == (K_obs, 3)),
        ("W sparse indices", W_idx.shape == (K_obs, 3)),
        ("W sparse values", W_val.shape == (K_obs, 3)),
        ("observed_idx compat", observed_idx.shape == (K_obs,)),
        ("camera_K", Kmat.shape == (3, 3)),
        ("camera_R", R.shape == (3, 3)),
        ("camera_t", t.shape == (3,)),
    ]

    for name, ok in checks:
        print(f"{name:25s}: {ok}")
        shape_ok = shape_ok and ok

    print("Shape check:", "PASSED" if shape_ok else "FAILED")
    print("T frames:", T)
    print("N vertices:", N)
    print("K observations:", K_obs)

    print("\n========== Surface Correspondence Check ==========")

    face_id_valid = bool(np.all((obs_face_id >= 0) & (obs_face_id < faces.shape[0])))
    face_vertices_valid = bool(np.all((obs_face_vertices >= 0) & (obs_face_vertices < N)))
    bary_sum = obs_barycentric.sum(axis=1)
    bary_sum_ok = bool(np.allclose(bary_sum, 1.0, atol=1e-5))
    bary_nonneg_ok = bool(np.all(obs_barycentric >= -1e-7))
    bary_leq_ok = bool(np.all(obs_barycentric <= 1.0 + 1e-7))

    face_vertices_from_face_id = faces[obs_face_id]
    face_vertices_match = bool(np.array_equal(face_vertices_from_face_id, obs_face_vertices))

    W_idx_match = bool(np.array_equal(W_idx, obs_face_vertices))
    W_val_match = bool(np.allclose(W_val, obs_barycentric, atol=1e-6))

    dominant_corner = np.argmax(obs_barycentric, axis=1)
    observed_idx_expected = obs_face_vertices[np.arange(K_obs), dominant_corner]
    observed_idx_ok = bool(np.array_equal(observed_idx, observed_idx_expected))

    print("obs_face_id valid range:", face_id_valid)
    print("obs_face_vertices valid range:", face_vertices_valid)
    print("barycentric sum min/max:", float(bary_sum.min()), float(bary_sum.max()))
    print("barycentric sums to 1:", bary_sum_ok)
    print("barycentric non-negative:", bary_nonneg_ok)
    print("barycentric <= 1:", bary_leq_ok)
    print("faces[obs_face_id] == obs_face_vertices:", face_vertices_match)
    print("W sparse indices match face vertices:", W_idx_match)
    print("W sparse values match barycentric:", W_val_match)
    print("observed_idx compatibility matches dominant bary vertex:", observed_idx_ok)

    print("\nFirst 5 surface observations:")
    for k in range(min(5, K_obs)):
        print(
            f"k={k:02d} | face={int(obs_face_id[k])} | "
            f"verts={obs_face_vertices[k].tolist()} | "
            f"bary={obs_barycentric[k].tolist()} | "
            f"dominant_vertex={int(observed_idx[k])}"
        )

    dense_path = os.path.join(data_dir, "W_gt_dense.npy")
    if os.path.exists(dense_path):
        print("\n========== Dense W_gt Check ==========")
        W_dense = np.load(dense_path).astype(np.float32)
        describe_array("W_gt_dense", W_dense)

        W_shape_ok = W_dense.shape == (K_obs, N)
        W_row_sum = W_dense.sum(axis=1)
        W_row_sum_ok = bool(np.allclose(W_row_sum, 1.0, atol=1e-5))

        # Rebuild sparse-to-dense and compare
        W_rebuild = np.zeros((K_obs, N), dtype=np.float32)
        for k in range(K_obs):
            for j in range(3):
                W_rebuild[k, W_idx[k, j]] += W_val[k, j]

        _, max_diff, mean_diff = check_close("dense W vs sparse rebuild", W_dense, W_rebuild, atol=1e-6, rtol=1e-6)

        print("W_gt_dense shape ok:", W_shape_ok)
        print("W_gt_dense row sum min/max:", float(W_row_sum.min()), float(W_row_sum.max()))
        print("W_gt_dense rows sum to 1:", W_row_sum_ok)
        print("Dense W max diff:", max_diff)
        print("Dense W mean diff:", mean_diff)

    print("\n========== Recompute Surface Observation ==========")

    X_tri = X0[obs_face_vertices]  # [K,3,3]
    X_surface = np.sum(obs_barycentric[:, :, None] * X_tri, axis=1).astype(np.float32)

    U_tri_seq = U_seq[:, obs_face_vertices, :]  # [T,K,3,3]
    U_surface_seq = np.sum(obs_barycentric[None, :, :, None] * U_tri_seq, axis=2).astype(np.float32)

    uv0_re, Xcam0_re = perspective_project(X_surface, Kmat, R, t)
    uvt_re, Xcamt_re = perspective_project(X_surface[None, :, :] + U_surface_seq, Kmat, R, t)

    y_pixel_re = uvt_re - uv0_re[None, :, :]
    fx = float(Kmat[0, 0])
    y_clean_re = y_pixel_re / fx

    check_close("track_uv0 recompute", track_uv0, uv0_re, atol=1e-4, rtol=1e-5)
    check_close("track_uv_seq recompute", track_uv_seq, uvt_re, atol=1e-4, rtol=1e-5)
    check_close("y_pixel recompute", y_pixel, y_pixel_re, atol=1e-4, rtol=1e-5)
    check_close("y_clean recompute", y_clean, y_clean_re, atol=1e-6, rtol=1e-5)

    y_noise = y - y_clean
    describe_array("y_noise = y - y_clean", y_noise)

    print("\n========== Camera Depth Check ==========")
    print("Xcam0 depth min/max:", float(Xcam0_re[:, 2].min()), float(Xcam0_re[:, 2].max()))
    print("Xcamt depth min/max:", float(Xcamt_re[..., 2].min()), float(Xcamt_re[..., 2].max()))
    print("all reference surface points in front:", bool(np.all(Xcam0_re[:, 2] > 1e-6)))
    print("all deformed surface points in front:", bool(np.all(Xcamt_re[..., 2] > 1e-6)))

    print("\n========== Displacement Magnitude ==========")
    U_mag = np.linalg.norm(U, axis=1)
    U_seq_mag = np.linalg.norm(U_seq, axis=2)
    U_surface_mag = np.linalg.norm(U_surface_seq, axis=2)

    print("U max disp:", float(U_mag.max()))
    print("U mean disp:", float(U_mag.mean()))
    print("U median disp:", float(np.median(U_mag)))
    print("U_seq max disp:", float(U_seq_mag.max()))
    print("U_seq mean disp:", float(U_seq_mag.mean()))
    print("U_surface_seq max disp:", float(U_surface_mag.max()))
    print("U_surface_seq mean disp:", float(U_surface_mag.mean()))

    print("\n========== y Signal Strength ==========")
    print("y max abs:", float(np.max(np.abs(y))))
    print("y mean abs:", float(np.mean(np.abs(y))))
    print("y_clean max abs:", float(np.max(np.abs(y_clean))))
    print("y_pixel max abs:", float(np.max(np.abs(y_pixel))))
    print("y_pixel mean abs:", float(np.mean(np.abs(y_pixel))))

    if np.max(np.abs(y)) < 1e-4:
        print("WARNING: y is very small.")
    if np.mean(np.abs(y)) < 1e-5:
        print("WARNING: mean abs y is extremely small.")

    print("\n========== Input-output Relation Check ==========")
    y_frame_amp = np.mean(np.abs(y), axis=(1, 2))
    U_frame_amp = np.mean(np.abs(U_seq), axis=(1, 2))
    U_surface_frame_amp = np.mean(np.abs(U_surface_seq), axis=(1, 2))

    corr_full = safe_corr(y_frame_amp, U_frame_amp)
    corr_surface = safe_corr(y_frame_amp, U_surface_frame_amp)

    print("corr(mean_abs_y_per_frame, mean_abs_U_full_per_frame):", corr_full)
    print("corr(mean_abs_y_per_frame, mean_abs_U_surface_per_frame):", corr_surface)
    print("y_frame_amp min/max:", float(y_frame_amp.min()), float(y_frame_amp.max()))
    print("U_frame_amp min/max:", float(U_frame_amp.min()), float(U_frame_amp.max()))
    print("U_surface_frame_amp min/max:", float(U_surface_frame_amp.min()), float(U_surface_frame_amp.max()))

    print("\n========== Camera Check ==========")
    print("camera_K:\n", Kmat)
    print("camera_R:\n", R)
    print("camera_t:\n", t)
    print("camera_eye:", camera_eye)
    print("camera_target:", camera_target)
    print("det(camera_R):", float(np.linalg.det(R)))
    print("R @ R.T close to I:", bool(np.allclose(R @ R.T, np.eye(3), atol=1e-5)))

    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        print("\n========== Meta ==========")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        keys = [
            "split",
            "sample_id",
            "seed",
            "observation_mode",
            "correspondence_type",
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
            "X0_shape",
            "faces_shape",
            "U_seq_shape",
            "y_shape",
        ]

        for k in keys:
            print(f"{k}: {meta.get(k)}")

    print("\n========== Overall Result ==========")

    critical_ok = all([
        shape_ok,
        face_id_valid,
        face_vertices_valid,
        bary_sum_ok,
        bary_nonneg_ok,
        bary_leq_ok,
        face_vertices_match,
        W_idx_match,
        W_val_match,
        observed_idx_ok,
        bool(np.all(Xcam0_re[:, 2] > 1e-6)),
        bool(np.all(Xcamt_re[..., 2] > 1e-6)),
    ])

    print("CRITICAL CHECK:", "PASSED" if critical_ok else "FAILED")

    if critical_ok:
        print("Data looks structurally valid for surface-correspondence training.")
    else:
        print("Some critical checks failed. Inspect the messages above.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    check_one_sample(parse_args())
