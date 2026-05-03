

from pathlib import Path

path = Path("src/eval.py")
if not path.exists():
    raise FileNotFoundError("Cannot find src/eval.py. Run this script from the project root.")

s = path.read_text(encoding="utf-8")
backup = path.with_suffix(".py.bak")
backup.write_text(s, encoding="utf-8")
print(f"Backup saved to: {backup}")


def must_replace(old: str, new: str, name: str, count: int = -1):
    global s
    n = s.count(old)
    if n == 0:
        raise RuntimeError(f"Patch failed: block not found for {name}")
    if count > 0 and n < count:
        raise RuntimeError(f"Patch failed: expected at least {count} blocks for {name}, found {n}")
    s = s.replace(old, new, count if count > 0 else n)
    print(f"Patched {name}: {min(n, count) if count > 0 else n} occurrence(s)")

# ---------------------------------------------------------------------
# 1) Add calibration helper
# ---------------------------------------------------------------------
must_replace(
'''def gaussian_nll_numpy(target, mu, std, eps=1e-8):
    var = std ** 2 + eps
    return float(np.mean(0.5 * (np.log(var) + (target - mu) ** 2 / var)))


def confidence_filtering_metrics''',
'''def gaussian_nll_numpy(target, mu, std, eps=1e-8):
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


def confidence_filtering_metrics''',
"calibration helper",
count=1,
)

# ---------------------------------------------------------------------
# 2) Extend predict_sequence_real_like signature
# ---------------------------------------------------------------------
must_replace(
'''    mc_samples=8,
    normalize_y_for_projection=True,
    save_W=True,
):''',
'''    mc_samples=8,
    normalize_y_for_projection=True,
    save_W=True,
    return_samples=False,
):''',
"predict signature",
count=1,
)

# ---------------------------------------------------------------------
# 3) Return posterior samples and variance decomposition
# ---------------------------------------------------------------------
must_replace(
'''    if save_W:
        W_seq = np.mean(np.stack(all_W_samples, axis=0), axis=0).astype(np.float32)
    else:
        W_seq = None

    return U_pred_seq, U_std_seq, y_hat_seq, W_seq, W_entropy_seq
''',
'''    if save_W:
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
''',
"posterior extra return",
count=1,
)

# ---------------------------------------------------------------------
# 4) Unpack new return and pass return_samples
# ---------------------------------------------------------------------
must_replace(
'''    U_pred_seq, U_std_seq, y_hat_seq, W_seq, W_entropy_seq = predict_sequence_real_like(
        model=model,''',
'''    U_pred_seq, U_std_seq, y_hat_seq, W_seq, W_entropy_seq, posterior_extra = predict_sequence_real_like(
        model=model,''',
"prediction unpack",
count=1,
)

must_replace(
'''        normalize_y_for_projection=not args.no_normalize_y_for_projection,
        save_W=args.save_W,
    )''',
'''        normalize_y_for_projection=not args.no_normalize_y_for_projection,
        save_W=args.save_W,
        return_samples=args.save_samples,
    )''',
"prediction call args",
count=1,
)

# ---------------------------------------------------------------------
# 5) Add calibration metrics inside GT block
# ---------------------------------------------------------------------
must_replace(
'''            "coverage_3std_vertex": coverage_rate(vertex_error, vertex_std, 3.0),
            "error_std_corr_vertex": safe_corr(vertex_error, vertex_std),''',
'''            "coverage_3std_vertex": coverage_rate(vertex_error, vertex_std, 3.0),
            "calibration_scale_component": rms_calibration_scale(abs_error_seq, U_std_seq),
            "calibration_scale_vertex": rms_calibration_scale(vertex_error, vertex_std),
            "error_std_corr_vertex": safe_corr(vertex_error, vertex_std),''',
"calibration metrics",
count=1,
)

must_replace(
'''        metrics.update(confidence_filtering_metrics(vertex_error, vertex_std, args.keep_rates))
    else:''',
'''        # Calibrated coverage. This only reports calibration quality; it does not change prediction.
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
    else:''',
"calibrated coverage block",
count=1,
)

# ---------------------------------------------------------------------
# 6) Save samples and variance components
# ---------------------------------------------------------------------
must_replace(
'''        np.save(os.path.join(out_dir, "W_entropy_seq.npy"), W_entropy_seq.astype(np.float32))

        if W_seq is not None:''',
'''        np.save(os.path.join(out_dir, "W_entropy_seq.npy"), W_entropy_seq.astype(np.float32))

        if posterior_extra is not None:
            np.save(os.path.join(out_dir, "U_mu_samples.npy"), posterior_extra["U_mu_samples"].astype(np.float32))
            np.save(os.path.join(out_dir, "U_epistemic_std_seq.npy"), posterior_extra["U_epistemic_std_seq"].astype(np.float32))
            np.save(os.path.join(out_dir, "U_aleatoric_std_seq.npy"), posterior_extra["U_aleatoric_std_seq"].astype(np.float32))

        if W_seq is not None:''',
"save posterior samples",
count=1,
)

# ---------------------------------------------------------------------
# 7) Add summary keys
# ---------------------------------------------------------------------
must_replace(
'''        "coverage_2std_component",
        "mean_vertex_std",''',
'''        "coverage_2std_component",
        "coverage_2std_component_cal",
        "calibration_scale_component",
        "calibration_scale_vertex",
        "mean_vertex_std",
        "mean_epistemic_std",
        "mean_aleatoric_std",''',
"summary keys",
count=1,
)

# ---------------------------------------------------------------------
# 8) Add CLI option and print
# ---------------------------------------------------------------------
must_replace(
'''    parser.add_argument("--save_W", action="store_true", help="Save W_pred_seq.npy. Can be large but ok for this dataset.")
    parser.add_argument("--no_normalize_y_for_projection", action="store_true")''',
'''    parser.add_argument("--save_W", action="store_true", help="Save W_pred_seq.npy. Can be large but ok for this dataset.")
    parser.add_argument("--save_samples", action="store_true", help="Save posterior U samples and epistemic/aleatoric std maps.")
    parser.add_argument("--no_normalize_y_for_projection", action="store_true")''',
"cli save_samples",
count=1,
)

must_replace(
'''    print("save_W:", args.save_W)''',
'''    print("save_W:", args.save_W)
    print("save_samples:", args.save_samples)''',
"print save_samples",
count=1,
)

path.write_text(s, encoding="utf-8")
print("\nDone. Patched src/eval.py successfully.")
print("Run syntax check: python -m py_compile src/eval.py")
