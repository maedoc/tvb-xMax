"""Generate reproducible FC and metastability README figures.

This is a synthetic, seeded Hopf validation experiment.  It checks that a
surrogate trained on SDDE-derived FC + metastability features interpolates a
global-coupling sweep; it is not a biological validation dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvb_xmax.compiler import numpy_codegen, numpy_sim, numpy_vectorize


SEED = 20260722
N_REGIONS = 24
NLAT = 16
K_VALUES = np.linspace(.05, .45, 16, dtype=np.float32)
META_WEIGHT = 32
PARAMS = {"D": .01, "eta": .1, "omega": 1.0}
OUT = Path("docs/figures")


def decode_factory(rng):
    """Create a fixed latent-to-connectome decoder for this synthetic study."""
    rows, cols = np.triu_indices(N_REGIONS, 1)
    weights = rng.normal(0, .22, (NLAT, len(rows))).astype("f")
    bias = rng.normal(-3.8, .25, len(rows)).astype("f")

    def decode(u):
        values = 1 / (1 + np.exp(-(np.asarray(u) @ weights + bias)))
        matrix = np.zeros((N_REGIONS, N_REGIONS), dtype=np.float32)
        matrix[rows, cols] = values
        return matrix + matrix.T

    return decode, rows, cols


def theta_for_k(k):
    """Return normalized Hopf parameters matching the registered bounds."""
    return np.asarray([k, PARAMS["D"], (PARAMS["eta"] + 2) / 4,
                       (PARAMS["omega"] - .9 * np.pi) / (.2 * np.pi)], dtype=np.float32)


def raw_theta(theta):
    return np.asarray([theta[0], theta[1], theta[2] * 4 - 2,
                       .9 * np.pi + theta[3] * .2 * np.pi], dtype=np.float32)


def fc_and_metastability(connectivity, theta, n_steps, seed):
    """Extract FC and phase-order metastability from one SDDE simulation."""
    raw = raw_theta(theta)
    dynamics = numpy_sim.HopfDFun(raw[2], raw[3], raw[0])
    trace = numpy_sim.run_heun(connectivity, np.zeros(np.count_nonzero(connectivity), dtype=int),
                               dynamics, [raw[1], raw[1]], horizon=2, seed=seed,
                               n_steps=n_steps, dt=.1)
    trace = trace[n_steps // 5:]
    activity = trace[:, 0, :, 0]
    fc = np.nan_to_num(np.corrcoef(activity.T), nan=0.0)
    phase = np.arctan2(trace[:, 1, :, 0], trace[:, 0, :, 0])
    order = np.abs(np.mean(np.exp(1j * phase), axis=1))
    return fc.astype(np.float32), float(np.std(order))


def encode_feature(fc, metastability, rows, cols):
    # Repeat the scalar so its loss contribution is comparable to the 276 FC
    # entries; otherwise a joint MSE feature head learns FC while ignoring it.
    return np.concatenate([fc[rows, cols], np.full(META_WEIGHT, metastability)]).astype(np.float32)


def decode_feature(feature, rows, cols):
    fc = np.eye(N_REGIONS, dtype=np.float32)
    n_fc = len(rows)
    fc[rows, cols] = feature[:n_fc]
    fc[cols, rows] = feature[:n_fc]
    return np.clip(fc, -1, 1), float(np.mean(feature[n_fc:]))


def median_time(fn, repeats=3):
    times = []
    for _ in range(repeats):
        started = time.perf_counter(); fn(); times.append(time.perf_counter() - started)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="small validation run; do not write figures")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    decode, rows, cols = decode_factory(rng)
    n_latents, n_k, train_steps, niter = (4, 3, 200, 25) if args.smoke else (16, 8, 800, 700)
    eval_steps = 300 if args.smoke else 1_200
    latent_bank = rng.normal(size=(n_latents, NLAT)).astype(np.float32)
    train_k = np.linspace(.05, .45, n_k, dtype=np.float32)
    U = np.repeat(latent_bank, n_k, axis=0)
    theta = np.tile(np.stack([theta_for_k(k) for k in train_k]), (n_latents, 1))
    features = []
    for index, (u, t) in enumerate(zip(U, theta)):
        # Fixed stochastic forcing makes this a learnable conditional feature
        # map; a seed that changes independently per datum is irreducible noise.
        fc, meta = fc_and_metastability(decode(u), t, train_steps, SEED)
        features.append(encode_feature(fc, meta, rows, cols))
    features = np.stack(features)
    assert np.isfinite(features).all() and features.shape[1] == len(rows) + META_WEIGHT
    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), 1e-3)
    standardized_features = (features - feature_mean) / feature_scale
    artifact = numpy_codegen.compile_artifact("hopf", "fc_metastability", (U, theta, standardized_features),
                                              NLAT, features.shape[1], hidden=96, niter=niter, lr=8e-4)

    # Evaluate a denser coupling grid for one latent represented in the
    # simulation budget: an in-distribution interpolation experiment.
    test_u = latent_bank[0]
    sweep_theta = np.stack([theta_for_k(k) for k in K_VALUES])
    truth, truth_meta = [], []
    for index, t in enumerate(sweep_theta):
        fc, meta = fc_and_metastability(decode(test_u), t, eval_steps, SEED)
        truth.append(fc); truth_meta.append(meta)
    truth = np.asarray(truth); truth_meta = np.asarray(truth_meta)
    predicted_features = numpy_vectorize.batched_features(
        artifact, np.repeat(test_u[None], len(K_VALUES), axis=0), sweep_theta)
    predicted_features = predicted_features * feature_scale + feature_mean
    prediction = [decode_feature(x, rows, cols) for x in predicted_features]
    predicted_fc = np.asarray([x[0] for x in prediction]); predicted_meta = np.asarray([x[1] for x in prediction])
    assert np.isfinite(predicted_fc).all() and np.isfinite(predicted_meta).all()

    chosen = len(K_VALUES) // 2
    single_sim = median_time(lambda: fc_and_metastability(decode(test_u), sweep_theta[chosen], eval_steps, SEED))
    single_surrogate = median_time(lambda: artifact(test_u, sweep_theta[chosen]) * feature_scale + feature_mean, repeats=20)
    sweep_sim = median_time(lambda: [fc_and_metastability(decode(test_u), t, eval_steps, SEED)
                                     for i, t in enumerate(sweep_theta)], repeats=1)
    sweep_surrogate = median_time(lambda: numpy_vectorize.batched_features(
        artifact, np.repeat(test_u[None], len(K_VALUES), axis=0), sweep_theta) * feature_scale + feature_mean, repeats=20)
    meta_mae = float(np.mean(np.abs(predicted_meta - truth_meta)))
    fc_mae = float(np.mean(np.abs(predicted_fc[chosen] - truth[chosen])))
    result = {"seed": SEED, "regions": N_REGIONS, "n_train": len(U),
              "train_steps": train_steps, "eval_steps": eval_steps, "niter": niter,
              "stochastic_seed": SEED, "target_standardized": True,
              "metastability_loss_weight": META_WEIGHT, "evaluation": "in-distribution coupling interpolation",
              "k_values": K_VALUES.tolist(), "fc_mae_at_k": fc_mae,
              "metastability_mae": meta_mae, "single_sim_seconds": single_sim,
              "single_surrogate_seconds": single_surrogate,
              "single_speedup": single_sim / single_surrogate,
              "sweep_sim_seconds": sweep_sim, "sweep_surrogate_seconds": sweep_surrogate,
              "sweep_speedup": sweep_sim / sweep_surrogate}
    print(json.dumps(result, indent=2))
    if args.smoke:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "dynamics_feature_benchmark.json", "w") as handle:
        json.dump(result, handle, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    for axis, data, title in zip(axes, [truth[chosen], predicted_fc[chosen]],
                                 ["Ground-truth SDDE", "Compiled surrogate"]):
        image = axis.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_title(title); axis.set_xlabel("Region"); axis.set_ylabel("Region")
    fig.colorbar(image, ax=axes, shrink=.8, label="Functional connectivity")
    fig.suptitle(f"Functional connectivity at k={K_VALUES[chosen]:.2f}  (MAE={fc_mae:.3f})")
    fig.savefig(OUT / "fc_ground_truth_vs_surrogate.png", dpi=180, transparent=False)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
    axis.plot(K_VALUES, truth_meta, "o-", color="#4b7289", label="Ground-truth SDDE")
    axis.plot(K_VALUES, predicted_meta, "s--", color="#c9854c", label="Compiled surrogate")
    axis.set(xlabel="Global coupling k", ylabel="Metastability", title=f"Metastability sweep (MAE={meta_mae:.3f})")
    axis.legend(frameon=False)
    fig.savefig(OUT / "metastability_ground_truth_vs_surrogate.png", dpi=180, transparent=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
