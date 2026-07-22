"""Vectorize: GPU batch evaluation for "maxest speedup".

Takes a :class:`CompiledArtifact` and lifts its single-sample forward pass
to batched evaluation across devices (``pmap``) and within device
(``vmap``).  This is where the ~10^3-10^4x amortized speedup is realized:
10^4 posterior draws cost one batched forward pass instead of 10^4 SDE
integrations.
"""

from __future__ import annotations

import functools
import time

import jax
import jax.numpy as jnp

from ..ir import CompiledArtifact


def batched_features(artifact: CompiledArtifact, U: jax.Array,
                     Theta: jax.Array) -> jax.Array:
    """Evaluate the surrogate over a batch of (u, theta) pairs.

    Args:
        U: (B, nlat) latent codes.
        Theta: (B, d_param) parameter vectors.

    Returns:
        (B, d_feat) feature predictions.
    """
    return jax.jit(jax.vmap(artifact.surrogate_apply))(U, Theta)


def batched_posterior(artifact: CompiledArtifact, xf_obs: jax.Array,
                      n_samples: int) -> jax.Array:
    """Draw ``n_samples`` posterior thetas for each of B observations.

    Args:
        xf_obs: (B, d_feat) observed features.
        n_samples: posterior draws per observation.

    Returns:
        (n_samples, B, d_param) posterior draws.
    """
    if artifact.posterior_sample is None:
        raise RuntimeError("artifact has no trained posterior; target='features' only")
    f = artifact.posterior_sample
    # The in-tree posterior exposes the same batched sampling shape.
    return f(xf_obs, n_samples)


def sharded_features(artifact: CompiledArtifact, U: jax.Array,
                      Theta: jax.Array) -> jax.Array:
    """Multi-GPU evaluation via ``pmap`` across the leading axis.

    Reshapes (B, ...) to (n_devices, B//n_devices, ...) and maps the
    surrogate, then flattens back.  Falls back to ``vmap`` on a single
    device.
    """
    n_dev = jax.device_count()
    if n_dev == 1:
        return batched_features(artifact, U, Theta)
    lead = U.shape[0]
    assert lead % n_dev == 0, f"batch {lead} not divisible by {n_dev} devices"
    U_ = U.reshape((n_dev, -1) + U.shape[1:])
    T_ = Theta.reshape((n_dev, -1) + Theta.shape[1:])
    out = jax.pmap(jax.vmap(artifact.surrogate_apply))(U_, T_)
    return out.reshape((lead,) + out.shape[2:])


def benchmark_speedup(artifact: CompiledArtifact, sim_fn, U: jax.Array,
                        Theta: jax.Array, n_warmup: int = 3) -> dict:
    """Measure t_sim / t_surrogate for the same (U, Theta) batch.

    Args:
        sim_fn: the real simulation callable ``sim(u, theta) -> features``
            (e.g. wrapped apvbt ``sample_model`` output).
    Returns:
        dict with ``t_sim``, ``t_surrogate``, ``speedup``, ``batch``.
    """
    B = U.shape[0]
    # warmup surrogate
    for _ in range(n_warmup):
        batched_features(artifact, U, Theta).block_until_ready()
    t0 = time.perf_counter()
    batched_features(artifact, U, Theta).block_until_ready()
    t_surr = time.perf_counter() - t0
    # time the real sim on a small sub-batch then extrapolate
    sub = min(B, 32)
    t0 = time.perf_counter()
    for i in range(sub):
        sim_fn(U[i], Theta[i]).block_until_ready()
    t_sim = (time.perf_counter() - t0) * (B / sub)
    return {
        "t_sim": t_sim,
        "t_surrogate": t_surr,
        "speedup": t_sim / t_surr,
        "batch": B,
    }


def benchmark_single_feature_eval(artifact: CompiledArtifact, sim_fn,
                                  u: jax.Array, theta: jax.Array,
                                  n_warmup: int = 3,
                                  n_repeat: int = 5) -> dict:
    """Measure warmed single-call SDE and surrogate feature latency.

    Both paths are synchronized with ``block_until_ready`` so asynchronous
    JAX dispatch is not reported as computation time. The median over repeats
    reduces timing noise while retaining one-connectome, one-parameter-input
    semantics.
    """
    for _ in range(n_warmup):
        artifact(u, theta).block_until_ready()
        sim_fn(u, theta).block_until_ready()

    sim_times = []
    surrogate_times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        sim_fn(u, theta).block_until_ready()
        sim_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        artifact(u, theta).block_until_ready()
        surrogate_times.append(time.perf_counter() - t0)

    t_sim = float(jnp.median(jnp.asarray(sim_times)))
    t_surrogate = float(jnp.median(jnp.asarray(surrogate_times)))
    return {
        "t_sim": t_sim,
        "t_surrogate": t_surrogate,
        "speedup": t_sim / t_surrogate,
        "repeats": n_repeat,
    }
