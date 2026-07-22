"""Real vbjax-based SDE simulation callables for benchmark_speedup."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import vbjax as vb

from ..surrogates import get_surrogate


def _make_hopf_sde(n_regions: int, dt: float = 0.1):
    """Build Hopf oscillator SDE (step, loop) pair.

    Drift::
        dx = (eta - r^2) * x - omega * y + k * C @ x
        dy = (eta - r^2) * y + omega * x + k * C @ y
    """

    def drift(state, p):
        C, eta_arr, omega_arr, k, _ = p
        nr = eta_arr.shape[0]
        x = state[:nr]
        y = state[nr:]
        r2 = x**2 + y**2
        dx = (eta_arr - r2) * x - omega_arr * y + k * (C @ x)
        dy = (eta_arr - r2) * y + omega_arr * x + k * (C @ y)
        return jnp.concatenate([dx, dy])

    def gfun(state, p):
        return p[-1]

    return vb.make_sde(dt, drift, gfun)


def _make_hopf_sim_fn(
    crosscoder: Any,
    nlat: int,
    parc: str,
    param_bounds: tuple,
    n_steps: int = 200,
    n_transient: int = 50,
    dt: float = 0.1,
) -> Callable:
    """Return a Hopf sim_fn(u, theta) -> features.

    The returned function closes over the cross-coder, SDE loop, and
    parameter bounds so it is self-contained for benchmark_speedup.
    """
    # Determine n_regions from the decoder output dimension
    iparc = crosscoder.parcs.index(parc)
    _, (_, b_dec) = crosscoder._get_arch(nlat).wbs[iparc]
    n_triu = int(b_dec.shape[0])
    n_regions = int(math.ceil((1 + math.sqrt(1 + 8 * n_triu)) / 2))

    p_lo = jnp.array([b[0] for b in param_bounds])
    p_hi = jnp.array([b[1] for b in param_bounds])

    _, sde_loop = _make_hopf_sde(n_regions, dt)

    prng_key = jax.random.PRNGKey(42)

    def sim_fn(u: jax.Array, theta: jax.Array) -> jax.Array:
        C = crosscoder.decode_conn(nlat, parc, u[None, :],
                                   clip_positive=True)[0]
        raw = p_lo + theta * (p_hi - p_lo)
        k_val, D_val, eta_val, omega_val = raw

        eta_arr = jnp.full(n_regions, eta_val)
        omega_arr = jnp.full(n_regions, omega_val)

        state0 = jnp.concatenate([jnp.zeros(n_regions),
                                  jnp.ones(n_regions)])

        nonlocal prng_key
        prng_key, subkey = jax.random.split(prng_key)
        zs = jax.random.normal(subkey, (n_steps, 2 * n_regions))

        states = sde_loop(state0, zs,
                          (C, eta_arr, omega_arr, k_val, D_val))
        x_activity = states[n_transient:, :n_regions]
        return jnp.var(x_activity, axis=0)

    return sim_fn


def make_real_sim_fn(
    model: str,
    crosscoder: Any,
    nlat: int,
    parc: Optional[str] = None,
    n_steps: int = 200,
    n_transient: int = 50,
    dt: float = 0.1,
) -> Callable:
    """Build a real vbjax SDE simulation callable for benchmark_speedup.

    Args:
        model: surrogate model name (e.g. 'hopf').
        crosscoder: trained CrossCoder to decode u -> connectivity.
        nlat: latent dimension matching the cross-coder architecture.
        parc: parcellation name (defaults to first available view).
        n_steps: number of integration steps.
        n_transient: steps to discard as transient.
        dt: integration time step.

    Returns:
        sim_fn(u, theta) -> features array.

    Raises:
        NotImplementedError if model has no real sim implementation.
    """
    if parc is None:
        parc = crosscoder.parcs[0]

    surr = get_surrogate(model)
    param_bounds = surr.param_bounds

    if model == "hopf":
        return _make_hopf_sim_fn(
            crosscoder, nlat, parc, param_bounds,
            n_steps=n_steps, n_transient=n_transient, dt=dt,
        )

    raise NotImplementedError(
        f"real sim for model {model!r} not yet implemented"
    )
