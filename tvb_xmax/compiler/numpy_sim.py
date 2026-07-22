"""Portable NumPy/SciPy simulation primitives adapted from ``tvbl.core``."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..ir import SimBudget


def delayed_coupling(step, buffer, weights: sparse.csr_matrix, delays, horizon):
    """Sparse delayed coupling with linear interpolation, as used by TVBL."""
    delays = np.asarray(delays, dtype=int)
    if delays.shape != (weights.nnz,):
        raise ValueError("delays must have one entry per sparse connection")
    left = buffer[weights.indices, (step - delays) % horizon] * weights.data[:, None]
    right = buffer[weights.indices, (step - delays - 1) % horizon] * weights.data[:, None]
    return np.stack([np.add.reduceat(left, weights.indptr[:-1], axis=0),
                     np.add.reduceat(right, weights.indptr[:-1], axis=0)])


def run_heun(weights, delays, dfun, z_scale, horizon, *, seed=43, n_items=1,
             n_steps=1000, dt=.1, skip=1):
    """Run a TVBL-style stochastic delayed system with Heun integration."""
    weights = sparse.csr_matrix(weights)
    n_nodes, n_state = weights.shape[0], int(dfun.n_state)
    if np.max(delays, initial=0) >= horizon - 1:
        raise ValueError("horizon must exceed the largest delay by two steps")
    rng = np.random.default_rng(seed)
    buffer = np.zeros((n_nodes, horizon, n_items), dtype=np.float32)
    state = np.zeros((n_state, n_nodes, n_items), dtype=np.float32)
    trace = []
    scale = np.asarray(z_scale, dtype=np.float32).reshape(n_state, 1, 1)
    for step in range(n_steps):
        coupling = delayed_coupling(step, buffer, weights, delays, horizon)
        noise = rng.normal(size=state.shape).astype(np.float32) * scale
        first = dfun(state, coupling[0])
        second = dfun(state + dt * first + noise, coupling[1])
        state = state + dt * .5 * (first + second) + noise
        buffer[:, step % horizon] = state[0]
        if step % skip == 0:
            trace.append(state.copy())
    return np.asarray(trace)


class HopfDFun:
    """Two-state Hopf normal form matching tvb-xMax's portable feature path."""

    n_state = 2

    def __init__(self, eta, omega, coupling):
        self.eta, self.omega, self.coupling = eta, omega, coupling

    def __call__(self, state, coupling):
        x, y = state
        radius2 = x ** 2 + y ** 2
        return np.asarray([(self.eta - radius2) * x - self.omega * y + self.coupling * coupling,
                           (self.eta - radius2) * y + self.omega * x + self.coupling * coupling])


def hopf_features(connectivity, theta, *, n_steps=1000, dt=.1, seed=43):
    """Simulate Hopf activity and return per-region temporal variance."""
    connectivity = np.asarray(connectivity, dtype=np.float32)
    k, noise, eta, omega = np.asarray(theta, dtype=np.float32)
    simulator = HopfDFun(eta, omega, k)
    trace = run_heun(connectivity, np.zeros(sparse.csr_matrix(connectivity).nnz, dtype=int),
                     simulator, [noise, noise], horizon=2, seed=seed,
                     n_steps=n_steps, dt=dt)
    return np.var(trace[len(trace) // 10:, 0, :, 0], axis=0)


def hopf_budget(U, theta, decode_connectivity, param_bounds, *, n_steps=1000, dt=.1):
    """Generate a portable ``SimBudget`` from IR samples and a NumPy decoder."""
    U, theta = np.asarray(U), np.asarray(theta)
    bounds = np.asarray(param_bounds, dtype=np.float32)
    raw = bounds[:, 0] + theta * (bounds[:, 1] - bounds[:, 0])
    features = np.stack([hopf_features(decode_connectivity(u), t, n_steps=n_steps,
                                       dt=dt, seed=i + 43)
                         for i, (u, t) in enumerate(zip(U, raw))])
    return SimBudget(U=U, Theta=theta, XF=features, model="hopf", feature="var", nlat=U.shape[-1])
