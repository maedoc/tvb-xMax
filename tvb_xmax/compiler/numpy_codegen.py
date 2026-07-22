"""Portable NumPy/Autograd surrogate compiler."""

from __future__ import annotations

import time

import autograd.numpy as anp
from autograd import grad
import numpy as np

from ..ir import CompiledArtifact
from ..surrogates import get_surrogate


def _trunk(x, params):
    for weight, bias in params:
        x = anp.tanh(anp.dot(x, weight) + bias)
    return x


def _head(h, params):
    weight, bias = params[0]
    return anp.dot(h, weight) + bias


def rebuild_apply_fns(trunk_params, head_params):
    """Rebuild pickle-safe NumPy apply callables from weight arrays."""
    def trunk_apply(x):
        return np.asarray(_trunk(np.asarray(x), trunk_params))
    def head_apply(h):
        return np.asarray(_head(np.asarray(h), head_params))
    def apply(u, theta):
        return head_apply(trunk_apply(np.concatenate([np.asarray(u), np.asarray(theta)])))
    return apply, trunk_apply, head_apply


def _init(nlat, d_param, d_feat, hidden, seed):
    rng = np.random.RandomState(seed)
    def layer(left, right):
        return (rng.randn(left, right).astype("f") * np.sqrt(2 / (left + right)),
                np.zeros(right, dtype="f"))
    return [layer(nlat + d_param, hidden), layer(hidden, hidden)], [layer(hidden, d_feat)]


def train_surrogate(sim_pairs, nlat: int, d_feat: int, hidden: int = 128,
                    niter: int = 2000, lr: float = 3e-4, seed: int = 0):
    """Train the same two-layer tanh trunk + linear head with Autograd Adam."""
    U, theta, xf = (np.asarray(x, dtype="f") for x in sim_pairs)
    trunk, head = _init(nlat, theta.shape[-1], d_feat, hidden, seed)
    params = {"trunk": trunk, "head": head}
    def loss(p):
        x = anp.concatenate([U, theta], axis=1)
        pred = _head(_trunk(x, p["trunk"]), p["head"])
        return anp.mean((pred - xf) ** 2)
    derivative = grad(loss)
    first = {k: [(anp.zeros_like(w), anp.zeros_like(b)) for w, b in v] for k, v in params.items()}
    second = {k: [(anp.zeros_like(w), anp.zeros_like(b)) for w, b in v] for k, v in params.items()}
    for step in range(niter):
        grads = derivative(params)
        updated = {}
        for group in params:
            values, moments, variances = [], [], []
            for (w, b), (mw, mb), (vw, vb), (gw, gb) in zip(params[group], first[group], second[group], grads[group]):
                mw, mb = .9 * mw + .1 * gw, .9 * mb + .1 * gb
                vw, vb = .999 * vw + .001 * gw ** 2, .999 * vb + .001 * gb ** 2
                values.append((w - lr * (mw / (1 - .9 ** (step + 1))) / (anp.sqrt(vw / (1 - .999 ** (step + 1))) + 1e-8),
                               b - lr * (mb / (1 - .9 ** (step + 1))) / (anp.sqrt(vb / (1 - .999 ** (step + 1))) + 1e-8)))
                moments.append((mw, mb)); variances.append((vw, vb))
            updated[group], first[group], second[group] = values, moments, variances
        params = updated
    return params["trunk"], params["head"], float(loss(params))


def compile_artifact(model, feature, sim_pairs, nlat, d_feat, **kw):
    """Compile a portable surrogate artifact."""
    started = time.perf_counter()
    trunk, head, mse = train_surrogate(sim_pairs, nlat, d_feat, **kw)
    apply, trunk_apply, head_apply = rebuild_apply_fns(trunk, head)
    target = get_surrogate(model)
    return CompiledArtifact(model, feature, nlat, apply, backend="numpy",
                            param_names=target.param_names, param_bounds=target.param_bounds,
                            surrogate_mse=mse, train_sim_budget=len(sim_pairs[0]),
                            compile_seconds=time.perf_counter() - started,
                            trunk_apply=trunk_apply, head_apply=head_apply,
                            trunk_params=trunk, head_params=head)
