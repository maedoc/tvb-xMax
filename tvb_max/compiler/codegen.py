"""Codegen: the neural surrogate that *replaces* the simulation.

This is the heart of the "compiler".  A :class:`SurrogateNet` learns the
map ``(u, theta) -> features`` directly from a one-time simulation budget.
Once trained, evaluating it is a single MLP forward pass (~ms) instead of
integrating an SDE for seconds.  The trained net is the "object code"
stored in a :class:`CompiledArtifact`.

Architecture: a shared trunk over ``[u; theta]`` with a feature-specific
head.  This lets one trunk serve many feature heads (the "feature swap").
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp

from ..ir import CompiledArtifact
from ..surrogates import get_surrogate


def _mlp(x, params):
    """Apply a 3-layer MLP with tanh activations."""
    for w, b in params:
        x = jnp.tanh(x @ w + b)
    return x


def init_mlp(in_dim: int, hidden: int, out_dim: int, key) -> Tuple[list, list]:
    """Xavier-init a 3-layer MLP; returns (params, shapes)."""
    k1, k2, k3 = jax.random.split(key, 3)
    params = [
        (jax.nn.initializers.glorot_uniform()(k1, (in_dim, hidden)),
         jnp.zeros(hidden)),
        (jax.nn.initializers.glorot_uniform()(k2, (hidden, hidden)),
         jnp.zeros(hidden)),
        (jax.nn.initializers.glorot_uniform()(k3, (hidden, out_dim)),
         jnp.zeros(out_dim)),
    ]
    return params


def make_surrogate_apply(nlat: int, d_param: int, d_feat: int,
                          hidden: int = 128, key=jax.random.PRNGKey(0)):
    """Build a JIT-able surrogate forward function and its params.

    Returns:
        (apply_fn, params) where apply_fn(u, theta) -> features.
        ``u`` is (nlat,), ``theta`` is (d_param,); output is (d_feat,).
    """
    params = init_mlp(nlat + d_param, hidden, d_feat, key)

    @jax.jit
    def apply_fn(u, theta, params=params):
        x = jnp.concatenate([u, theta])
        return _mlp(x, params)

    return apply_fn, params


def train_surrogate(model: str, feature: str, sim_pairs, nlat: int,
                    d_feat: int, hidden: int = 128, niter: int = 2000,
                    lr: float = 3e-4, key=jax.random.PRNGKey(0)):
    """Train a surrogate on ``(u, theta, xf)`` simulation pairs.

    Args:
        sim_pairs: tuple ``(U, Theta, XF)`` of arrays from the one-time
            simulation budget (produced by apvbt's ``sample_model`` or
            ``sample_subj_model``).
        nlat: latent dimension the cross-coder was trained at.
        d_feat: feature dimension (depends on feature extractor).

    Returns:
        (apply_fn, params, final_mse).
    """
    U, Theta, XF = sim_pairs
    U, Theta, XF = jnp.asarray(U), jnp.asarray(Theta), jnp.asarray(XF)
    d_param = Theta.shape[-1]
    apply_fn, params = make_surrogate_apply(nlat, d_param, d_feat, hidden, key)

    @jax.jit
    def loss(params, U, Theta, XF):
        def one(u, t, x):
            return jnp.mean((_mlp(jnp.concatenate([u, t]), params) - x) ** 2)
        return jnp.mean(jax.vmap(one)(U, Theta, XF))

    grad = jax.jit(jax.grad(loss))
    opt_init, opt_update, get_params = _adam(lr)
    state = opt_init(params)
    for i in range(niter):
        g = grad(get_params(state), U, Theta, XF)
        state = opt_update(i, g, state)
    params = get_params(state)
    final_mse = float(loss(params, U, Theta, XF))

    @functools.partial(jax.jit, static_argnums=())
    def apply_trained(u, theta):
        return _mlp(jnp.concatenate([u, theta]), params)

    return apply_trained, params, final_mse


def _adam(lr):
    """Minimal Adam optimizer (avoids importing jax.example_libraries)."""
    b1, b2, eps = 0.9, 0.999, 1e-8

    def init(params):
        return [(p, jnp.zeros_like(p), jnp.zeros_like(p)) for (w, b) in params
                for p in (w, b)]

    def update(i, grads, state):
        out = []
        for (p, m, v), g in zip(state, grads):
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mh = m / (1 - b1 ** (i + 1))
            vh = v / (1 - b2 ** (i + 1))
            out.append((p - lr * mh / (jnp.sqrt(vh) + eps), m, v))
        return out

    def get_params(state):
        return [(state[2 * k][0], state[2 * k + 1][0])
                for k in range(len(state) // 2)]

    return init, update, get_params


def compile_artifact(model: str, feature: str, sim_pairs, nlat: int,
                     d_feat: int, posterior_sample=None, **kw) -> CompiledArtifact:
    """Train a surrogate and wrap it as a :class:`CompiledArtifact`."""
    import time
    t0 = time.time()
    apply_fn, params, mse = train_surrogate(
        model, feature, sim_pairs, nlat, d_feat, **kw)
    surr = get_surrogate(model)
    return CompiledArtifact(
        model=model,
        feature=feature,
        nlat=nlat,
        surrogate_apply=apply_fn,
        posterior_sample=posterior_sample,
        param_names=surr.param_names,
        param_bounds=surr.param_bounds,
        surrogate_mse=mse,
        train_sim_budget=int(sim_pairs[0].shape[0]),
        compile_seconds=time.time() - t0,
    )
