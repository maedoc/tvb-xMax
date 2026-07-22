"""Codegen: the neural surrogate that *replaces* the simulation.

This is the heart of the "compiler".  A surrogate learns the
map ``(u, theta) -> features`` directly from a one-time simulation budget.
Once trained, evaluating it is a single forward pass (~ms) instead of
integrating an SDE for seconds.  The trained net is the "object code"
stored in a :class:`CompiledArtifact`.

Architecture: a shared **trunk** over ``[u; theta]`` (a 2-layer tanh MLP)
followed by a feature-specific **head** (a single linear layer).  The trunk
maps the concatenated latent + parameters to a hidden representation ``h``
of dim ``hidden``; the head maps ``h`` to the feature vector ``xf``.
Splitting the network this way lets one trunk serve many feature heads:
swapping the feature (e.g. ``var`` -> ``fc``) only requires retraining the
cheap head, not the trunk (the "feature swap").
"""

from __future__ import annotations

import functools
from typing import List, Tuple

import jax
import jax.numpy as jnp
import optax

from ..ir import CompiledArtifact
from ..surrogates import get_surrogate


def _trunk_apply(x, trunk_params):
    """Apply the 2-layer tanh trunk: ``[u; theta] -> h``."""
    for w, b in trunk_params:
        x = jnp.tanh(x @ w + b)
    return x


def _head_apply(h, head_params):
    """Apply the 1-layer linear head: ``h -> xf``."""
    w, b = head_params[0]
    return h @ w + b


def init_trunk(in_dim: int, hidden: int, key) -> List[Tuple[jax.Array, jax.Array]]:
    """Xavier-init a 2-layer tanh trunk.

    Layers: ``in_dim -> hidden`` (tanh), ``hidden -> hidden`` (tanh).
    """
    k1, k2 = jax.random.split(key)
    return [
        (jax.nn.initializers.glorot_uniform()(k1, (in_dim, hidden)),
         jnp.zeros(hidden)),
        (jax.nn.initializers.glorot_uniform()(k2, (hidden, hidden)),
         jnp.zeros(hidden)),
    ]


def init_head(hidden: int, out_dim: int, key) -> List[Tuple[jax.Array, jax.Array]]:
    """Xavier-init a 1-layer linear head.

    Layer: ``hidden -> out_dim`` (linear, no activation).
    """
    w = jax.nn.initializers.glorot_uniform()(key, (hidden, out_dim))
    b = jnp.zeros(out_dim)
    return [(w, b)]


def make_trunk_head(nlat: int, d_param: int, d_feat: int,
                    hidden: int = 128, key=jax.random.PRNGKey(0)):
    """Build trunk + head params and their closed-over apply functions.

    Args:
        nlat: latent dimension the cross-coder was trained at.
        d_param: parameter vector dimension.
        d_feat: feature dimension (output of the head).
        hidden: trunk hidden width.

    Returns:
        ``(trunk_params, head_params, trunk_apply_fn, head_apply_fn)`` where
        ``trunk_apply_fn(x) -> h`` and ``head_apply_fn(h) -> xf`` close over
        their respective params.
    """
    k1, k2 = jax.random.split(key)
    trunk_params = init_trunk(nlat + d_param, hidden, k1)
    head_params = init_head(hidden, d_feat, k2)

    def trunk_apply_fn(x):
        return _trunk_apply(x, trunk_params)

    def head_apply_fn(h):
        return _head_apply(h, head_params)

    return trunk_params, head_params, trunk_apply_fn, head_apply_fn


def rebuild_apply_fns(trunk_params, head_params):
    """Rebuild ``surrogate_apply``, ``trunk_apply``, ``head_apply`` from raw params.

    Pickle cannot serialize the closures produced by :func:`make_trunk_head` /
    :func:`train_surrogate` because they close over JAX params.  This helper
    reconstructs equivalent callables from the stored (picklable) param trees,
    so a :class:`CompiledArtifact` can be rehydrated after unpickling.

    Args:
        trunk_params: trunk weight tree (list of ``(w, b)`` tuples).
        head_params: head weight tree (list of one ``(w, b)`` tuple).

    Returns:
        ``(surrogate_apply, trunk_apply, head_apply)`` where
        ``surrogate_apply(u, theta) -> xf``, ``trunk_apply(x) -> h``,
        ``head_apply(h) -> xf``.
    """
    def trunk_apply(x):
        return _trunk_apply(x, trunk_params)

    def head_apply(h):
        return _head_apply(h, head_params)

    def surrogate_apply(u, theta):
        return head_apply(trunk_apply(jnp.concatenate([u, theta])))

    return surrogate_apply, trunk_apply, head_apply


def make_surrogate_apply(nlat: int, d_param: int, d_feat: int,
                          hidden: int = 128, key=jax.random.PRNGKey(0)):
    """Build a JIT-able surrogate forward function and its params.

    Composes ``head(trunk([u; theta]))``.

    Returns:
        ``(apply_fn, (trunk_params, head_params))`` where
        ``apply_fn(u, theta) -> features``.  ``u`` is ``(nlat,)``,
        ``theta`` is ``(d_param,)``; output is ``(d_feat,)``.
    """
    trunk_params, head_params, _, _ = make_trunk_head(
        nlat, d_param, d_feat, hidden, key)

    @jax.jit
    def apply_fn(u, theta):
        h = _trunk_apply(jnp.concatenate([u, theta]), trunk_params)
        return _head_apply(h, head_params)

    return apply_fn, (trunk_params, head_params)


def train_surrogate(model: str, feature: str, sim_pairs, nlat: int,
                    d_feat: int, hidden: int = 128, niter: int = 2000,
                    lr: float = 3e-4, key=jax.random.PRNGKey(0)):
    """Train a trunk + head surrogate on ``(u, theta, xf)`` simulation pairs.

    The trunk and head are trained *jointly*: the loss backprops through
    both, and a single ``optax.adam`` optimizer updates the combined
    parameter tree ``{"trunk": ..., "head": ...}``.

    Args:
        model: literature model name (for registry bookkeeping).
        feature: feature-extractor name (selects the head identity).
        sim_pairs: tuple ``(U, Theta, XF)`` of arrays from the one-time
            simulation budget (produced by apvbt's ``sample_model`` or
            ``sample_subj_model``).
        nlat: latent dimension the cross-coder was trained at.
        d_feat: feature dimension (depends on feature extractor).
        hidden: trunk hidden width.
        niter: number of Adam steps.
        lr: learning rate.

    Returns:
        ``(apply_fn, (trunk_params, head_params), final_mse)`` where
        ``apply_fn(u, theta) -> xf`` composes head over trunk.
    """
    U, Theta, XF = sim_pairs
    U, Theta, XF = jnp.asarray(U), jnp.asarray(Theta), jnp.asarray(XF)
    d_param = Theta.shape[-1]
    trunk_params, head_params, _, _ = make_trunk_head(
        nlat, d_param, d_feat, hidden, key)
    params = {"trunk": trunk_params, "head": head_params}

    @jax.jit
    def loss(params, U, Theta, XF):
        def one(u, t, x):
            h = _trunk_apply(jnp.concatenate([u, t]), params["trunk"])
            pred = _head_apply(h, params["head"])
            return jnp.mean((pred - x) ** 2)
        return jnp.mean(jax.vmap(one)(U, Theta, XF))

    grad = jax.jit(jax.grad(loss))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    for i in range(niter):
        g = grad(params, U, Theta, XF)
        updates, opt_state = optimizer.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
    final_mse = float(loss(params, U, Theta, XF))

    trunk_params = params["trunk"]
    head_params = params["head"]

    @functools.partial(jax.jit, static_argnums=())
    def apply_trained(u, theta):
        h = _trunk_apply(jnp.concatenate([u, theta]), trunk_params)
        return _head_apply(h, head_params)

    return apply_trained, (trunk_params, head_params), final_mse


def compile_artifact(model: str, feature: str, sim_pairs, nlat: int,
                     d_feat: int, posterior_sample=None, **kw) -> CompiledArtifact:
    """Train a trunk+head surrogate and wrap it as a :class:`CompiledArtifact`."""
    import time
    t0 = time.time()
    apply_fn, (trunk_params, head_params), mse = train_surrogate(
        model, feature, sim_pairs, nlat, d_feat, **kw)
    surr = get_surrogate(model)

    def trunk_apply(x):
        return _trunk_apply(x, trunk_params)

    def head_apply(h):
        return _head_apply(h, head_params)

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
        trunk_apply=trunk_apply,
        head_apply=head_apply,
        trunk_params=trunk_params,
        head_params=head_params,
    )
