"""Lowering: cross-code connectivity into the parcellation-invariant latent.

This is the stage that gives tvb-xMax its "any connectivity in" property.
A raw connectome in *any* parcellation known to the cross-coder is encoded
to the shared latent ``u``; a pre-encoded latent passes through unchanged.
Parameters are normalized to ``[0,1]`` via the model's declared bounds so
the surrogate trunk is parcellation- and scale-invariant.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from ..ir import IRSpec, IRProgram
from ..surrogates import get_surrogate


def _encode_subject(crosscoder: Any, nlat: int, parc: str,
                     triu: jnp.ndarray) -> jnp.ndarray:
    """Encode a single subject's flat upper-triangular connectome.

    ``CrossCoder.encode`` and ``XCode.encode_conn`` only encode the *stored*
    cohort (``self.conns``); neither accepts a new subject's matrix.  This
    helper applies the cross-coder's trained encoder weights to *new* subject
    data, normalizing it the same way the view was normalized at training
    time so the resulting latent is comparable to the cohort distribution.

    Args:
        crosscoder: a ``vbjax.CrossCoder`` or apvbt ``XCode``.
        nlat: latent dimension (selects the trained architecture).
        parc: parcellation name (selects the view).
        triu: 1-D array of flat upper-triangular connectome weights.

    Returns:
        Latent vector ``u`` of shape ``(nlat,)``.
    """
    iparc = crosscoder.parcs.index(parc)
    mean = jnp.asarray(crosscoder.means[iparc])

    norm_types = getattr(crosscoder, "norm_types", None)
    if not norm_types:
        norm_types = ["center"] * len(crosscoder.parcs)
    stds = getattr(crosscoder, "stds", None)
    if not stds:
        stds = [1.0] * len(crosscoder.parcs)
    scales = getattr(crosscoder, "scales", None)
    if not scales:
        scales = [1.0] * len(crosscoder.parcs)

    nt = norm_types[iparc]
    std = float(stds[iparc])
    scale = float(scales[iparc])
    if nt == "zscore":
        norm = (triu - mean) / std
    elif nt == "center":
        norm = triu - mean
    elif nt == "logit":
        eps = 1e-6
        x = (triu / (scale + 1e-9)) * (1 - 2 * eps) + eps
        logits = jnp.log(x / (1 - x))
        norm = (logits - mean) / std
    else:
        norm = triu

    if hasattr(crosscoder, "_get_arch"):
        ta = crosscoder._get_arch(nlat)
        wb = ta.wbs[iparc]
        if getattr(crosscoder, "variational", False):
            (w_mu, b_mu), _ = wb[0]
            u = norm @ w_mu + b_mu
        else:
            (ew, eb), _ = wb
            u = norm @ ew + eb
    else:
        iarch = list(crosscoder.arch).index(nlat)
        (ew, eb), _ = crosscoder.wbs[iarch][iparc]
        u = norm @ ew + eb

    return jnp.asarray(u)


def lower(spec: IRSpec, crosscoder) -> IRProgram:
    """Lower an :class:`IRSpec` to an :class:`IRProgram` using ``crosscoder``.

    Args:
        spec: validated source spec.
        crosscoder: a ``vbjax.CrossCoder`` (or apvbt ``XCode``) with a trained
            architecture whose ``nlat`` matches the surrogate's expected
            latent dimension.

    Returns:
        IRProgram ready for optimization / codegen.
    """
    surr = get_surrogate(spec.model)
    pspace = surr.get_parameter_space()
    nlat = surr.nlat

    # --- cross-code connectivity -> latent u ---
    if spec.connectivity_is_latent:
        u = jnp.asarray(spec.connectivity)
        if u.shape[-1] != nlat:
            raise ValueError(
                f"latent u has dim {u.shape[-1]}, surrogate expects nlat={nlat}"
            )
    else:
        parc = spec.parcellation
        if parc not in crosscoder.parcs:
            raise KeyError(
                f"parcellation {parc!r} not in cross-coder views {crosscoder.parcs}"
            )
        w = jnp.asarray(spec.connectivity)
        if w.ndim != 2:
            raise ValueError(
                f"connectivity must be a 2-D matrix, got shape {w.shape}"
            )
        i, j = jnp.triu_indices(w.shape[-1], k=1)
        triu = w[i, j]
        u = _encode_subject(crosscoder, nlat, parc, triu)
        u = jnp.asarray(u)
        if u.ndim > 1:
            u = u.reshape(-1)

    # --- normalize parameters -> [0,1]^d ---
    names, lo, hi = [], [], []
    for name, pdef in pspace.parameters.items():
        val = spec.parameters.get(name, pdef.default)
        names.append(name)
        lo.append(pdef.bounds[0])
        hi.append(pdef.bounds[1])
    lo, hi = jnp.array(lo), jnp.array(hi)
    theta_raw = jnp.array([float(spec.parameters.get(n, pspace.parameters[n].default))
                           for n in names])
    theta = (theta_raw - lo) / (hi - lo)
    theta = jnp.clip(theta, 0.0, 1.0)

    return IRProgram(
        model=spec.model,
        u=u,
        theta=theta,
        param_names=tuple(names),
        feature=spec.feature,
        target=spec.target,
        n_posterior=spec.n_posterior,
        seed=spec.seed,
        parcellation=spec.parcellation,
        param_bounds=tuple(zip(lo.tolist(), hi.tolist())),
    )
