"""Adapters for producing a SimBudget from external simulation pipelines.

Provides :func:`from_apvbt` which wraps the extracted apvbt simulation
pipeline (T5.1) to produce a real :class:`SimBudget` from a trained XCode
cross-coder and a cohort MVN.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import jax.numpy as jnp

from ..ir import SimBudget


def from_apvbt(
    xc,
    model: Union[str, Callable],
    mvn,
    parc: str,
    n_samples: int = 4096,
    feature: str = "var",
    model_fn: Optional[Callable] = None,
    num_batch: Optional[int] = None,
    batch_size: int = 128,
    prog: bool = False,
    use_pmap: bool = False,
) -> SimBudget:
    """Produce a SimBudget by calling the extracted apvbt simulation pipeline.

    Args:
        xc: A trained apvbt XCode (or compatible cross-coder).  Must have
            ``parcs``, ``arch``, ``decode_conn``.
        model: Model name string (stored in the returned SimBudget) **or**
            a callable dynamics function ``f(w, k, D, use_pmap=True)``.
            If a string is given, ``model_fn`` must be provided; the function
            does NOT look up ``vbjax.sim`` because ``vbjax`` has no ``sim``
            submodule.
        mvn: Cohort MVN distribution (``MvNorm`` from ``_apvbt.utils``).
            The latent dimension is inferred from ``mvn.u_mean``.
        parc: Parcellation name.  Must be in ``xc.parcs``.
        n_samples: Total number of simulation samples to draw.  The underlying
            apvbt sampler works in ``(num_batch, batch_size)`` chunks; these
            are derived from ``n_samples`` when ``num_batch`` is not specified.
        feature: Feature extraction name for the returned SimBudget metadata.
        model_fn: Explicit dynamics callable.  When provided, ``model`` can be
            a plain string (used for SimBudget metadata only) and this callable
            is used for the actual simulation.  **Required** when ``model`` is
            a string, because ``vbjax`` has no ``sim`` submodule.
        num_batch: Number of batches to draw.  When not set, inferred from
            ``n_samples / batch_size``.
        batch_size: Samples per batch (default 128).  Always forwarded to
            ``sample_model``.
        prog: Show progress bar.
        use_pmap: Use ``pmap`` for parallel simulation.

    Returns:
        SimBudget with real simulation data.

    Raises:
        ImportError: If the apvbt extracted bits are not available.
        ValueError: If ``xc`` / ``parc`` / ``mvn`` / ``model`` are incompatible,
            or if ``model`` is a string and ``model_fn`` is not provided.
    """
    try:
        from tvb_xmax._apvbt import simulation as apvbt_sim
    except ImportError:
        raise ImportError(
            "apvbt extraction not available; supply your own SimBudget "
            "or re-run T5.1 to reintegrate the apvbt simulation pipeline"
        )

    # ---- resolve the dynamics callable ----
    if model_fn is not None:
        dynamics_fn = model_fn
    elif callable(model):
        dynamics_fn = model
    else:
        raise ValueError(
            f"Model {model!r} requires an explicit model_fn callable. "
            f"vbjax has no 'sim' submodule; pass model_fn=<callable> with "
            f"signature f(w, k, D, use_pmap) -> features, where w is "
            f"(batch, n_roi, n_roi), k and D are (batch,)."
        )

    # ---- derive latent dimension from mvn ----
    nlat = int(mvn.u_mean.shape[0])

    # ---- convert n_samples -> (num_batch, batch_size) ----
    if num_batch is None:
        num_batch = max(1, math.ceil(n_samples / batch_size))

    # ---- run simulation ----
    thetas, xfs = apvbt_sim.sample_model(
        xc, dynamics_fn, mvn, parc,
        num_batch, batch_size,
        prog=prog, use_pmap=use_pmap,
    )

    # ---- unpack: thetas = [U, k, D] concatenated ----
    U = jnp.asarray(thetas[:, :nlat])
    Theta = jnp.asarray(thetas[:, nlat:])
    XF = jnp.asarray(xfs)

    return SimBudget(
        U=U,
        Theta=Theta,
        XF=XF,
        model=model if isinstance(model, str) else "",
        feature=feature,
        nlat=nlat,
    )
