"""Swap: the free re-evaluations the compiler exposes.

Because the compiled artifact operates on the parcellation-invariant latent
``u`` and normalized ``theta``, four swaps are essentially free (no
re-simulation, no re-training):

  * swap_parcellation  : re-encode a new connectivity matrix -> new u
  * swap_parameters    : new theta (and optionally re-draw posterior)
  * swap_model         : pick a different artifact from the registry
  * swap_features      : pick an artifact with a different feature head

Only ``swap_model`` / ``swap_features`` require a *different* pre-compiled
artifact; the other two reuse the exact same artifact.
"""

from __future__ import annotations

from typing import Any

from ..ir import IRSpec, SwapKind


def swap_parcellation(spec: IRSpec, connectivity: Any, parcellation: str) -> IRSpec:
    """Return a new spec with a different connectivity/parcellation.

    The same compiled artifact serves this spec: the cross-coder encodes
    the new matrix into the same latent space.  No cache lookup needed.
    """
    return IRSpec(
        model=spec.model,
        connectivity=connectivity,
        connectivity_is_latent=False,
        parcellation=parcellation,
        parameters=dict(spec.parameters),
        feature=spec.feature,
        target=spec.target,
        n_posterior=spec.n_posterior,
        seed=spec.seed,
    )


def swap_parameters(spec: IRSpec, **new_params) -> IRSpec:
    """Return a new spec with overridden parameters.

    The same compiled artifact serves this spec.  No cache lookup needed.
    """
    params = dict(spec.parameters)
    params.update(new_params)
    return IRSpec(
        model=spec.model,
        connectivity=spec.connectivity,
        connectivity_is_latent=spec.connectivity_is_latent,
        parcellation=spec.parcellation,
        parameters=params,
        feature=spec.feature,
        target=spec.target,
        n_posterior=spec.n_posterior,
        seed=spec.seed,
    )


def swap_model(spec: IRSpec, model: str) -> IRSpec:
    """Return a new spec targeting a different model.

    Use with ``pipeline.resolve_artifact(spec, crosscoder, cache, ...)``
    to fetch or compile the artifact.
    """
    return IRSpec(
        model=model,
        connectivity=spec.connectivity,
        connectivity_is_latent=spec.connectivity_is_latent,
        parcellation=spec.parcellation,
        parameters={},  # new model => new parameter space
        feature=spec.feature,
        target=spec.target,
        n_posterior=spec.n_posterior,
        seed=spec.seed,
    )


def swap_features(spec: IRSpec, feature: str) -> IRSpec:
    """Return a new spec targeting a different feature extractor.

    Use with ``pipeline.resolve_artifact(spec, crosscoder, cache, ...)``
    to fetch or compile the artifact.
    """
    return IRSpec(
        model=spec.model,
        connectivity=spec.connectivity,
        connectivity_is_latent=spec.connectivity_is_latent,
        parcellation=spec.parcellation,
        parameters=dict(spec.parameters),
        feature=feature,
        target=spec.target,
        n_posterior=spec.n_posterior,
        seed=spec.seed,
    )


SWAP_TABLE = {
    SwapKind.PARCELLATION: swap_parcellation,
    SwapKind.PARAMETERS: swap_parameters,
    SwapKind.MODEL: swap_model,
    SwapKind.FEATURES: swap_features,
}


def apply_swap(spec: IRSpec, kind: str, **kw) -> IRSpec:
    """Dispatch a swap by name."""
    if kind not in SWAP_TABLE:
        raise KeyError(f"unknown swap {kind!r}; known: {list(SWAP_TABLE)}")
    return SWAP_TABLE[kind](spec, **kw)
