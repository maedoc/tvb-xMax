"""NumPy implementation of the lowering and conditioning stages."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..ir import IRProgram, IRSpec
from ..surrogates import get_surrogate


def _encode_subject(crosscoder: Any, nlat: int, parc: str, triu) -> np.ndarray:
    """Apply a trained cross-coder encoder using only NumPy operations."""
    iparc = crosscoder.parcs.index(parc)
    triu = np.asarray(triu, dtype=np.float32)
    mean = np.asarray(crosscoder.means[iparc], dtype=np.float32)
    norm_types = getattr(crosscoder, "norm_types", None) or ["center"] * len(crosscoder.parcs)
    stds = getattr(crosscoder, "stds", None) or [1.0] * len(crosscoder.parcs)
    scales = getattr(crosscoder, "scales", None) or [1.0] * len(crosscoder.parcs)
    nt, std, scale = norm_types[iparc], float(stds[iparc]), float(scales[iparc])
    if nt == "zscore":
        norm = (triu - mean) / std
    elif nt == "center":
        norm = triu - mean
    elif nt == "logit":
        eps = 1e-6
        x = (triu / (scale + 1e-9)) * (1 - 2 * eps) + eps
        norm = (np.log(x / (1 - x)) - mean) / std
    else:
        norm = triu
    if hasattr(crosscoder, "_get_arch"):
        wb = crosscoder._get_arch(nlat).wbs[iparc]
        if getattr(crosscoder, "variational", False):
            (weight, bias), _ = wb[0]
        else:
            (weight, bias), _ = wb
    else:
        iarch = list(crosscoder.arch).index(nlat)
        (weight, bias), _ = crosscoder.wbs[iarch][iparc]
    return np.asarray(norm @ np.asarray(weight) + np.asarray(bias), dtype=np.float32)


def lower(spec: IRSpec, crosscoder) -> IRProgram:
    """Lower an ``IRSpec`` without importing JAX."""
    surr = get_surrogate(spec.model)
    pspace, nlat = surr.get_parameter_space(), surr.nlat
    if spec.connectivity_is_latent:
        u = np.asarray(spec.connectivity, dtype=np.float32).reshape(-1)
        if u.shape[-1] != nlat:
            raise ValueError(f"latent u has dim {u.shape[-1]}, surrogate expects nlat={nlat}")
    else:
        if spec.parcellation not in crosscoder.parcs:
            raise KeyError(f"parcellation {spec.parcellation!r} not in cross-coder views {crosscoder.parcs}")
        w = np.asarray(spec.connectivity, dtype=np.float32)
        rows, cols = np.triu_indices(w.shape[-1], k=1)
        u = _encode_subject(crosscoder, nlat, spec.parcellation, w[..., rows, cols]).reshape(-1)
    names = tuple(pspace.parameters)
    lo = np.asarray([pspace.parameters[n].bounds[0] for n in names], dtype=np.float32)
    hi = np.asarray([pspace.parameters[n].bounds[1] for n in names], dtype=np.float32)
    raw = np.asarray([spec.parameters.get(n, pspace.parameters[n].default) for n in names], dtype=np.float32)
    return IRProgram(spec.model, u, np.clip((raw - lo) / (hi - lo), 0, 1), names,
                     spec.feature, spec.target, spec.n_posterior, spec.seed,
                     spec.parcellation, tuple(zip(lo.tolist(), hi.tolist())))


def optimize(prog: IRProgram, mvn=None) -> IRProgram:
    """Whiten a latent against an MVN using NumPy."""
    if mvn is None:
        return prog
    mean = np.asarray(mvn.u_mean if hasattr(mvn, "u_mean") else mvn.mean)
    cov = np.asarray(mvn.u_cov if hasattr(mvn, "u_cov") else mvn.cov)
    return replace(prog, u=(np.asarray(prog.u) - mean) / np.sqrt(np.clip(np.diag(cov), 1e-6, None)))
