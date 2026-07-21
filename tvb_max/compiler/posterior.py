"""Posterior: amortized neural posterior estimation (NPE) over IR.

Wraps ``sbi`` (the same library apvbt uses) but operates entirely in IR
space: the posterior is over ``theta`` (normalized parameters) conditioned
on surrogate-produced features ``xf``.  Because both the surrogate and the
posterior are amortized, drawing ``10^4`` posterior samples for a new
subject is a single batched forward pass, not a new round of simulation.
"""

from __future__ import annotations

import io
import contextlib
import pickle
import numpy as np

from ..ir import CompiledArtifact


def _to_torch(x, device="cpu"):
    import torch
    return torch.from_numpy(np.asarray(x)).float().to(device)


def train_posterior(theta, features, algo="maf", device="cpu", prog=True,
                    **kw):
    """Train an NPE posterior on (theta, features) simulation pairs.

    Thin wrapper over ``sbi.inference.NPE_C``/``NPE_A`` mirroring apvbt's
    :func:`apvbt.inference.run_sbi`, but the inputs are IR tensors.
    """
    from sbi.inference import NPE_C, NPE_A
    NPE = {"maf": NPE_C, "mdn": NPE_A}[algo]
    mu = _to_torch(np.mean(theta, axis=0), device)
    cov = _to_torch(np.cov(theta.T), device)
    prior = __import__("torch").distributions.MultivariateNormal(mu, cov)
    inference = NPE(prior=prior, show_progress_bars=prog, **kw)
    inference.append_simulations(_to_torch(theta, device), _to_torch(features, device))
    if prog:
        inference.train()
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            inference.train()
    return inference.build_posterior()


def attach_posterior(artifact: CompiledArtifact, theta, features, **kw):
    """Train a posterior on the same sim budget used to train the surrogate
    and bind it into ``artifact.posterior_sample``."""
    posterior = train_posterior(theta, features, **kw)

    def sample_fn(xf_obs, n_samples):
        # sbi posterior.sample_batched((n,), x=...) expects torch tensors
        x = _to_torch(xf_obs)
        s = posterior.sample_batched((n_samples,), x=x, show_progress_bars=False)
        return np.asarray(s)  # (n_samples, B, d_param)

    artifact.posterior_sample = sample_fn
    return artifact


def save_artifact(artifact: CompiledArtifact, fname: str) -> None:
    """Pickle a compiled artifact (surrogate params + posterior)."""
    with open(fname, "wb") as fd:
        pickle.dump(artifact, fd)


def load_artifact(fname: str) -> CompiledArtifact:
    with open(fname, "rb") as fd:
        return pickle.load(fd)
