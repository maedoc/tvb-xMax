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
import torch

from ..ir import CompiledArtifact


def _to_torch(x, device="cpu"):
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
    prior = torch.distributions.MultivariateNormal(mu, cov)
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


def compute_sbc(artifact, theta, features, n_sbc_samples=100,
               n_posterior_samples=500):
    """Compute simulation-based calibration score.

    Draws posterior samples for a subset of test (theta, xf) pairs and
    checks whether the ranks of the true theta among posterior samples
    are uniformly distributed.  Uses Kolmogorov-Smirnov against U[0,1].

    Returns a float in [0, 1] where 1.0 = perfectly calibrated ranks
    (uniform) and 0.0 = completely miscalibrated.
    """
    if artifact.posterior_sample is None:
        return float("nan")
    import numpy as np
    from scipy.stats import kstest

    theta = np.asarray(theta)
    features = np.asarray(features)
    n_test = min(n_sbc_samples, theta.shape[0])
    d_param = theta.shape[-1]

    rng = np.random.default_rng(42)
    idx = rng.choice(theta.shape[0], n_test, replace=False)

    all_ranks = np.zeros((n_test, d_param))
    for i, ix in enumerate(idx):
        xf_obs = features[ix:ix + 1]
        post = artifact.posterior_sample(xf_obs, n_posterior_samples)
        post = np.asarray(post)[:, 0, :]
        true_t = theta[ix]
        all_ranks[i] = np.sum(post < true_t, axis=0)

    ranks_norm = all_ranks.ravel() / float(n_posterior_samples)
    stat, _ = kstest(ranks_norm, "uniform")
    return float(1.0 - stat)


def compute_c2st(artifact, theta, features, n_samples=100):
    """Compute classifier two-sample test score.

    Draws one posterior sample per test theta (data-averaged posterior
    style) and compares them to the true thetas using a trained
    classifier.  Returns the raw c2st accuracy:

        0.5 = indistinguishable (posterior matches truth)  *good*
        1.0 = perfectly distinguishable (posterior is wrong) *bad*
    """
    if artifact.posterior_sample is None:
        return float("nan")
    from sbi.utils.metrics import c2st as _c2st
    import numpy as np
    import torch

    theta = np.asarray(theta)
    features = np.asarray(features)
    n_test = min(n_samples, theta.shape[0])
    d_param = theta.shape[-1]

    rng = np.random.default_rng(42)
    idx = rng.choice(theta.shape[0], n_test, replace=False)

    dap = np.zeros((n_test, d_param))
    true = np.zeros((n_test, d_param))
    for j, ix in enumerate(idx):
        xf_obs = features[ix:ix + 1]
        post = artifact.posterior_sample(xf_obs, 1)
        dap[j] = np.asarray(post)[0, 0]
        true[j] = theta[ix]

    acc = _c2st(torch.from_numpy(dap).float(),
                torch.from_numpy(true).float())
    return float(acc)


def save_artifact(artifact: CompiledArtifact, fname: str) -> None:
    """Pickle a compiled artifact (surrogate params + posterior).

    The artifact's apply closures (``surrogate_apply``, ``trunk_apply``,
    ``head_apply``) are rebuilt on load from the stored ``trunk_params`` /
    ``head_params`` via :func:`codegen.rebuild_apply_fns`, so the surrogate
    round-trips with bit-identical outputs.

    The ``posterior_sample`` callable wraps an ``sbi`` posterior, which is
    NOT picklable; it is dropped on save and set to ``None`` on load.  To
    restore posterior sampling after ``load_artifact``, call
    :func:`attach_posterior` again (cheap: reuses the same sim budget).
    """
    with open(fname, "wb") as fd:
        pickle.dump(artifact, fd)


def load_artifact(fname: str) -> CompiledArtifact:
    with open(fname, "rb") as fd:
        return pickle.load(fd)
