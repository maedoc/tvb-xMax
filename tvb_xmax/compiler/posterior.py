"""Posterior: in-tree amortized neural posterior estimation over IR.

Uses the lightweight conditional mixture-density implementation adapted from
``maedoc/tvbl``.  This removes the runtime dependency on both ``sbi`` and
Torch while retaining batched conditional posterior samples.
"""

from __future__ import annotations

import pickle
import numpy as np

from ..ir import CompiledArtifact
from ..cde import MDNEstimator, MAFEstimator


def train_posterior(theta, features, algo="mdn", prog=True, **kw):
    """Train an NPE posterior on (theta, features) simulation pairs.

    ``mdn`` and ``maf`` are in-tree, TVBL-derived conditional estimators.
    """
    theta, features = np.asarray(theta), np.asarray(features)
    estimators = {"mdn": MDNEstimator, "maf": MAFEstimator}
    if algo not in estimators:
        raise ValueError(f"unknown posterior algorithm {algo!r}; choose 'mdn' or 'maf'")
    train_keys = {"n_iter", "learning_rate", "seed"}
    train_kw = {key: kw.pop(key) for key in tuple(kw) if key in train_keys}
    estimator = estimators[algo](theta.shape[-1], features.shape[-1], **kw)
    return estimator.train(theta, features, prog=prog, **train_kw)


def attach_posterior(artifact: CompiledArtifact, theta, features, **kw):
    """Train a posterior on the same sim budget used to train the surrogate
    and bind it into ``artifact.posterior_sample``."""
    posterior = train_posterior(theta, features, **kw)

    def sample_fn(xf_obs, n_samples):
        return posterior.sample_batched((n_samples,), x=xf_obs)

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
    import numpy as np

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

    # A dependency-free C2ST: nearest-centroid classifier evaluated on a
    # deterministic hold-out split.  0.5 is indistinguishable; 1.0 is bad.
    split = max(1, n_test // 2)
    train = np.concatenate([dap[:split], true[:split]])
    labels = np.concatenate([np.zeros(split), np.ones(split)])
    centroid0 = train[labels == 0].mean(axis=0)
    centroid1 = train[labels == 1].mean(axis=0)
    test = np.concatenate([dap[split:], true[split:]])
    expected = np.concatenate([np.zeros(n_test - split), np.ones(n_test - split)])
    if not len(test):
        return float("nan")
    predicted = (np.sum((test - centroid1) ** 2, axis=1)
                 < np.sum((test - centroid0) ** 2, axis=1)).astype(float)
    return float(np.mean(predicted == expected))


def save_artifact(artifact: CompiledArtifact, fname: str) -> None:
    """Pickle a compiled artifact (surrogate params + posterior).

    The artifact's apply closures (``surrogate_apply``, ``trunk_apply``,
    ``head_apply``) are rebuilt on load from the stored ``trunk_params`` /
    ``head_params`` via :func:`codegen.rebuild_apply_fns`, so the surrogate
    round-trips with bit-identical outputs.

    The ``posterior_sample`` callable closes over a trained estimator, which
    is NOT picklable; it is dropped on save and set to ``None`` on load.  To
    restore posterior sampling after ``load_artifact``, call
    :func:`attach_posterior` again (cheap: reuses the same sim budget).
    """
    with open(fname, "wb") as fd:
        pickle.dump(artifact, fd)


def load_artifact(fname: str) -> CompiledArtifact:
    with open(fname, "rb") as fd:
        return pickle.load(fd)
