"""CPU-batched NumPy evaluation for portable artifacts."""

import numpy as np


def batched_features(artifact, U, theta):
    x = np.concatenate([np.asarray(U), np.asarray(theta)], axis=1)
    return artifact.head_apply(artifact.trunk_apply(x))


def batched_posterior(artifact, xf_obs, n_samples):
    if artifact.posterior_sample is None:
        raise RuntimeError("artifact has no trained posterior; target='features' only")
    return artifact.posterior_sample(np.asarray(xf_obs), n_samples)


def sharded_features(artifact, U, theta):
    """Portable fallback: a single CPU BLAS batch replaces device sharding."""
    return batched_features(artifact, U, theta)
