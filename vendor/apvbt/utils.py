"""Utility functions for APVBT project.

Pure utility functions with no class state dependencies.
"""

import numpy as np
import pickle


def load_pkl(fname):
    """Load a pickle file."""
    with open(fname, 'rb') as fd:
        pkl = pickle.load(fd)
    return pkl


seed = 42


def small(*sh):
    """Generate small random initial weights for neural networks."""
    import jax
    return jax.random.normal(
        jax.random.PRNGKey(seed), sh) * 1e-3


def triu_to_mat(triu):
    """Convert upper triangular representation to full symmetric matrix.

    Args:
        triu: Array of shape (n_subj, n_edges) containing upper triangular values

    Returns:
        Array of shape (n_subj, n_roi, n_roi) with symmetric matrices
    """
    import jax.numpy as jp
    n = triu.shape[1]
    nn = int(jp.ceil(jp.sqrt(n*2)))
    i, j = jp.triu_indices(nn, k=1)
    mat = jp.zeros((triu.shape[0], nn, nn), 'f')
    mat = mat.at[:, i, j].set(triu).at[:, j, i].set(triu)
    return mat


def all_conf_rates(wbs, conns):
    """Calculate confusion rates between parcellations.

    Measures how well the cross-coder can recover connectomes when
    encoding from one parcellation and decoding to another.

    Args:
        wbs: List of encoder-decoder weight/bias tuples
        conns: List of connectome arrays per parcellation

    Returns:
        Confusion rate matrix (n_parc, n_parc)
    """
    import jax, jax.numpy as jp
    @jax.jit
    def dist(c_d, c_d_h):
        return jp.sum((c_d[:,None] - c_d_h)**2, axis=-1)
    # TODO consider graph & dynamical metrics
    crs = np.zeros((len(conns),)*2)
    for i, (((ew, eb), _), c_e) in enumerate(zip(wbs, conns)):
        u = c_e @ ew + eb  # encode from parc i
        for j, ((_, (dw, db)), c_d) in enumerate(zip(wbs, conns)):
            c_d_h = u @ dw + db
            ok = dist(c_d, c_d_h).argmin(axis=1) == jp.r_[:c_d.shape[0]]
            crs[i, j] = 1 - ok.mean()
    return crs


class MvNorm:
    """Multivariate normal distribution helper for sampling latent codes."""

    def __init__(self, us, u_mean, u_cov, key=None):
        """Initialize multivariate normal distribution.

        Args:
            us: Latent codes array
            u_mean: Mean vector
            u_cov: Covariance matrix
            key: JAX random key (optional)
        """
        import jax
        self.us = us
        self.u_mean = u_mean
        self.u_cov = u_cov
        self.key = key or jax.random.PRNGKey(42)

    def sample(self, n):
        """Sample n latent codes from the distribution."""
        import jax
        self.key, key = jax.random.split(self.key)
        return jax.random.multivariate_normal(
            key, self.u_mean, self.u_cov, shape=(n,))


def apply(f, *args, B=1):
    """Apply function in parallel across cores with batching.

    Args:
        f: Function to apply
        *args: Arguments to vectorize over
        B: Batch size per core

    Returns:
        Results reshaped to (n_total, ...)
    """
    import vbjax as vb, jax, jax.numpy as jp
    args_ = [_.reshape((-1, vb.cores, B//vb.cores) + _.shape[1:]) for _ in args]
    pvf = jax.pmap(jax.vmap(f))
    xs = jp.array([
        pvf(*[_[i] for _ in args_])
        for i in range(args_[0].shape[0])])
    return xs.reshape((-1, ) + xs.shape[3:])
