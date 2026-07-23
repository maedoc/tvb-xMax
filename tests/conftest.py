"""Shared test fixtures for the tvb-xMax test suite.

All compiler-stage test files import from here so cross-coder, simulation
budget, and MVN objects are built once and reused across P2 tests.
"""

from __future__ import annotations

# Force CPU before JAX initializes, to avoid CUDA plugin crash
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from typing import Any, List

import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# Module-level constants (no fixture injection needed)
# ---------------------------------------------------------------------------
NLAT = 16
D_PARAM = 4
D_FEAT = 8
N_BUDGET = 64
N_REGIONS = 10                     # synthetic connectome size
N_TRIU = N_REGIONS * (N_REGIONS - 1) // 2  # 45


# ---------------------------------------------------------------------------
# Simulated cross-coder
# ---------------------------------------------------------------------------
class _SimCrossCoder:
    """Minimal cross-coder stand-in for compiler tests.

    Satisfies every attribute access made by ``_encode_subject`` in the
    ``_get_arch`` branch (the ``vbjax.CrossCoder`` path).  Two different
    input connectivity matrices always produce two different ``u`` vectors
    because the encoder weights are non-degenerate random projections.
    """

    def __init__(self, nlat: int, n_triu: int, key: jax.Array):
        self.parcs = ["parc_a", "parc_b"]
        self.n_triu = n_triu

        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

        # Per-parcellation normalisation metadata
        self.means = [
            jax.random.normal(k1, (n_triu,)),
            jax.random.normal(k2, (n_triu,)),
        ]
        self.stds = [0.5, 0.7]
        self.scales = [1.0, 1.0]
        self.norm_types = ["center", "zscore"]

        # Encoder weights: one ``((ew, eb), (dw, db))`` pair per parc
        self._wbs = [
            (
                (jax.random.normal(k3, (n_triu, nlat)),   # ew
                 jax.random.normal(k4, (nlat,))),           # eb
                (jax.random.normal(k5, (nlat, n_triu)),    # dw
                 jax.random.normal(k6, (n_triu,))),          # db
            ),
            (
                (jax.random.normal(k7, (n_triu, nlat)),
                 jax.random.normal(k8, (nlat,))),
                (jax.random.normal(k9, (nlat, n_triu)),
                 jax.random.normal(k10, (n_triu,))),
            ),
        ]

    def _get_arch(self, nlat: int) -> Any:
        """Return a mock architecture object (``vbjax.CrossCoder`` path)."""
        return _MockArch(self._wbs)

    @property
    def nonneg(self):
        return [False] * len(self.parcs)

    @property
    def variational(self):
        return False

    def decode_conn(self, arch: int, parc: str, z, clip_positive=None):
        """Minimal decode_conn matching vbjax.CrossCoder interface."""
        from vbjax.crosscoder import triu_to_mat, _denorm
        iparc = self.parcs.index(parc)
        _, (w_dec, b_dec) = self._get_arch(arch).wbs[iparc]
        flat = z @ w_dec + b_dec
        flat = _denorm(flat, self.norm_types[iparc], self.means[iparc],
                       self.stds[iparc], self.scales[iparc], self.nonneg[iparc])
        if clip_positive:
            flat = jnp.maximum(flat, 0.0)
        return triu_to_mat(flat)


class _MockArch:
    """Returned by ``_get_arch``; exposes ``.wbs``."""

    def __init__(self, wbs: List):
        self.wbs = wbs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def toy_nlat() -> int:
    return NLAT


@pytest.fixture(scope="session")
def toy_d_param() -> int:
    return D_PARAM


@pytest.fixture(scope="session")
def toy_d_feat() -> int:
    return D_FEAT


@pytest.fixture(scope="session")
def toy_crosscoder() -> _SimCrossCoder:
    return _SimCrossCoder(NLAT, N_TRIU, jax.random.PRNGKey(0))


@pytest.fixture(scope="session")
def toy_sim_budget() -> tuple:
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    U = jax.random.normal(k1, (N_BUDGET, NLAT))
    Theta = jax.random.uniform(k2, (N_BUDGET, D_PARAM))
    XF = jnp.tanh(U[:, :D_FEAT] + 0.3 * Theta[:, :1])
    return U, Theta, XF


@pytest.fixture(scope="session")
def toy_mvn() -> Any:
    class _MockMVN:
        u_mean = jnp.zeros(NLAT)
        u_cov = jnp.eye(NLAT)
    return _MockMVN()
