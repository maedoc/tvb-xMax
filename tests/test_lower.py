"""Tests for the lowering compiler stage (tvb_xmax.compiler.lower).

Covers latent passthrough, matrix encoding, T0.1 regression (different
connectivity -> different u), param normalization, and input validation.
"""

import jax
import jax.numpy as jnp
import pytest

from tvb_xmax import ir
from tvb_xmax.compiler import lower
from tests.conftest import NLAT, D_PARAM, N_REGIONS, N_TRIU


def test_lower_latent_passthrough(toy_crosscoder):
    """Latent-mode IRSpec: connectivity passes through as u unchanged."""
    u_in = jnp.linspace(-1, 1, NLAT)
    spec = ir.IRSpec(
        model="hopf",
        connectivity=u_in,
        connectivity_is_latent=True,
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, toy_crosscoder)
    assert prog.model == "hopf"
    assert prog.u.shape == (NLAT,)
    assert jnp.allclose(prog.u, u_in)
    assert prog.theta.shape == (D_PARAM,)
    assert jnp.all(prog.theta >= 0.0)
    assert jnp.all(prog.theta <= 1.0)
    assert prog.param_names == ("k", "D", "eta", "omega")


def test_lower_matrix_mode_different_u(toy_crosscoder):
    """T0.1 regression: different connectivity matrices yield different u."""
    spec1 = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    spec2 = ir.IRSpec(
        model="hopf",
        connectivity=jnp.ones((N_REGIONS, N_REGIONS)),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog1 = lower.lower(spec1, toy_crosscoder)
    prog2 = lower.lower(spec2, toy_crosscoder)
    assert not jnp.allclose(prog1.u, prog2.u, atol=1e-5), (
        "T0.1 regression: different input connectivities must produce different u"
    )


def test_lower_same_matrix_same_u(toy_crosscoder):
    """Deterministic encoding: same input -> same u."""
    conn = jnp.eye(N_REGIONS)
    spec1 = ir.IRSpec(
        model="hopf",
        connectivity=conn,
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    spec2 = ir.IRSpec(
        model="hopf",
        connectivity=conn,
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog1 = lower.lower(spec1, toy_crosscoder)
    prog2 = lower.lower(spec2, toy_crosscoder)
    assert jnp.allclose(prog1.u, prog2.u, atol=1e-7)


def test_lower_param_normalization(toy_crosscoder):
    """Params at known bounds produce normalized theta of 0.0 and 1.0."""
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 1.0, "D": 0.0},
    )
    prog = lower.lower(spec, toy_crosscoder)
    assert prog.param_names == ("k", "D", "eta", "omega")
    assert prog.theta.shape == (D_PARAM,)
    # k at upper bound [0,1] -> 1.0
    assert abs(float(prog.theta[0]) - 1.0) < 1e-6
    # D at lower bound [0,1] -> 0.0
    assert abs(float(prog.theta[1]) - 0.0) < 1e-6
    # eta default 1.0 in [-2, 2] -> (1 - -2) / (2 - -2) = 0.75
    assert abs(float(prog.theta[2]) - 0.75) < 1e-6
    # omega default pi in [0.9*pi, 1.1*pi] -> (pi - 0.9*pi) / (0.2*pi) = 0.5
    assert abs(float(prog.theta[3]) - 0.5) < 1e-6


def test_lower_wrong_nlat_raises(toy_crosscoder):
    """Latent with wrong dimension raises ValueError mentioning nlat."""
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(5),
        connectivity_is_latent=True,
    )
    with pytest.raises(ValueError, match="nlat"):
        lower.lower(spec, toy_crosscoder)


def test_lower_unknown_parc_raises(toy_crosscoder):
    """Matrix mode with unknown parcellation raises KeyError."""
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros((N_REGIONS, N_REGIONS)),
        connectivity_is_latent=False,
        parcellation="parc_unknown",
    )
    with pytest.raises(KeyError):
        lower.lower(spec, toy_crosscoder)


# ---------------------------------------------------------------------------
# Edge-case coverage for lower.py (T2.10)
# ---------------------------------------------------------------------------


def test_lower_missing_norm_attrs(toy_nlat):
    """Coverage: crosscoder without norm_types/stds/scales uses defaults."""
    class _MinimalCrossCoder:
        parcs = ["parc_a"]
        means = [jnp.zeros(N_TRIU)]
        nlat = toy_nlat

        def _get_arch(self, nlat):
            class _Arch:
                wbs = [
                    (
                        (jax.random.normal(jax.random.PRNGKey(0), (N_TRIU, nlat)),
                         jnp.zeros(nlat)),
                        (jax.random.normal(jax.random.PRNGKey(1), (nlat, N_TRIU)),
                         jnp.zeros(N_TRIU)),
                    ),
                ]
            return _Arch()

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, _MinimalCrossCoder())
    assert prog.u.shape == (toy_nlat,)


def test_lower_parc_b_zscore(toy_crosscoder):
    """Coverage: parc_b with zscore normalization hits the zscore branch."""
    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_b",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, toy_crosscoder)
    assert prog.u.shape == (NLAT,)


def test_lower_logit_norm(toy_nlat):
    """Coverage: logit normalization type."""
    class _LogitCrossCoder:
        parcs = ["parc_a"]
        means = [jnp.zeros(N_TRIU)]
        stds = [0.5]
        scales = [2.0]
        norm_types = ["logit"]
        nlat = toy_nlat

        def _get_arch(self, nlat):
            class _Arch:
                wbs = [
                    (
                        (jax.random.normal(jax.random.PRNGKey(0), (N_TRIU, nlat)),
                         jnp.zeros(nlat)),
                        (jax.random.normal(jax.random.PRNGKey(1), (nlat, N_TRIU)),
                         jnp.zeros(N_TRIU)),
                    ),
                ]
            return _Arch()

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, _LogitCrossCoder())
    assert prog.u.shape == (toy_nlat,)


def test_lower_unknown_norm(toy_nlat):
    """Coverage: unknown norm type passes raw triu through."""
    class _UnknownNormCrossCoder:
        parcs = ["parc_a"]
        means = [jnp.zeros(N_TRIU)]
        stds = [0.5]
        scales = [1.0]
        norm_types = ["raw"]

        def _get_arch(self, nlat):
            class _Arch:
                wbs = [
                    (
                        (jax.random.normal(jax.random.PRNGKey(0), (N_TRIU, nlat)),
                         jnp.zeros(nlat)),
                        (jax.random.normal(jax.random.PRNGKey(1), (nlat, N_TRIU)),
                         jnp.zeros(N_TRIU)),
                    ),
                ]
            return _Arch()

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, _UnknownNormCrossCoder())
    assert prog.u.shape == (toy_nlat,)


def test_lower_variational_encoder(toy_nlat):
    """Coverage: variational=True path in _encode_subject."""
    class _VariationalCrossCoder:
        parcs = ["parc_a"]
        means = [jnp.zeros(N_TRIU)]
        stds = [0.5]
        scales = [1.0]
        norm_types = ["center"]
        variational = True

        def _get_arch(self, nlat):
            class _Arch:
                wbs = [
                    (
                        ((jax.random.normal(jax.random.PRNGKey(0), (N_TRIU, nlat)),
                          jnp.zeros(nlat)),
                         (jax.random.normal(jax.random.PRNGKey(1), (N_TRIU, nlat)),
                          jnp.zeros(nlat))),
                        ((jax.random.normal(jax.random.PRNGKey(2), (nlat, N_TRIU)),
                          jnp.zeros(N_TRIU)),
                         (jax.random.normal(jax.random.PRNGKey(3), (nlat, N_TRIU)),
                          jnp.zeros(N_TRIU))),
                    ),
                ]
            return _Arch()

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, _VariationalCrossCoder())
    assert prog.u.shape == (toy_nlat,)


def test_lower_xcode_style(toy_nlat):
    """Coverage: crosscoder without _get_arch (XCode/arch+wb path)."""
    class _XCodeCrossCoder:
        parcs = ["parc_a"]
        means = [jnp.zeros(N_TRIU)]
        stds = [0.5]
        scales = [1.0]
        norm_types = ["center"]
        arch = [toy_nlat]
        wbs = [
            [
                ((jax.random.normal(jax.random.PRNGKey(0), (N_TRIU, toy_nlat)),
                  jnp.zeros(toy_nlat)),
                 (jax.random.normal(jax.random.PRNGKey(1), (toy_nlat, N_TRIU)),
                  jnp.zeros(N_TRIU))),
            ],
        ]

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.eye(N_REGIONS),
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    prog = lower.lower(spec, _XCodeCrossCoder())
    assert prog.u.shape == (toy_nlat,)


def test_lower_batch_connectivity_raises(toy_crosscoder):
    """Batch connectivity (>2D) raises ValueError."""
    conn = jnp.zeros((2, N_REGIONS, N_REGIONS))
    spec = ir.IRSpec(
        model="hopf",
        connectivity=conn,
        connectivity_is_latent=False,
        parcellation="parc_a",
        parameters={"k": 0.5, "D": 0.3},
    )
    with pytest.raises(ValueError, match="connectivity must be a 2-D matrix"):
        lower.lower(spec, toy_crosscoder)
