"""Tests for SimBudget dataclass and the from_apvbt adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tvb_xmax._apvbt import XCode, MvNorm
from tvb_xmax.ir import SimBudget


# ---------------------------------------------------------------------------
# Unit tests for SimBudget dataclass
# ---------------------------------------------------------------------------

class TestSimBudgetValidation:
    """Shape and value checks for the SimBudget dataclass."""

    def test_valid_budget(self):
        budget = SimBudget(
            U=jnp.zeros((10, 4)),
            Theta=jnp.zeros((10, 2)),
            XF=jnp.zeros((10, 8)),
            model="hopf",
            feature="var",
            nlat=4,
        )
        budget.validate()  # should not raise

    def test_mismatched_U_Theta(self):
        budget = SimBudget(
            U=jnp.zeros((10, 4)),
            Theta=jnp.zeros((5, 2)),
            XF=jnp.zeros((10, 8)),
            nlat=4,
        )
        with pytest.raises(ValueError, match="Theta samples"):
            budget.validate()

    def test_mismatched_U_XF(self):
        budget = SimBudget(
            U=jnp.zeros((10, 4)),
            Theta=jnp.zeros((10, 2)),
            XF=jnp.zeros((5, 8)),
            nlat=4,
        )
        with pytest.raises(ValueError, match="XF samples"):
            budget.validate()

    def test_mismatched_nlat(self):
        budget = SimBudget(
            U=jnp.zeros((10, 4)),
            Theta=jnp.zeros((10, 2)),
            XF=jnp.zeros((10, 8)),
            nlat=8,
        )
        with pytest.raises(ValueError, match="nlat"):
            budget.validate()

    def test_len_and_properties(self):
        budget = SimBudget(
            U=jnp.zeros((10, 4)),
            Theta=jnp.zeros((10, 2)),
            XF=jnp.zeros((10, 8)),
        )
        assert len(budget) == 10
        assert budget.d_param == 2
        assert budget.d_feat == 8

    def test_len_and_properties_varied(self):
        budget = SimBudget(
            U=jnp.zeros((25, 16)),
            Theta=jnp.zeros((25, 4)),
            XF=jnp.zeros((25, 10)),
        )
        assert len(budget) == 25
        assert budget.d_param == 4
        assert budget.d_feat == 10


# ---------------------------------------------------------------------------
# Integration test: from_apvbt (slow, requires building a real XCode)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_from_apvbt_produces_valid_budget():
    """Verify from_apvbt produces a SimBudget with correct shapes using
    a minimal XCode trained on synthetic data and a toy model_fn."""
    from tvb_xmax.compiler.sim_budget import from_apvbt

    n_regions = 6
    n_triu = n_regions * (n_regions - 1) // 2  # 15
    n_subjects = 40
    nlat = 4
    n_samples = 16
    batch_size = 8

    # Build minimal XCode
    xc = XCode()
    xc.parcs = ["parc_a"]
    xc.tts = n_subjects // 2
    xc.wbs = []
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    conn_data = jax.random.normal(k1, (n_subjects, n_triu))
    xc.conns = [jnp.array(conn_data.astype(jnp.float32))]
    xc.means = [jnp.array(conn_data.mean(axis=0).astype(jnp.float32))]
    xc.train(nlat, niter=10, mb=8, nlog=10)
    mvn = xc.calc_mvn(nlat)

    # Toy model_fn
    def model_fn(w, k, D, use_pmap=True):
        inp = w.sum(axis=-1)
        noise = jax.random.normal(jax.random.PRNGKey(0), (w.shape[0], w.shape[-1]))
        return inp + k[:, None] + D[:, None] * noise

    # Call the adapter
    budget = from_apvbt(
        xc, model="test", mvn=mvn, parc="parc_a",
        n_samples=n_samples, model_fn=model_fn,
        prog=False, use_pmap=False,
        batch_size=batch_size, num_batch=n_samples // batch_size,
    )

    # Validate
    budget.validate()
    assert isinstance(budget, SimBudget)
    assert budget.U.shape == (n_samples, nlat)
    assert budget.Theta.shape == (n_samples, 2)  # k, D
    assert budget.XF.shape == (n_samples, n_regions)
    assert budget.model == "test"
    assert budget.feature == "var"
    assert budget.nlat == nlat

    # Verify Theta values are in the prior range [0.1, 0.3] for k and [0.2, 0.4] for D
    # (the apvbt sample_model samples k ~ U(0.1, 0.3) and D ~ U(0.2, 0.4))
    assert jnp.all(budget.Theta[:, 0] >= 0.09), "k values below prior range"
    assert jnp.all(budget.Theta[:, 0] <= 0.31), "k values above prior range"
    assert jnp.all(budget.Theta[:, 1] >= 0.19), "D values below prior range"
    assert jnp.all(budget.Theta[:, 1] <= 0.41), "D values above prior range"


@pytest.mark.slow
def test_from_apvbt_surrogate_compile():
    """Verify a SimBudget from from_apvbt can be used to train a surrogate."""
    from tvb_xmax.compiler.sim_budget import from_apvbt
    from tvb_xmax.compiler import codegen

    n_regions = 6
    n_triu = n_regions * (n_regions - 1) // 2
    n_subjects = 40
    nlat = 4
    n_samples = 16
    batch_size = 8

    xc = XCode()
    xc.parcs = ["parc_a"]
    xc.tts = n_subjects // 2
    xc.wbs = []
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    conn_data = jax.random.normal(k1, (n_subjects, n_triu))
    xc.conns = [jnp.array(conn_data.astype(jnp.float32))]
    xc.means = [jnp.array(conn_data.mean(axis=0).astype(jnp.float32))]
    xc.train(nlat, niter=10, mb=8, nlog=10)
    mvn = xc.calc_mvn(nlat)

    def model_fn(w, k, D, use_pmap=True):
        inp = w.sum(axis=-1)
        noise = jax.random.normal(jax.random.PRNGKey(0), (w.shape[0], w.shape[-1]))
        return inp + k[:, None] + D[:, None] * noise

    budget = from_apvbt(
        xc, model="test", mvn=mvn, parc="parc_a",
        n_samples=n_samples, model_fn=model_fn,
        prog=False, use_pmap=False,
        batch_size=batch_size, num_batch=n_samples // batch_size,
    )

    # Compile surrogate from budget
    artifact = codegen.compile_artifact(
        model="hopf",
        feature="var",
        sim_pairs=(budget.U, budget.Theta, budget.XF),
        nlat=nlat,
        d_feat=n_regions,
        niter=200,
    )
    assert jnp.isfinite(artifact.surrogate_mse)
    assert artifact.surrogate_mse < 10.0

    # Forward pass with correct d_param
    u = mvn.sample(1)[0]
    theta = jnp.array([0.5, 0.5])  # 2 params: k, D
    xf = artifact(u, theta)
    assert xf.shape == (n_regions,)
    assert jnp.all(jnp.isfinite(xf))
