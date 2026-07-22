"""Tests for the tvb-xMax compiler optimize stage."""
import jax.numpy as jnp
import pytest

from tvb_xmax.ir import IRProgram
from tvb_xmax.compiler import optimize


def _make_prog(u=None, theta=None):
    return IRProgram(
        model="hopf",
        u=jnp.arange(16, dtype=jnp.float32) if u is None else u,
        theta=jnp.zeros(4, dtype=jnp.float32) if theta is None else theta,
        param_names=("k", "D", "eta", "omega"),
        feature="var",
        target="features",
        n_posterior=100,
        seed=0,
    )


def test_condition_latent_no_mvn():
    prog = _make_prog()
    out = optimize.condition_latent(prog, mvn=None)
    assert jnp.array_equal(out.u, prog.u)
    assert out.model == prog.model
    assert jnp.array_equal(out.theta, prog.theta)
    assert out.param_names == prog.param_names
    assert out.feature == prog.feature
    assert out.target == prog.target
    assert out.n_posterior == prog.n_posterior
    assert out.seed == prog.seed


def test_condition_latent_with_mvn(toy_mvn):
    u_in = jnp.ones(16, dtype=jnp.float32) * 5.0
    prog = _make_prog(u=u_in)
    out = optimize.condition_latent(prog, mvn=toy_mvn)
    expected = jnp.ones(16, dtype=jnp.float32) * 5.0
    assert jnp.allclose(out.u, expected, atol=1e-6)
    assert out.model == prog.model
    assert jnp.array_equal(out.theta, prog.theta)
    assert out.param_names == prog.param_names


def test_reparam_heterogeneous_is_noop():
    prog = _make_prog()
    out = optimize.reparam_heterogeneous(prog)
    assert jnp.array_equal(out.u, prog.u)
    assert jnp.array_equal(out.theta, prog.theta)
    assert out.model == prog.model
    assert out.param_names == prog.param_names
    assert out.feature == prog.feature
    assert out.target == prog.target
    assert out.n_posterior == prog.n_posterior
    assert out.seed == prog.seed


def test_optimize_runs_both_passes(toy_mvn):
    prog = _make_prog()
    out = optimize.optimize(prog, mvn=toy_mvn)
    assert isinstance(out, IRProgram)
    assert out.u.shape == prog.u.shape
    assert out.theta.shape == prog.theta.shape
    assert out.model == prog.model
    assert out.param_names == prog.param_names
    assert out.feature == prog.feature
    assert out.target == prog.target
    assert out.n_posterior == prog.n_posterior
    assert out.seed == prog.seed


def test_mvn_uses_u_mean_u_cov_attrs():
    nlat = 16
    u_in = jnp.ones(nlat, dtype=jnp.float32) * 7.0
    prog = _make_prog(u=u_in)
    custom_mvn = type("Obj", (), {
        "u_mean": jnp.ones(nlat) * 2.0,
        "u_cov": jnp.eye(nlat) * 4.0,
    })()
    out = optimize.condition_latent(prog, mvn=custom_mvn)
    expected = jnp.ones(nlat, dtype=jnp.float32) * 2.5  # (7 - 2) / 2
    assert jnp.allclose(out.u, expected, atol=1e-6)
    assert jnp.array_equal(out.theta, prog.theta)
