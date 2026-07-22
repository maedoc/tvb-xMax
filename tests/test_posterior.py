"""Tests for the posterior module: NPE training and artifact save/load."""

import os

import jax
import jax.numpy as jnp
import pytest

from tvb_xmax.compiler import codegen
from tvb_xmax.compiler.posterior import (
    attach_posterior,
    load_artifact,
    save_artifact,
    train_posterior,
)
from conftest import D_PARAM, D_FEAT, NLAT


@pytest.mark.slow
def test_train_posterior_returns_callable(toy_sim_budget):
    _, theta, features = toy_sim_budget
    posterior = train_posterior(theta, features)
    assert hasattr(posterior, "sample_batched")
    assert callable(posterior.sample_batched)


@pytest.mark.slow
def test_train_posterior_suppress_progress(toy_sim_budget):
    """Coverage: train_posterior with prog=False hits non-prog path (lines 42-44)."""
    _, theta, features = toy_sim_budget
    posterior = train_posterior(theta, features, prog=False)
    assert hasattr(posterior, "sample_batched")
    # Use a NumPy feature batch, matching the in-tree estimator API.
    import numpy as np
    xf_obs = np.ones((1, D_FEAT), dtype=np.float32)
    samples = posterior.sample_batched((5,), x=xf_obs, show_progress_bars=False)
    assert samples.shape == (5, 1, D_PARAM)


@pytest.mark.slow
def test_attach_posterior_binds_sample_fn(toy_sim_budget, toy_nlat, toy_d_feat):
    _, theta, features = toy_sim_budget
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, toy_nlat, toy_d_feat, niter=100
    )
    artifact = attach_posterior(artifact, theta, features)
    assert callable(artifact.posterior_sample)
    xf_obs = jnp.ones((1, toy_d_feat))
    samples = artifact.posterior_sample(xf_obs, 10)
    assert samples.shape == (10, 1, D_PARAM)


def test_save_artifact_roundtrip(toy_sim_budget, toy_nlat, toy_d_feat, tmp_path):
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, toy_nlat, toy_d_feat, niter=100
    )

    u = jnp.zeros(NLAT)
    theta = jnp.zeros(D_PARAM)
    orig_features = artifact(u, theta)

    fname = tmp_path / "test_posterior_rt.pkl"
    save_artifact(artifact, fname)
    try:
        loaded = load_artifact(fname)

        assert loaded.surrogate_apply is not None
        assert loaded.trunk_apply is not None
        assert loaded.head_apply is not None

        loaded_features = loaded(u, theta)
        assert jnp.allclose(orig_features, loaded_features, atol=1e-5)

        assert loaded.posterior_sample is None
    finally:
        os.remove(fname)


def test_save_load_preserves_fields(toy_sim_budget, toy_nlat, toy_d_feat, tmp_path):
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, toy_nlat, toy_d_feat, niter=100
    )
    artifact.param_names = ("a", "b", "c", "d")
    artifact.surrogate_mse = 0.12345

    fname = tmp_path / "test_posterior_fields.pkl"
    save_artifact(artifact, fname)
    try:
        loaded = load_artifact(fname)
        assert loaded.param_names == ("a", "b", "c", "d")
        assert loaded.surrogate_mse == pytest.approx(0.12345)
        assert loaded.model == "hopf"
        assert loaded.feature == "var"
        assert loaded.nlat == NLAT
    finally:
        os.remove(fname)
