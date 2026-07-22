"""Tests for the tvb-xMax compiler codegen (neural surrogate training) stage."""
import os
import tempfile

import jax
import jax.numpy as jnp
import pytest

from tvb_xmax.compiler import codegen
from tvb_xmax.compiler.posterior import save_artifact, load_artifact
from tvb_xmax.ir import CompiledArtifact
from conftest import NLAT, D_PARAM, D_FEAT, N_BUDGET


def test_init_trunk_head_shapes():
    key = jax.random.PRNGKey(0)
    trunk_params = codegen.init_trunk(NLAT + D_PARAM, 128, key)
    head_params = codegen.init_head(128, D_FEAT, key)

    assert len(trunk_params) == 2
    assert len(head_params) == 1

    w1, b1 = trunk_params[0]
    assert w1.shape == (NLAT + D_PARAM, 128)
    assert b1.shape == (128,)

    w2, b2 = trunk_params[1]
    assert w2.shape == (128, 128)
    assert b2.shape == (128,)

    w, b = head_params[0]
    assert w.shape == (128, D_FEAT)
    assert b.shape == (D_FEAT,)


def test_make_trunk_head_apply():
    key = jax.random.PRNGKey(0)
    trunk_params, head_params, trunk_fn, head_fn = codegen.make_trunk_head(
        NLAT, D_PARAM, D_FEAT, 128, key)

    h = trunk_fn(jnp.zeros(NLAT + D_PARAM))
    assert h.shape == (128,)

    xf = head_fn(jnp.zeros(128))
    assert xf.shape == (D_FEAT,)

    surrogate_apply, _, _ = codegen.rebuild_apply_fns(trunk_params, head_params)
    out = surrogate_apply(jnp.zeros(NLAT), jnp.zeros(D_PARAM))
    assert out.shape == (D_FEAT,)


@pytest.mark.slow
def test_train_surrogate_mse_decreases(toy_sim_budget):
    _, _, initial_mse = codegen.train_surrogate(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=0)
    apply_fn, (trunk_params, head_params), final_mse = codegen.train_surrogate(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=200)

    assert final_mse > 0
    assert final_mse < initial_mse

    out = apply_fn(jnp.zeros(NLAT), jnp.zeros(D_PARAM))
    assert out.shape == (D_FEAT,)


def test_compile_artifact_fields(toy_sim_budget):
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=100)

    assert isinstance(artifact, CompiledArtifact)
    assert artifact.model == "hopf"
    assert artifact.feature == "var"
    assert artifact.nlat == NLAT
    assert artifact.surrogate_mse < float("inf")
    assert artifact.train_sim_budget == N_BUDGET
    assert artifact.trunk_params is not None
    assert artifact.head_params is not None
    assert artifact.trunk_apply is not None
    assert artifact.head_apply is not None


def test_trunk_head_pickle_roundtrip(toy_sim_budget):
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=100)

    u = jnp.zeros(NLAT)
    theta = jnp.zeros(D_PARAM)
    orig_features = artifact(u, theta)

    f = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    fname = f.name
    f.close()
    try:
        save_artifact(artifact, fname)
        loaded = load_artifact(fname)

        loaded_features = loaded(u, theta)
        assert jnp.allclose(orig_features, loaded_features, atol=1e-5)
    finally:
        os.unlink(fname)


def test_rebuild_apply_fns(toy_sim_budget):
    _, (trunk_params, head_params), _ = codegen.train_surrogate(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=50)

    surrogate_apply, trunk_apply, head_apply = codegen.rebuild_apply_fns(
        trunk_params, head_params)

    u = jnp.ones(NLAT)
    theta = jnp.ones(D_PARAM)
    x = jnp.concatenate([u, theta])

    h = trunk_apply(x)
    assert h.shape == (128,)

    xf_direct = head_apply(h)
    assert xf_direct.shape == (D_FEAT,)

    xf_composed = surrogate_apply(u, theta)
    assert xf_composed.shape == (D_FEAT,)

    assert jnp.allclose(xf_direct, xf_composed, atol=1e-6)


def test_different_heads_different_features(toy_sim_budget):
    key = jax.random.PRNGKey(0)
    apply_fn, (trunk_params, _), _ = codegen.train_surrogate(
        "hopf", "var", toy_sim_budget, NLAT, D_FEAT, niter=100, key=key)

    u = jnp.zeros(NLAT)
    theta = jnp.zeros(D_PARAM)
    features_orig = apply_fn(u, theta)

    head_params_new = codegen.init_head(128, D_FEAT, jax.random.PRNGKey(999))
    surr_new, _, _ = codegen.rebuild_apply_fns(trunk_params, head_params_new)

    features_new = surr_new(u, theta)

    assert not jnp.allclose(features_orig, features_new, atol=1e-5)


def test_make_surrogate_apply(toy_nlat, toy_d_param, toy_d_feat):
    """Coverage: make_surrogate_apply returns a JIT-able function."""
    key = jax.random.PRNGKey(0)
    apply_fn, (trunk_params, head_params) = codegen.make_surrogate_apply(
        toy_nlat, toy_d_param, toy_d_feat, hidden=64, key=key)
    u = jnp.zeros(toy_nlat)
    theta = jnp.zeros(toy_d_param)
    out = apply_fn(u, theta)
    assert out.shape == (toy_d_feat,)
    assert trunk_params is not None
    assert head_params is not None
    assert len(trunk_params) == 2
    assert len(head_params) == 1


def test_compile_artifact_trunk_head_callable(toy_sim_budget, toy_nlat,
                                              toy_d_param, toy_d_feat):
    """Coverage: artifact.trunk_apply and head_apply are callable."""
    artifact = codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, toy_nlat, toy_d_feat, niter=50)
    # trunk_apply expects concatenated [u; theta] of dim nlat + d_param
    x = jnp.ones(toy_nlat + toy_d_param)
    h = artifact.trunk_apply(x)
    assert h.shape == (128,)
    xf = artifact.head_apply(h)
    assert xf.shape == (toy_d_feat,)
