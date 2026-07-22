import unittest.mock as mock

import pytest
import jax
import jax.numpy as jnp
from tvb_xmax.compiler import vectorize, codegen
from tvb_xmax.ir import CompiledArtifact


def _make_artifact_no_posterior():
    """Build a minimal artifact without posterior for batched_posterior error test."""
    import jax
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    trunk_params = [(jax.random.normal(k1, (16, 128)), jnp.zeros(128)),
                    (jax.random.normal(k2, (128, 128)), jnp.zeros(128))]
    head_params = [(jnp.zeros((128, 8)), jnp.zeros(8))]
    return CompiledArtifact(
        model="hopf", feature="var", nlat=16,
        surrogate_apply=lambda u, t: jnp.zeros(8),
        posterior_sample=None,
        trunk_params=trunk_params,
        head_params=head_params,
    )


@pytest.fixture(scope="module")
def artifact(toy_sim_budget, toy_nlat, toy_d_feat):
    return codegen.compile_artifact(
        "hopf", "var", toy_sim_budget, toy_nlat, toy_d_feat, niter=100
    )


class TestVectorize:
    def test_batched_features_shape(self, artifact, toy_nlat, toy_d_param,
                                    toy_d_feat):
        B = 8
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        U = jax.random.normal(k1, (B, toy_nlat))
        Theta = jax.random.uniform(k2, (B, toy_d_param))
        out = vectorize.batched_features(artifact, U, Theta)
        assert out.shape == (B, toy_d_feat)
        assert jnp.all(jnp.isfinite(out))

    def test_sharded_features_fallback(self, artifact, toy_nlat, toy_d_param):
        B = 8
        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        U = jax.random.normal(k1, (B, toy_nlat))
        Theta = jax.random.uniform(k2, (B, toy_d_param))
        batched = vectorize.batched_features(artifact, U, Theta)
        sharded = vectorize.sharded_features(artifact, U, Theta)
        assert jnp.allclose(batched, sharded)

    def test_benchmark_speedup(self, artifact, toy_nlat, toy_d_param,
                                toy_d_feat):
        B = 8
        key = jax.random.PRNGKey(2)
        k1, k2 = jax.random.split(key)
        U = jax.random.normal(k1, (B, toy_nlat))
        Theta = jax.random.uniform(k2, (B, toy_d_param))

        def sim_fn(u, t):
            return jnp.zeros(toy_d_feat)

        result = vectorize.benchmark_speedup(artifact, sim_fn, U, Theta)
        assert isinstance(result, dict)
        assert "t_sim" in result
        assert "t_surrogate" in result
        assert "speedup" in result
        assert "batch" in result
        assert result["speedup"] > 0
        assert result["batch"] == B

    def test_sharded_features_multi_device(self, artifact, toy_nlat,
                                            toy_d_param, toy_d_feat):
        if jax.device_count() <= 1:
            pytest.skip("only 1 device")
        B = 8
        key = jax.random.PRNGKey(3)
        k1, k2 = jax.random.split(key)
        U = jax.random.normal(k1, (B, toy_nlat))
        Theta = jax.random.uniform(k2, (B, toy_d_param))
        out = vectorize.sharded_features(artifact, U, Theta)
        assert out.shape == (B, toy_d_feat)

    def test_batched_posterior_no_posterior_raises(self):
        """Coverage: batched_posterior raises RuntimeError when posterior is None."""
        art = _make_artifact_no_posterior()
        with pytest.raises(RuntimeError, match="no trained posterior"):
            vectorize.batched_posterior(art, jnp.zeros((1, 8)), 10)

    def test_benchmark_speedup_large_batch(self, artifact, toy_nlat,
                                            toy_d_param, toy_d_feat):
        """Coverage: benchmark_speedup with larger batch hits warmup loop."""
        B = 64
        key = jax.random.PRNGKey(10)
        k1, k2 = jax.random.split(key)
        U = jax.random.normal(k1, (B, toy_nlat))
        Theta = jax.random.uniform(k2, (B, toy_d_param))

        def sim_fn(u, t):
            return jnp.zeros(toy_d_feat)

        result = vectorize.benchmark_speedup(artifact, sim_fn, U, Theta)
        assert result["batch"] == B
        assert result["speedup"] > 0

    def test_benchmark_single_feature_eval(self, artifact, toy_nlat,
                                           toy_d_param, toy_d_feat):
        u = jnp.zeros(toy_nlat)
        theta = jnp.zeros(toy_d_param)

        def sim_fn(input_u, input_theta):
            return jnp.zeros(toy_d_feat)

        result = vectorize.benchmark_single_feature_eval(
            artifact, sim_fn, u, theta, n_warmup=1, n_repeat=2
        )
        assert result["t_sim"] >= 0
        assert result["t_surrogate"] >= 0
        assert result["speedup"] >= 0
        assert result["repeats"] == 2
