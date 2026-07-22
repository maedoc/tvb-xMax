"""Tests for the pipeline orchestrator (tvb_xmax.compiler.pipeline).

Covers the full compile -> run lifecycle through compile_spec, run,
and run_batch.
"""

import jax.numpy as jnp
import pytest

from tvb_xmax import ir
from tvb_xmax.compiler import pipeline
from conftest import NLAT, D_PARAM, D_FEAT


class TestPipeline:
    """Grouped tests sharing a compiled artifact via class-scoped fixture."""

    @pytest.fixture(scope="class")
    def compiled_report(self, toy_crosscoder, toy_sim_budget, toy_nlat,
                        toy_d_feat):
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
            feature="var",
            target="features",
        )
        return pipeline.compile_spec(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )

    # ------------------------------------------------------------------
    # Test 1: compile_spec returns a valid CompileReport
    # ------------------------------------------------------------------
    def test_compile_spec_returns_report(self, compiled_report):
        report = compiled_report
        assert isinstance(report, ir.CompileReport)
        assert isinstance(report.artifact, ir.CompiledArtifact)
        for key in ("frontend", "lower", "optimize", "codegen"):
            assert key in report.stages
        assert jnp.isfinite(report.artifact.surrogate_mse)
        assert jnp.isfinite(report.speedup_vs_sim), \
            f"expected finite speedup, got {report.speedup_vs_sim}"
        bench = report.stages.get("benchmark", {})
        assert isinstance(bench, dict), f"expected dict, got {bench!r}"
        assert bench.get("speedup", -1) > 0

    # ------------------------------------------------------------------
    # Test 2: run returns features with correct shapes
    # ------------------------------------------------------------------
    def test_run_returns_features(self, compiled_report, toy_crosscoder,
                                  toy_nlat, toy_d_param, toy_d_feat):
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
        )
        artifact = compiled_report.artifact
        out = pipeline.run(artifact, spec, toy_crosscoder)

        assert "features" in out
        assert out["features"].shape == (toy_d_feat,)
        assert out["u"].shape == (toy_nlat,)
        assert out["theta"].shape == (toy_d_param,)
        assert jnp.all(jnp.isfinite(out["features"]))
        assert jnp.all(jnp.isfinite(out["u"]))
        assert jnp.all(jnp.isfinite(out["theta"]))

    # ------------------------------------------------------------------
    # Test 3: run_batch vectorizes over multiple specs
    # ------------------------------------------------------------------
    def test_run_batch(self, compiled_report, toy_crosscoder, toy_nlat,
                       toy_d_feat):
        artifact = compiled_report.artifact
        specs = [
            ir.IRSpec(
                model="hopf",
                connectivity=jnp.zeros(toy_nlat),
                connectivity_is_latent=True,
                parameters={"k": k, "D": 0.3},
            )
            for k in (0.1, 0.5, 0.9)
        ]
        out = pipeline.run_batch(artifact, specs, toy_crosscoder)

        assert "features" in out
        assert "U" in out
        assert "Theta" in out
        assert out["features"].shape == (3, toy_d_feat)

    # ------------------------------------------------------------------
    # Test 4: compile_spec accepts an MVN (latent whitening)
    # ------------------------------------------------------------------
    def test_compile_spec_with_mvn(self, toy_crosscoder, toy_sim_budget,
                                   toy_nlat, toy_d_feat, toy_mvn):
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
        )
        report = pipeline.compile_spec(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            mvn=toy_mvn, train_posterior=False,
        )
        assert isinstance(report, ir.CompileReport)
        assert jnp.isfinite(report.artifact.surrogate_mse)

    # ------------------------------------------------------------------
    # Test 5 (slow): compile + run with posterior sampling
    # ------------------------------------------------------------------
    @pytest.mark.slow
    def test_run_with_posterior(self, toy_crosscoder, toy_sim_budget,
                                toy_nlat, toy_d_feat):
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
            feature="var",
            target="both",
            n_posterior=50,
        )
        report = pipeline.compile_spec(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=True, algo="mdn",
        )
        artifact = report.artifact
        out = pipeline.run(artifact, spec, toy_crosscoder)

        assert "posterior" in out
        assert out["posterior"].shape[-1] == D_PARAM
        assert out["posterior"].shape[0] == spec.n_posterior
        assert jnp.all(jnp.isfinite(out["posterior"]))

    # ------------------------------------------------------------------
    # Test 6: resolve_artifact returns CompileReport with artifact
    # ------------------------------------------------------------------
    def test_resolve_artifact(self, compiled_report, toy_crosscoder,
                              toy_sim_budget, toy_nlat, toy_d_feat):
        from tvb_xmax.compiler.artifact_cache import ArtifactCache
        cache = ArtifactCache()
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
            feature="var",
            target="features",
        )
        artifact = pipeline.resolve_artifact(
            spec, toy_crosscoder, cache, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        assert artifact is not None
        assert artifact.model == "hopf"
        assert artifact.feature == "var"

    # ------------------------------------------------------------------
    # Test 7: run_cached returns dict with cache key
    # ------------------------------------------------------------------
    def test_run_cached(self, compiled_report, toy_crosscoder,
                        toy_sim_budget, toy_nlat, toy_d_param, toy_d_feat):
        from tvb_xmax.compiler.artifact_cache import ArtifactCache
        cache = ArtifactCache()
        spec = ir.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
        )
        out = pipeline.run_cached(
            spec, toy_crosscoder, cache, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        assert "features" in out
        assert out["features"].shape == (toy_d_feat,)
        assert "cache" in out
        assert out["cache"] == "miss"
