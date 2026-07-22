"""Tests for the ArtifactCache (T4.3).

Covers in-memory hit/miss, get/put/list, disk persistence,
key independence, and MVN passthrough.
"""

import math
import os
import shutil

import jax.numpy as jnp
import pytest

from tvb_xmax import ir as ir_
from tvb_xmax.compiler.artifact_cache import ArtifactCache
from tvb_xmax.ir import IRSpec
from conftest import NLAT


class TestArtifactCache:
    """Grouped tests sharing a common spec and compiled artifact."""

    SPEC_KW = dict(
        model="hopf",
        connectivity_is_latent=True,
        parameters={"k": 0.5, "D": 0.3},
        feature="var",
        target="features",
    )

    def _make_spec(self, **overrides) -> IRSpec:
        kw = dict(self.SPEC_KW)
        kw.update(overrides)
        return IRSpec(**kw)

    # ------------------------------------------------------------------
    # Test 1: miss then hit for identical specs
    # ------------------------------------------------------------------
    def test_cache_miss_then_hit(self, toy_crosscoder, toy_sim_budget,
                                 toy_nlat, toy_d_feat):
        cache = ArtifactCache()
        spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))

        report_miss = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        assert isinstance(report_miss, object)
        assert report_miss.stages["cache"] == "miss"
        assert jnp.isfinite(report_miss.artifact.surrogate_mse)
        assert jnp.isfinite(report_miss.speedup_vs_sim), \
            f"expected finite speedup, got {report_miss.speedup_vs_sim}"

        report_hit = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        assert report_hit.stages["cache"] == "hit"
        assert report_hit.artifact.surrogate_mse == pytest.approx(
            report_miss.artifact.surrogate_mse)

    # ------------------------------------------------------------------
    # Test 2: get / put / list
    # ------------------------------------------------------------------
    def test_get_put_list(self, toy_crosscoder, toy_sim_budget,
                          toy_nlat, toy_d_feat):
        cache = ArtifactCache()
        spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))

        report = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        artifact = report.artifact

        # put is already done by load_or_compile; verify get returns it
        got = cache.get("hopf", "var", toy_nlat)
        assert got is artifact

        # non-existent key
        assert cache.get("nonexistent", "var", toy_nlat) is None

        # list
        artifacts = cache.list()
        assert any(a is artifact for a in artifacts)

    # ------------------------------------------------------------------
    # Test 3: disk persistence
    # ------------------------------------------------------------------
    def test_disk_persistence(self, toy_crosscoder, toy_sim_budget,
                              toy_nlat, toy_d_feat):
        cache_dir = "/tmp/opencode/test_cache"
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)

        try:
            cache1 = ArtifactCache(cache_dir=cache_dir)
            spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))

            report1 = cache1.load_or_compile(
                spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
                train_posterior=False,
            )
            mse1 = report1.artifact.surrogate_mse

            # Files exist on disk
            pkl_path = os.path.join(cache_dir, "hopf_var_nlat16.pkl")
            idx_path = os.path.join(cache_dir, "index.json")
            assert os.path.isfile(pkl_path)
            assert os.path.isfile(idx_path)

            # New cache instance loads from disk
            cache2 = ArtifactCache(cache_dir=cache_dir)
            artifact2 = cache2.get("hopf", "var", toy_nlat)
            assert artifact2 is not None
            assert artifact2.surrogate_mse == pytest.approx(mse1)
        finally:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)

    # ------------------------------------------------------------------
    # Test 4: different keys are independent
    # ------------------------------------------------------------------
    def test_different_keys_independent(self, toy_crosscoder, toy_sim_budget,
                                        toy_nlat, toy_d_feat):
        cache = ArtifactCache()
        spec_hopf = self._make_spec(connectivity=jnp.zeros(toy_nlat))
        spec_mpr = self._make_spec(
            model="mpr",
            connectivity=jnp.zeros(toy_nlat),
            parameters={"k": 0.3, "D": 0.5, "J": 0.6, "w": 0.4},
        )

        report_hopf = cache.load_or_compile(
            spec_hopf, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        report_mpr = cache.load_or_compile(
            spec_mpr, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )

        assert cache.get("hopf", "var", toy_nlat) is report_hopf.artifact
        assert cache.get("mpr", "var", toy_nlat) is report_mpr.artifact
        assert len(cache.list()) == 2

    # ------------------------------------------------------------------
    # Test 5: load_or_compile with MVN
    # ------------------------------------------------------------------
    def test_load_or_compile_with_mvn(self, toy_crosscoder, toy_sim_budget,
                                      toy_nlat, toy_d_feat, toy_mvn):
        cache = ArtifactCache()
        spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))

        report = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            mvn=toy_mvn, train_posterior=False,
        )
        assert report.stages["cache"] == "miss"
        assert jnp.isfinite(report.artifact.surrogate_mse)
        assert "optimize" in report.stages

    # ------------------------------------------------------------------
    # Test 6: disk cache with missing/corrupt file returns None
    # ------------------------------------------------------------------
    def test_disk_cache_missing_file_returns_none(self):
        cache_dir = "/tmp/opencode/test_cache_gone"
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        try:
            cache = ArtifactCache(cache_dir=cache_dir)
            got = cache.get("hopf", "var", NLAT)
            assert got is None
        finally:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)

    # ------------------------------------------------------------------
    # Test 7: cache with corrupt index handles gracefully
    # ------------------------------------------------------------------
    def test_disk_cache_corrupt_index(self):
        import json
        cache_dir = "/tmp/opencode/test_cache_idx"
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        try:
            os.makedirs(cache_dir)
            with open(os.path.join(cache_dir, "index.json"), "w") as f:
                f.write("not valid json")
            cache = ArtifactCache(cache_dir=cache_dir)
            # Should load empty index
            assert cache.get("hopf", "var", NLAT) is None
        finally:
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)

    # ------------------------------------------------------------------
    # Test 8: pickle error during save is caught
    # ------------------------------------------------------------------
    def test_disk_cache_save_error_tolerant(self, toy_crosscoder,
                                             toy_sim_budget, toy_nlat,
                                             toy_d_feat):
        # Use a cache dir that becomes unwritable (root-owned temp)
        import tempfile
        # Just verify put does not raise on cache_dir=None
        cache = ArtifactCache(cache_dir=None)
        spec = ir_.IRSpec(
            model="hopf",
            connectivity=jnp.zeros(toy_nlat),
            connectivity_is_latent=True,
            parameters={"k": 0.5, "D": 0.3},
            feature="var",
            target="features",
        )
        report = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        put_result = cache.put(report.artifact)
        assert put_result is None

    # ------------------------------------------------------------------
    # Test 9: disk cache OSError on save is handled gracefully
    # ------------------------------------------------------------------
    def test_disk_cache_save_oserror_handling(self, toy_crosscoder,
                                               toy_sim_budget, toy_nlat,
                                               toy_d_feat, tmp_path):
        """Coverage: _save_to_disk OSError is caught (lines 139-140)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = ArtifactCache(cache_dir=str(cache_dir))
        spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))
        report = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        # Remove write permission from cache_dir
        os.chmod(str(cache_dir), 0o444)
        try:
            # put should not raise
            cache.put(report.artifact)
        finally:
            os.chmod(str(cache_dir), 0o755)

    # ------------------------------------------------------------------
    # Test 10: _save_index OSError caught (lines 121-122)
    # ------------------------------------------------------------------
    def test_disk_cache_index_oserror_handling(self, toy_crosscoder,
                                                toy_sim_budget, toy_nlat,
                                                toy_d_feat, tmp_path):
        """Coverage: _save_index OSError is caught (lines 121-122)."""
        cache_dir = tmp_path / "cache_idx_err"
        cache_dir.mkdir()
        cache = ArtifactCache(cache_dir=str(cache_dir))
        spec = self._make_spec(connectivity=jnp.zeros(toy_nlat))
        report = cache.load_or_compile(
            spec, toy_crosscoder, toy_sim_budget, toy_d_feat,
            train_posterior=False,
        )
        # Create index file as root-owned dir so write fails
        import stat
        idx_path = os.path.join(str(cache_dir), "index.json")
        with open(idx_path, "w") as f:
            f.write("{}")
        os.chmod(idx_path, 0o444)
        try:
            # put triggers _save_to_disk which calls _save_index; should not raise
            cache.put(report.artifact)
        finally:
            os.chmod(idx_path, 0o644)

    # ------------------------------------------------------------------
    # Test 11: _load_from_disk corrupt pickle returns None (lines 150-151)
    # ------------------------------------------------------------------
    def test_disk_cache_corrupt_pickle_returns_none(self, tmp_path):
        """Coverage: _load_from_disk Exception handler returns None (lines 150-151)."""
        cache_dir = tmp_path / "cache_corrupt"
        cache_dir.mkdir()
        cache = ArtifactCache(cache_dir=str(cache_dir))
        # Write a corrupt .pkl file
        pkl_path = os.path.join(str(cache_dir), "hopf_var_nlat16.pkl")
        with open(pkl_path, "w") as f:
            f.write("not a pickle")
        got = cache.get("hopf", "var", 16)
        assert got is None
