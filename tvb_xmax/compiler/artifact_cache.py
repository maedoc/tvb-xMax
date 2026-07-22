"""ArtifactCache: in-memory + on-disk cache of compiled artifacts.

Key: ``(model, feature, nlat)``.  Enables swap_model / swap_features to
return a cached artifact instead of re-compiling, and prevents redundant
compilation across repeated identical specs.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from typing import Dict, List, Optional, Tuple

from ..ir import CompileReport, CompiledArtifact, IRSpec
from ..surrogates import get_surrogate

log = logging.getLogger(__name__)


def _cache_key(model: str, feature: str, nlat: int) -> str:
    return f"{model}_{feature}_nlat{nlat}"


class ArtifactCache:
    """In-memory + on-disk cache of compiled artifacts.

    Key: ``(model: str, feature: str, nlat: int)``.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._lock = threading.Lock()
        self._cache_dir = cache_dir
        self._mem: Dict[Tuple[str, str, int], CompiledArtifact] = {}
        self._index: Dict[str, dict] = {}

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, model: str, feature: str, nlat: int) -> Optional[CompiledArtifact]:
        """Return cached artifact or ``None``."""
        key = (model, feature, nlat)
        with self._lock:
            artifact = self._mem.get(key)
        if artifact is not None:
            return artifact
        # Try disk
        if self._cache_dir is not None:
            return self._load_from_disk(model, feature, nlat)
        return None

    def put(self, artifact: CompiledArtifact) -> None:
        """Store artifact, keyed by ``(artifact.model, artifact.feature, artifact.nlat)``."""
        key = (artifact.model, artifact.feature, artifact.nlat)
        with self._lock:
            self._mem[key] = artifact
        if self._cache_dir is not None:
            self._save_to_disk(artifact)

    def list(self) -> List[CompiledArtifact]:
        """Return all cached artifacts (in-memory entries)."""
        with self._lock:
            return list(self._mem.values())

    def load_or_compile(self, spec: IRSpec, crosscoder, sim_pairs, d_feat: int,
                        mvn=None, **compile_kw) -> CompileReport:
        """Get from cache or compile + cache.  Returns :class:`CompileReport`.

        If the artifact is found in cache the returned report has
        ``stages["cache"] = "hit"``; otherwise ``stages["cache"] = "miss"``
        and the result is compiled fresh then stored.
        """
        surr = get_surrogate(spec.model)
        nlat = surr.nlat
        cached = self.get(spec.model, spec.feature, nlat)
        if cached is not None:
            return CompileReport(
                artifact=cached,
                stages={"cache": "hit"},
            )
        # Keep the cache importable in the portable install; the JAX pipeline
        # is needed only when a cache miss actually compiles an artifact.
        from . import pipeline
        report = pipeline.compile_spec(
            spec, crosscoder, sim_pairs, d_feat, mvn=mvn, **compile_kw)
        report.stages["cache"] = "miss"
        self.put(report.artifact)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disk_path(self, model: str, feature: str, nlat: int) -> str:
        assert self._cache_dir is not None
        return os.path.join(self._cache_dir, f"{_cache_key(model, feature, nlat)}.pkl")

    def _index_path(self) -> str:
        assert self._cache_dir is not None
        return os.path.join(self._cache_dir, "index.json")

    def _load_index(self) -> None:
        ipath = self._index_path()
        if os.path.isfile(ipath):
            try:
                with open(ipath, "r") as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        try:
            ipath = self._index_path()
            with open(ipath, "w") as f:
                json.dump(self._index, f, indent=2)
        except OSError:
            log.warning("could not write index to %s", ipath)

    def _save_to_disk(self, artifact: CompiledArtifact) -> None:
        key = _cache_key(artifact.model, artifact.feature, artifact.nlat)
        path = self._disk_path(artifact.model, artifact.feature, artifact.nlat)
        try:
            with open(path, "wb") as f:
                pickle.dump(artifact, f)
            self._index[key] = {
                "model": artifact.model,
                "feature": artifact.feature,
                "nlat": artifact.nlat,
                "mse": artifact.surrogate_mse,
                "train_sim_budget": artifact.train_sim_budget,
                "compile_seconds": artifact.compile_seconds,
            }
            self._save_index()
        except OSError:
            log.warning("could not write artifact to %s", path)

    def _load_from_disk(self, model: str, feature: str, nlat: int
                        ) -> Optional[CompiledArtifact]:
        path = self._disk_path(model, feature, nlat)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
        except Exception:
            return None
        # Promote to in-memory
        key = (model, feature, nlat)
        with self._lock:
            self._mem[key] = artifact
        return artifact
