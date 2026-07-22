"""tvb-xMax compiler: an "advanced AI math compiler" for virtual brain simulation.

Maps a classical compiler pipeline onto amortized simulation-based inference:

    frontend  -> parse + validate the (model, connectivity, params, feature) spec
    lower     -> cross-code connectivity into the parcellation-invariant latent u
    optimize  -> IR transforms (latent whitening, hetero-param summarization)
    codegen   -> train a neural surrogate that replaces the SDE simulation
    vectorize -> GPU batch eval (pmap+vmap) for ~10^3-10^4x amortized speedup
    posterior -> amortized NPE over IR, batched posterior sampling

The cross-coder makes connectivity parcellation-invariant, so a single
compiled artifact serves any parcellation, any parameters, for free.
"""

from .. import ir
from . import frontend, posterior, swap
from .artifact_cache import ArtifactCache
from ..ir import IRSpec, IRProgram, CompiledArtifact, CompileReport, SimBudget, SwapKind

__all__ = [
    "ir", "frontend", "posterior", "swap",
    "ArtifactCache",
    "IRSpec", "IRProgram", "CompiledArtifact", "CompileReport", "SimBudget", "SwapKind",
]


def __getattr__(name):
    """Load the JAX compiler modules only when their API is requested."""
    if name in {"lower", "optimize", "codegen", "vectorize", "pipeline", "sim_budget",
                "numpy_lower", "numpy_codegen", "numpy_vectorize", "numpy_pipeline"}:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(name)
