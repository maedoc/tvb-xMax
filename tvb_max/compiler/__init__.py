"""tvb-max compiler: an "advanced AI math compiler" for virtual brain simulation.

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
from . import frontend, lower, optimize, codegen, vectorize, posterior, pipeline, swap
from ..ir import IRSpec, IRProgram, CompiledArtifact, CompileReport, SwapKind

__all__ = [
    "ir", "frontend", "lower", "optimize", "codegen", "vectorize",
    "posterior", "pipeline", "swap",
    "IRSpec", "IRProgram", "CompiledArtifact", "CompileReport", "SwapKind",
]
