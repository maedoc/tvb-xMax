"""tvb-max: an "advanced AI math compiler" for virtual brain simulation.

Parody project that treats amortized simulation-based inference as a
compiler: the cross-coder is the IR, a trained neural surrogate is the
"object code" that replaces the SDE simulation, and GPU batch eval gives
the "nearly infinite speedup".  Because the IR is parcellation-invariant,
swapping connectivity / parameters / models / features is free.

Built on top of vbjax (simulation substrate) and apvbt (cross-coder +
SBI patterns).  See PLAN.md for the full design.
"""

__version__ = "0.1.0"

from . import ir, compiler, surrogates
from .ir import IRSpec, IRProgram, CompiledArtifact, CompileReport, SwapKind

__all__ = [
    "ir", "compiler", "surrogates",
    "IRSpec", "IRProgram", "CompiledArtifact", "CompileReport", "SwapKind",
    "__version__",
]
