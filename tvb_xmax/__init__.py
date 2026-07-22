"""tvb-xMax: an "advanced AI math compiler" for virtual brain simulation.

Parody project that treats amortized simulation-based inference as a
compiler: the cross-coder is the IR, a trained neural surrogate is the
"object code" that replaces the SDE simulation, and GPU batch eval gives
the "nearly infinite speedup".  Because the IR is parcellation-invariant,
swapping connectivity / parameters / models / features is free.

``vbjax`` is a pip dependency (see ``pyproject.toml``).  ``apvbt`` is still
vendored under ``vendor/apvbt/`` but is not yet imported at runtime; its
extraction is tracked separately as T5.1.  See PLAN.md for the full design.
"""

__version__ = "0.1.0"

from . import ir, surrogates
from .ir import IRSpec, IRProgram, CompiledArtifact, CompileReport, SimBudget, SwapKind

__all__ = [
    "ir", "surrogates",
    "IRSpec", "IRProgram", "CompiledArtifact", "CompileReport", "SimBudget", "SwapKind",
    "__version__",
]
