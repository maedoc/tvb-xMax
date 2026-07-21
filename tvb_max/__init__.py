"""tvb-max: an "advanced AI math compiler" for virtual brain simulation.

Parody project that treats amortized simulation-based inference as a
compiler: the cross-coder is the IR, a trained neural surrogate is the
"object code" that replaces the SDE simulation, and GPU batch eval gives
the "nearly infinite speedup".  Because the IR is parcellation-invariant,
swapping connectivity / parameters / models / features is free.

vbjax and apvbt are **vendored** under ``vendor/`` (see ``vendor/README.md``);
this import block puts them on ``sys.path`` so ``import vbjax`` /
``import apvbt`` resolve to the pinned copies, not whatever is in
site-packages.  ``sbi`` / ``jax`` / ``torch`` remain pip dependencies.
See PLAN.md for the full design.
"""

import os as _os
import sys as _sys

_VENDOR = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "vendor")
if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)

__version__ = "0.1.0"

from . import ir, compiler, surrogates
from .ir import IRSpec, IRProgram, CompiledArtifact, CompileReport, SwapKind

__all__ = [
    "ir", "compiler", "surrogates",
    "IRSpec", "IRProgram", "CompiledArtifact", "CompileReport", "SwapKind",
    "__version__",
]
