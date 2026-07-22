"""Surrogate targets: one per brain dynamics model in the literature.

Each surrogate declares the model's :class:`ParameterSpace` (mirroring
apvbt's ``DynamicsModel`` plugin interface) and the expected latent
dimension ``nlat``.  The actual surrogate *network* is trained by
:mod:`tvb_xmax.compiler.codegen` on a one-time simulation budget; this
module only holds the metadata + validation needed by the frontend.

Registering a new literature model = adding one file here, exactly like
apvbt's ``ModelRegistry``.  The Discord "openclaw agents" each own one
of these and compete to produce the best-calibrated artifact for it.
"""

from .base import (
    SurrogateTarget,
    SurrogateRegistry,
    ParameterSpace,
    ParameterDefinition,
    DistributionType,
    get_surrogate,
    list_surrogates,
    register,
)
from .hopf import HopfSurrogate
from .mpr import MPRSurrogate
from .wilson_cowan import WilsonCowanSurrogate
from .wong_wang import WongWangSurrogate
from .kuramoto import KuramotoSurrogate
from .fitzhugh_nagumo import FitzHughNagumoSurrogate

__all__ = [
    "SurrogateTarget", "SurrogateRegistry", "ParameterSpace",
    "ParameterDefinition", "DistributionType",
    "get_surrogate", "list_surrogates", "register",
    "HopfSurrogate", "MPRSurrogate", "WilsonCowanSurrogate",
    "WongWangSurrogate", "KuramotoSurrogate", "FitzHughNagumoSurrogate",
]
