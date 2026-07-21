"""Intermediate Representation (IR) for the tvb-max compiler.

The IR is the parcellation-invariant latent tensor ``u`` produced by the
cross-coder, plus a normalized parameter vector.  Everything downstream
(surrogate codegen, posterior sampling, GPU vectorization) operates purely
on IR tensors, which is what makes model/parc/param/feature *swapping*
free.

Mapping to a classical compiler:

    source program  ->  IRSpec        (frontend parse target)
    AST              ->  IRProgram     (lowered, ready to optimize)
    object code      ->  CompiledArtifact (surrogate weights + posterior)

All tensors are ``jax.Array`` so the whole IR is JAX-traceable and the
"compiled" artifact runs end-to-end under ``vmap``/``pmap`` on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

__all__ = [
    "IRSpec",
    "IRProgram",
    "CompiledArtifact",
    "CompileReport",
    "SwapKind",
]


@dataclass
class IRSpec:
    """Source-level specification handed to the frontend.

    This is the user's "program": which model, which connectivity, which
    parameters, which features, and what they want back.  Connectivity may
    be a raw matrix in *any* parcellation known to the cross-coder, or a
    pre-encoded latent ``u``.
    """

    model: str                              # 'hopf' | 'mpr' | 'wilson-cowan' | ...
    connectivity: Any                       # (n,n) matrix OR (nlat,) latent u
    connectivity_is_latent: bool = False    # True => connectivity already a u
    parcellation: Optional[str] = None      # required if connectivity is a matrix
    parameters: dict = field(default_factory=dict)   # {k, D, eta, omega, ...}
    feature: str = "var"                    # 'var' | 'fc' | 'bold' | callable name
    target: str = "posterior"               # 'posterior' | 'features' | 'both'
    n_posterior: int = 1000                 # posterior samples requested
    seed: int = 42

    def validate(self) -> None:
        if self.connectivity is None:
            raise ValueError("IRSpec.connectivity is required")
        if not self.connectivity_is_latent and self.parcellation is None:
            raise ValueError(
                "parcellation must be set when connectivity is a raw matrix"
            )
        if self.target not in ("posterior", "features", "both"):
            raise ValueError(f"unknown target {self.target!r}")


@dataclass
class IRProgram:
    """Lowered IR: parcellation-invariant latent + normalized params.

    Produced by :mod:`tvb_max.compiler.lower` from an :class:`IRSpec`.
    This is the form the surrogate ("compiled code") consumes.
    """

    model: str
    u: jax.Array                  # (nlat,) latent connectome code
    theta: jax.Array              # (d_param,) normalized parameter vector in [0,1]
    param_names: Tuple[str, ...]  # ordered names matching theta
    feature: str
    target: str
    n_posterior: int
    seed: int
    # provenance for diagnostics / swap bookkeeping
    parcellation: Optional[str] = None
    param_bounds: Tuple[Tuple[float, float], ...] = ()

    @property
    def d_param(self) -> int:
        return int(self.theta.shape[0])

    @property
    def nlat(self) -> int:
        return int(self.u.shape[0])


@dataclass
class CompiledArtifact:
    """The "object code": a trained surrogate + amortized posterior.

    A single artifact serves every swap of parcellation (via the cross-coder
    latent) and every swap of parameters (just change the input).  Swapping
    the *model* or *feature* selects a different artifact from the registry.
    """

    model: str
    feature: str
    nlat: int
    surrogate_apply: Callable[[jax.Array, jax.Array], jax.Array]  # (u,theta)->xf
    posterior_sample: Optional[Callable[[jax.Array, jax.Array, int], jax.Array]] = None
    param_names: Tuple[str, ...] = ()
    param_bounds: Tuple[Tuple[float, float], ...] = ()
    # diagnostics captured at "compile" (train) time
    surrogate_mse: float = float("inf")
    sbc_score: float = float("nan")     # simulation-based calibration
    c2st_score: float = float("nan")    # classifier two-sample test
    train_sim_budget: int = 0           # how many real sims were spent
    compile_seconds: float = 0.0

    def __call__(self, u: jax.Array, theta: jax.Array) -> jax.Array:
        """Evaluate the compiled surrogate: features = f(latent, params)."""
        return self.surrogate_apply(u, theta)


@dataclass
class CompileReport:
    """Diagnostics returned alongside a :class:`CompiledArtifact`."""

    artifact: CompiledArtifact
    stages: dict = field(default_factory=dict)   # stage_name -> timing/notes
    speedup_vs_sim: float = float("inf")        # t_sim / t_surrogate (batch)


class SwapKind:
    """Tags for the free swaps the compiler exposes."""

    PARCELLATION = "parcellation"   # re-encode connectivity, reuse artifact
    PARAMETERS = "parameters"       # change theta, reuse artifact
    MODEL = "model"                 # select different artifact from registry
    FEATURES = "features"           # select different artifact head
