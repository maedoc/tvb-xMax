"""Base classes for surrogate targets.

Mirrors apvbt's ``DynamicsModel`` / ``ModelRegistry`` plugin pattern so a
new literature model is one file + one decorator.  The difference: a
:class:`SurrogateTarget` does *not* run the simulation itself; it only
declares the parameter space and validation, because the simulation is
what the surrogate *replaces*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"


@dataclass
class ParameterDefinition:
    name: str
    type: str                       # 'float' | 'array'
    bounds: Tuple[float, float]
    prior_type: DistributionType = DistributionType.UNIFORM
    prior_params: dict = field(default_factory=dict)
    description: str = ""
    default: float = 0.0
    hetero: bool = False            # heterogeneous across nodes?


@dataclass
class ParameterSpace:
    parameters: Dict[str, ParameterDefinition] = field(default_factory=dict)
    state_dim: int = 0
    feature_dim: int = None


class SurrogateTarget:
    """Base class: declares parameter space + validation for one model."""

    name: str = ""
    nlat: int = 16                  # expected cross-coder latent dim
    citation: str = ""
    param_names: Tuple[str, ...] = ()
    param_bounds: Tuple[Tuple[float, float], ...] = ()

    def get_parameter_space(self) -> ParameterSpace:
        raise NotImplementedError

    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Return a list of error strings (empty == valid)."""
        pspace = self.get_parameter_space()
        errors = []
        for name, pdef in pspace.parameters.items():
            if name not in params:
                continue  # defaults will be used
            v = params[name]
            if pdef.type == "float":
                if v < pdef.bounds[0] or v > pdef.bounds[1]:
                    errors.append(f"{name}={v} out of bounds {pdef.bounds}")
            else:  # array
                import jax.numpy as jnp
                arr = jnp.asarray(v)
                if arr.min() < pdef.bounds[0] or arr.max() > pdef.bounds[1]:
                    errors.append(f"{name} has values out of bounds {pdef.bounds}")
        return errors

    def default_parameters(self) -> Dict[str, Any]:
        return {n: p.default for n, p in self.get_parameter_space().parameters.items()}


class SurrogateRegistry:
    """Decorator-based registry, same pattern as apvbt's ModelRegistry."""

    _targets: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def deco(target_cls):
            cls._targets[name] = target_cls
            target_cls.name = name
            inst = target_cls()
            pspace = inst.get_parameter_space()
            target_cls.param_names = tuple(pspace.parameters.keys())
            target_cls.param_bounds = tuple(
                p.bounds for p in pspace.parameters.values())
            return target_cls
        return deco

    @classmethod
    def get(cls, name: str) -> SurrogateTarget:
        if name not in cls._targets:
            raise KeyError(f"no surrogate target {name!r}; "
                           f"known: {list(cls._targets)}")
        return cls._targets[name]()

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._targets)


# module-level convenience
register = SurrogateRegistry.register


def get_surrogate(name: str) -> SurrogateTarget:
    return SurrogateRegistry.get(name)


def list_surrogates() -> List[str]:
    return SurrogateRegistry.names()
