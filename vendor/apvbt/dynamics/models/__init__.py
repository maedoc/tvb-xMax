"""Dynamics Model Interface and Registry

This module provides the DynamicsModel interface and ModelRegistry for plugin discovery,
enabling pluggable brain dynamics models in APVBT.
"""

from typing import Protocol, Dict, Any, Optional, List, Type, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class DistributionType(Enum):
    """Types of prior distributions for parameters."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    BETA = "beta"
    LOGNORMAL = "lognormal"
    GAMMA = "gamma"
    CUSTOM = "custom"


@dataclass
class ParameterDefinition:
    """Definition of a model parameter with prior distribution."""

    name: str
    type: str  # 'float', 'int', 'array'
    bounds: tuple[float, float]  # Min, max values
    prior_type: DistributionType
    prior_params: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    default: Optional[float] = None
    hetero: bool = False  # Can be heterogeneous across nodes


@dataclass
class ParameterSpace:
    """Parameter space definition for a dynamics model."""

    parameters: Dict[str, ParameterDefinition]
    state_dim: Optional[int] = None  # State dimensionality, inferred if None
    feature_dim: Optional[int] = None  # Feature output dimension


@dataclass
class ModelMetadata:
    """Metadata about a dynamics model."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    parameters: List[str] = field(default_factory=list)
    state_dim: Optional[int] = None
    citation: str = ""
    references: List[str] = field(default_factory=list)
    author: str = ""
    year: int = 2024
    tags: List[str] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Configuration for running simulations."""

    dt: float = 1e-3
    simulation_duration: float = 1.0
    transient_duration: float = 0.0
    num_windows: int = 10
    use_pmap: bool = True
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Result of running a simulation."""

    features: Any  # Extracted features
    state_trajectory: Optional[Any] = None  # Full state time series
    time_points: Optional[Any] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DynamicsModel(Protocol):
    """Interface for brain dynamics models."""

    def get_name(self) -> str:
        """Return unique model identifier."""
        ...

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        ...

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        ...

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate parameter values are within valid ranges."""
        ...

    def get_default_config(self) -> SimulationConfig:
        """Get default simulation configuration."""
        ...


class ModelRegistry:
    """Registry for dynamics models with plugin discovery."""

    _models: Dict[str, Type[DynamicsModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator for registering dynamics models.

        Usage:
            @ModelRegistry.register('hopf')
            class HopfModel:
                ...
        """

        def decorator(model_class: Type[DynamicsModel]) -> Type[DynamicsModel]:
            cls._models[name] = model_class
            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[DynamicsModel]:
        """Get model class by name."""
        if name not in cls._models:
            available = ", ".join(cls.list_available())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if model is registered."""
        return name in cls._models

    @classmethod
    def get_all_metadata(cls) -> Dict[str, ModelMetadata]:
        """Get metadata for all registered models."""
        metadata = {}
        for name, model_class in cls._models.items():
            try:
                metadata[name] = model_class().get_metadata()
            except Exception:
                pass
        return metadata

    @classmethod
    def get_metadata(cls, name: str) -> ModelMetadata:
        """Get metadata for a specific model."""
        if name not in cls._models:
            available = ", ".join(cls.list_available())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name]().get_metadata()


def validate_parameter_space(param_space: ParameterSpace) -> ValidationResult:
    """Validate a parameter space definition.

    Args:
        param_space: Parameter space to validate

    Returns:
        ValidationResult with validation status and any errors/warnings
    """
    errors = []
    warnings = []

    if not param_space.parameters:
        errors.append("Parameter space must have at least one parameter")

    for name, param_def in param_space.parameters.items():
        if not param_def.name:
            errors.append(f"Parameter {name} must have a name")

        if param_def.bounds[0] >= param_def.bounds[1]:
            errors.append(f"Parameter {name}: invalid bounds {param_def.bounds}")

        if param_def.default is not None:
            if (
                param_def.default < param_def.bounds[0]
                or param_def.default > param_def.bounds[1]
            ):
                errors.append(
                    f"Parameter {name}: default {param_def.default} outside bounds"
                )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def get_model(model_name: str) -> Type[DynamicsModel]:
    """Convenience function to get a model by name.

    Args:
        model_name: Name of the model

    Returns:
        Model class

    Raises:
        ValueError: If model not found
    """
    return ModelRegistry.get(model_name)


def list_models() -> List[str]:
    """Convenience function to list all available models."""
    return ModelRegistry.list_available()


__all__ = [
    "DynamicsModel",
    "ModelRegistry",
    "ParameterDefinition",
    "ParameterSpace",
    "ModelMetadata",
    "SimulationConfig",
    "SimulationResult",
    "ValidationResult",
    "DistributionType",
    "validate_parameter_space",
    "get_model",
    "list_models",
    "HopfModel",
    "make_hopf",
    "MPRModel",
    "make_mpr",
    "WilsonCowanModel",
    "make_wilson_cowan",
    "WongWangModel",
    "make_wong_wang",
    "KuramotoModel",
    "make_kuramoto",
    "FitzHughNagumoModel",
    "make_fitzhugh_nagumo",
]


from .hopf import HopfModel, make_hopf
from .mpr import MPRModel, make_mpr
from .wilson_cowan import WilsonCowanModel, make_wilson_cowan
from .wong_wang import WongWangModel, make_wong_wang
from .kuramoto import KuramotoModel, make_kuramoto
from .fitzhugh_nagumo import FitzHughNagumoModel, make_fitzhugh_nagumo
