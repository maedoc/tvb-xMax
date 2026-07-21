"""Multi-Population Rate (MPR) Dynamics Model

Implements the MPR model as a DynamicsModel plugin.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from . import (
    DynamicsModel,
    ModelRegistry,
    ParameterSpace,
    ParameterDefinition,
    ModelMetadata,
    SimulationConfig,
    SimulationResult,
    ValidationResult,
    DistributionType,
)

try:
    import vbjax as vb
except ImportError:
    vb = None


@ModelRegistry.register("mpr")
class MPRModel:
    """Multi-Population Rate (MPR) dynamics model.

    The MPR model describes the dynamics of neural populations using
    rate equations with coupling.

    Parameters:
        k: Coupling strength (scalar)
        D: Noise intensity (scalar)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize MPR model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "mpr"

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        return ParameterSpace(
            parameters={
                "k": ParameterDefinition(
                    name="k",
                    type="float",
                    bounds=(0.0, 1.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Coupling strength",
                    default=0.15,
                    hetero=False,
                ),
                "D": ParameterDefinition(
                    name="D",
                    type="float",
                    bounds=(0.0, 1.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Noise intensity",
                    default=0.4,
                    hetero=False,
                ),
            },
            state_dim=2,  # Rate (r) and variance (V)
            feature_dim=None,  # Depends on feature extractor
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="mpr",
            version="1.0.0",
            description="Multi-Population Rate model for neural population dynamics",
            parameters=["k", "D"],
            state_dim=1,
            citation="Deco G, Kringelbach ML, et al. (2014). How structure shapes dynamics.",
            references=[
                "Deco G, Jirsa VK, et al. (2008). The dynamic brain...",
            ],
            author="APVBT Team",
            year=2024,
            tags=["rate", "population", "coupling"],
        )

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate parameter values are within valid ranges."""
        param_space = self.get_parameter_space()
        errors = []
        warnings = []

        for name, param_def in param_space.parameters.items():
            if name not in parameters:
                errors.append(f"Missing required parameter: {name}")
                continue

            value = parameters[name]

            if param_def.hetero:
                if jnp.isscalar(value):
                    warnings.append(
                        f"Parameter {name} is heterogeneous but scalar provided"
                    )
                else:
                    if jnp.any(value < param_def.bounds[0]) or jnp.any(
                        value > param_def.bounds[1]
                    ):
                        errors.append(
                            f"Parameter {name} values out of bounds {param_def.bounds}"
                        )
            else:
                if value < param_def.bounds[0] or value > param_def.bounds[1]:
                    errors.append(
                        f"Parameter {name} = {value} out of bounds {param_def.bounds}"
                    )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def get_default_config(self) -> SimulationConfig:
        """Get default simulation configuration."""
        return self._config

    def _mpr_dfun(
        self, state: jnp.ndarray, t: float, coupling: jnp.ndarray, parameters: tuple
    ) -> jnp.ndarray:
        """MPR dynamics function (JAX-compatible).

        Args:
            state: State vector (n_nodes,)
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (k, D, w)

        Returns:
            Derivative (n_nodes,)
        """
        if vb is None:
            raise ImportError("vbjax is required for MPR model simulations")

        k, D, w = parameters
        return vb.mpr_dfun(state, (k * coupling @ state, 0), vb.mpr_default_theta)

    def _g_fun(self, x, p):
        """Diffusion function."""
        return p[1]  # D

    def _make_simulation_loop(self, coupling_matrix: jnp.ndarray, parameters: tuple):
        """Create simulation loop using vbjax."""
        if vb is None:
            raise ImportError("vbjax is required for MPR model simulations")

        dt = self._config.dt

        k, D, w = parameters

        def mpr_dfun_wrapper(state, p):
            r, V = state
            return vb.mpr_dfun(state, (k * (w @ r), 0), vb.mpr_default_theta)

        def g_fun_wrapper(x, p):
            return D

        _, loop = vb.make_sde(
            dt, mpr_dfun_wrapper, g_fun_wrapper, adhoc=vb.mpr_r_positive
        )
        return loop

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run MPR simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'k', 'D'
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        k = float(parameters.get("k", 0.15))
        D = float(parameters.get("D", 0.4))

        w = coupling_matrix / coupling_matrix.max()
        params = (k, D, w)

        n_windows = config.num_windows
        n_steps = int(config.simulation_duration / config.dt)

        def simulate_window(x0, key):
            z = vb.randn(n_steps, 2, coupling_matrix.shape[0], key=key)
            loop = self._make_simulation_loop(w, params)
            x = loop(x0, z, params)
            return x[-1], x

        x0 = jnp.zeros((2, coupling_matrix.shape[0])) + jnp.c_[0.0, 0.0].T + 1e-4
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            features = state_traj.mean(axis=0)  # Mean activity
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * config.dt,
            parameters=parameters,
            metadata={"model": "mpr", "config": config},
        )


def make_mpr(config: Optional[SimulationConfig] = None) -> MPRModel:
    """Factory function to create MPR model instance.

    Args:
        config: Simulation configuration

    Returns:
        MPRModel instance
    """
    return MPRModel(config)


__all__ = ["MPRModel", "make_mpr"]
