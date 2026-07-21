"""Hopf Oscillator Dynamics Model

Implements the Hopf oscillator model as a DynamicsModel plugin.
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


@ModelRegistry.register("hopf")
class HopfModel:
    """Hopf oscillator dynamics model with supercritical bifurcation.

    The Hopf oscillator exhibits a supercritical bifurcation at eta=0:
    - eta < 0: Stable fixed point
    - eta > 0: Limit cycle oscillation

    Parameters:
        k: Coupling strength (scalar)
        D: Noise intensity (scalar)
        eta: Bifurcation parameter (can be heterogeneous across nodes)
        omega: Natural frequency (can be heterogeneous across nodes)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize Hopf model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "hopf"

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
                "eta": ParameterDefinition(
                    name="eta",
                    type="array",
                    bounds=(-2.0, 2.0),
                    prior_type=DistributionType.NORMAL,
                    prior_params={"mean": 1.0, "std": 0.1},
                    description="Bifurcation parameter (heterogeneous)",
                    default=1.0,
                    hetero=True,
                ),
                "omega": ParameterDefinition(
                    name="omega",
                    type="array",
                    bounds=(0.9 * jnp.pi, 1.1 * jnp.pi),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Natural frequency (heterogeneous)",
                    default=jnp.pi,
                    hetero=True,
                ),
            },
            state_dim=2,  # Complex state (real, imag)
            feature_dim=None,  # Depends on feature extractor
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="hopf",
            version="1.0.0",
            description="Hopf oscillator model with supercritical bifurcation",
            parameters=["k", "D", "eta", "omega"],
            state_dim=2,
            citation="Deco G, Kringelbach ML, et al. (2017). Dynamical Brain Biomarkers.",
            references=[
                "Deco G, Jirsa VK, et al. (2009). Spontaneous brain fluctuations arise...",
            ],
            author="APVBT Team",
            year=2024,
            tags=["oscillator", "bifurcation", "phase"],
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

    def _hopf_dfun(
        self, state: jnp.ndarray, t: float, coupling: jnp.ndarray, parameters: tuple
    ) -> jnp.ndarray:
        """Hopf oscillator dynamics function (JAX-compatible).

        Args:
            state: State vector (2, n_nodes) - real and imaginary parts
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (k, D, eta, omega, w)

        Returns:
            Derivative (2, n_nodes)
        """
        k, D, eta, omega, w = parameters
        y0, y1 = state

        cfun = vb.make_diff_cfun(jnp.array(w))
        Ic = (cfun(y0),)

        dy0 = y0 * (eta - y0**2 - y1**2) - omega * y1 + 100.0 * k * Ic[0]
        dy1 = y1 * (eta - y0**2 - y1**2) + omega * y1

        return jnp.array([dy0, dy1])

    def _g_fun(self, x, p):
        """Diffusion function."""
        return p[1]  # D

    def _make_simulation_loop(self, coupling_matrix: jnp.ndarray, parameters: tuple):
        """Create simulation loop using vbjax."""
        if vb is None:
            raise ImportError("vbjax is required for Hopf model simulations")

        dt = self._config.dt

        def hopf_dfun_wrapper(ys, p):
            y0, y1 = ys
            cfun = vb.make_diff_cfun(jnp.array(p[4]))
            Ic = (cfun(y0),)
            dy0 = y0 * (p[2] - y0**2 - y1**2) - p[3] * y1 + 100.0 * p[0] * Ic[0]
            dy1 = y1 * (p[2] - y0**2 - y1**2) + p[3] * y1
            return jnp.array([dy0, dy1])

        def g_fun_wrapper(x, p):
            return p[1]

        _, loop = vb.make_sde(dt, hopf_dfun_wrapper, g_fun_wrapper)
        return loop

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run Hopf oscillator simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'k', 'D', 'eta', 'omega'
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        k = parameters["k"]
        D = parameters["D"]

        eta = parameters.get("eta", 1.0)
        if jnp.isscalar(eta):
            eta = jnp.full(coupling_matrix.shape[0], eta)

        omega = parameters.get("omega", jnp.pi)
        if jnp.isscalar(omega):
            omega = jnp.full(coupling_matrix.shape[0], omega)

        w = coupling_matrix / coupling_matrix.max()

        params = (k, eta, omega, D, w)

        n_windows = config.num_windows
        n_steps = int(config.simulation_duration / config.dt)

        def simulate_window(x0, key):
            z = vb.randn(n_steps, 2, coupling_matrix.shape[0], key=key)
            loop = self._make_simulation_loop(w, params)
            x = loop(x0, z, params)
            return x[-1], x

        x0 = jnp.zeros((2, coupling_matrix.shape[0])) + 1e-4
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            features = state_traj[:, 0].var(axis=0)  # Variance of real part
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * config.dt,
            parameters=parameters,
            metadata={"model": "hopf", "config": config},
        )


def make_hopf(config: Optional[SimulationConfig] = None) -> HopfModel:
    """Factory function to create Hopf model instance.

    Args:
        config: Simulation configuration

    Returns:
        HopfModel instance
    """
    return HopfModel(config)


__all__ = ["HopfModel", "make_hopf"]
