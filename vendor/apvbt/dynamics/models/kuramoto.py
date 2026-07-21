"""Kuramoto Phase Oscillator Dynamics Model

Implements Kuramoto phase oscillator model as a DynamicsModel plugin.
Based on Kuramoto (1975) model for synchronizing coupled oscillators.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Callable

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


@ModelRegistry.register("kuramoto")
class KuramotoModel:
    """Kuramoto phase oscillator dynamics model.

    The Kuramoto model describes the dynamics of coupled phase oscillators,
    capturing synchronization behavior in systems of interacting rhythmic units.

    State variables (per node):
        theta: Phase angle (radians, typically in [-π, π])

    Parameters:
        K: Coupling strength (global synchronization strength)
        omega: Natural frequencies (heterogeneous across nodes)
        theta_0: Initial phases (optional, default random uniform)
        D: Noise intensity

    Dynamics:
        dθ_i/dt = ω_i + (K/N) * Σ_j sin(θ_j - θ_i) + ξ(t)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize Kuramoto model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "kuramoto"

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        return ParameterSpace(
            parameters={
                "K": ParameterDefinition(
                    name="K",
                    type="float",
                    bounds=(0.0, 10.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Coupling strength",
                    default=1.0,
                    hetero=False,
                ),
                "omega": ParameterDefinition(
                    name="omega",
                    type="array",
                    bounds=(-10.0, 10.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Natural frequencies (rad/s)",
                    default=1.0,
                    hetero=True,
                ),
                "theta_0": ParameterDefinition(
                    name="theta_0",
                    type="array",
                    bounds=(-jnp.pi, jnp.pi),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Initial phases (rad)",
                    default=0.0,
                    hetero=True,
                ),
                "D": ParameterDefinition(
                    name="D",
                    type="float",
                    bounds=(0.0, 1.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Noise intensity",
                    default=0.1,
                    hetero=False,
                ),
            },
            state_dim=1,  # theta per node
            feature_dim=None,  # Depends on feature extractor
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="kuramoto",
            version="1.0.0",
            description="Kuramoto phase oscillator model for synchronizing coupled oscillators",
            parameters=["K", "omega", "theta_0", "D"],
            state_dim=1,
            citation="Kuramoto Y (1975). Self-entrainment of a population of coupled non-linear oscillators.",
            references=[
                "Strogatz SH (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators.",
                "Acebrón JA et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena.",
            ],
            author="APVBT Team",
            year=2024,
            tags=["phase-oscillator", "synchronization", "coupling"],
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

    def _compute_order_parameter(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute Kuramoto order parameter.

        Args:
            theta: Phase angles (n_nodes,)

        Returns:
            Complex order parameter r*e^(i*psi)
        """
        z = jnp.sum(jnp.exp(1j * theta)) / len(theta)
        return z

    def _kuramoto_dfun(
        self,
        state: jnp.ndarray,
        t: float,
        coupling: jnp.ndarray,
        parameters: tuple,
    ) -> jnp.ndarray:
        """Kuramoto dynamics function (JAX-compatible).

        Args:
            state: State vector (n_nodes,) - [theta]
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (K, omega, n_nodes)

        Returns:
            Derivative (n_nodes,)
        """
        K, omega, n_nodes = parameters
        theta = state

        phase_diff = theta[None, :] - theta[:, None]
        coupling_sum = jnp.sum(coupling * jnp.sin(phase_diff), axis=1)
        dtheta = omega + (K / n_nodes) * coupling_sum

        return dtheta

    def _euler_step(
        self,
        state: jnp.ndarray,
        dt: float,
        coupling: jnp.ndarray,
        parameters: tuple,
        noise_scale: float,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single Euler integration step with noise.

        Args:
            state: Current state (n_nodes,)
            dt: Time step
            coupling: Coupling matrix
            parameters: Model parameters
            noise_scale: Noise amplitude
            key: Random key for noise

        Returns:
            Next state
        """
        det = self._kuramoto_dfun(state, 0.0, coupling, parameters)
        noise = jax.random.normal(key, shape=state.shape) * noise_scale
        new_state = state + dt * det + jnp.sqrt(dt) * noise

        return jnp.mod(new_state + jnp.pi, 2 * jnp.pi) - jnp.pi

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run Kuramoto simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'K', 'omega', 'theta_0', 'D'
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        K = parameters["K"]
        D = parameters["D"]

        omega = parameters.get("omega", 1.0)
        if jnp.isscalar(omega):
            omega = jnp.full(coupling_matrix.shape[0], omega)

        theta_0 = parameters.get("theta_0")
        n_nodes = coupling_matrix.shape[0]

        if theta_0 is None or jnp.all(theta_0 == 0.0):
            key_init, self._key = jax.random.split(self._key)
            theta_0 = jax.random.uniform(
                key_init, shape=(n_nodes,), minval=-jnp.pi, maxval=jnp.pi
            )

        params = (K, omega, n_nodes)

        dt = config.dt
        n_steps = int(config.simulation_duration / dt)
        n_windows = config.num_windows

        noise_scale = jnp.sqrt(D)

        def simulate_window(x0, key):
            def step(state, t_key):
                k1, k2 = jax.random.split(t_key)
                x_next = self._euler_step(
                    state, dt, coupling_matrix, params, noise_scale, k1
                )
                return x_next, x_next

            keys = jax.random.split(key, n_steps)
            x0, state_traj = jax.lax.scan(step, x0, keys)
            return x0, state_traj

        x0 = theta_0
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            z = self._compute_order_parameter(state_traj[-1, :])
            r = jnp.abs(z)
            phase_var = jnp.var(state_traj[-1, :])
            features = jnp.array([r, phase_var])
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * dt,
            parameters=parameters,
            metadata={"model": "kuramoto", "config": config},
        )


def make_kuramoto(config: Optional[SimulationConfig] = None) -> KuramotoModel:
    """Factory function to create Kuramoto model instance.

    Args:
        config: Simulation configuration

    Returns:
        KuramotoModel instance
    """
    return KuramotoModel(config)


__all__ = ["KuramotoModel", "make_kuramoto"]
