"""Wong-Wang Neural Mass Dynamics Model

Implements Wong-Wang neural mass model as a DynamicsModel plugin.
Based on Wong & Wang (2006) reduced spiking neuron model with NMDA receptor kinetics.
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


@ModelRegistry.register("wong-wang")
class WongWangModel:
    """Wong-Wang neural mass dynamics model.

    The Wong-Wang model is a reduced spiking neuron model with NMDA receptor kinetics,
    commonly used for simulating resting-state brain dynamics and generating BOLD-like signals.

    State variables (per node):
        S: Synaptic gating variable (NMDA receptor activity, 0-1)
        H: Total input current

    Parameters:
        J: Synaptic efficacy (excitatory coupling strength)
        a: NMDA receptor activation rate
        b: NMDA receptor deactivation rate
        D: Noise amplitude
        tau: NMDA time constant
        I0: External input current
        gamma: Global scaling factor
        sigma: Noise scaling factor

    Dynamics:
        dS/dt = -S/tau + gamma * (1-S) * H
        dH/dt = -H + a * J * S * I0 + noise
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize Wong-Wang model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "wong-wang"

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        return ParameterSpace(
            parameters={
                "J": ParameterDefinition(
                    name="J",
                    type="float",
                    bounds=(0.1, 5.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Synaptic efficacy (mV)",
                    default=0.27,
                    hetero=False,
                ),
                "a": ParameterDefinition(
                    name="a",
                    type="float",
                    bounds=(0.27, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="NMDA receptor activation rate (ms^-1)",
                    default=0.27,
                    hetero=False,
                ),
                "b": ParameterDefinition(
                    name="b",
                    type="float",
                    bounds=(0.004, 0.01),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="NMDA receptor deactivation rate (ms^-1)",
                    default=0.005,
                    hetero=False,
                ),
                "D": ParameterDefinition(
                    name="D",
                    type="float",
                    bounds=(0.0, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Noise amplitude",
                    default=0.1,
                    hetero=False,
                ),
                "tau": ParameterDefinition(
                    name="tau",
                    type="float",
                    bounds=(100.0, 200.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="NMDA time constant (ms)",
                    default=100.0,
                    hetero=False,
                ),
                "I0": ParameterDefinition(
                    name="I0",
                    type="float",
                    bounds=(0.3, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="External input current (nA)",
                    default=0.33,
                    hetero=False,
                ),
                "gamma": ParameterDefinition(
                    name="gamma",
                    type="float",
                    bounds=(0.1, 2.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Global scaling factor",
                    default=1.0,
                    hetero=False,
                ),
                "sigma": ParameterDefinition(
                    name="sigma",
                    type="float",
                    bounds=(0.0, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Noise scaling factor",
                    default=0.1,
                    hetero=False,
                ),
            },
            state_dim=2,  # S and H per node
            feature_dim=None,  # Depends on feature extractor
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="wong-wang",
            version="1.0.0",
            description="Wong-Wang reduced spiking neuron model with NMDA receptor kinetics",
            parameters=["J", "a", "b", "D", "tau", "I0", "gamma", "sigma"],
            state_dim=2,
            citation="Wong KF, Wang XJ (2006). A recurrent network mechanism...",
            references=[
                "Deco G, Jirsa VK (2012). Oscillations, resonances and brain dynamics.",
                "Honey CJ et al. (2009). Predicting human resting-state functional connectivity...",
            ],
            author="APVBT Team",
            year=2024,
            tags=["neural-mass", "nmda", "resting-state", "bold"],
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

    def _wong_wang_dfun(
        self,
        state: jnp.ndarray,
        t: float,
        coupling: jnp.ndarray,
        parameters: tuple,
    ) -> jnp.ndarray:
        """Wong-Wang dynamics function (JAX-compatible).

        Args:
            state: State vector (2, n_nodes) - [S, H]
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (J, a, b, tau, I0, gamma)

        Returns:
            Derivative (2, n_nodes)
        """
        J, a, b, tau, I0, gamma = parameters
        S, H = state

        n_nodes = S.shape[0]

        I_coupling = coupling @ S
        H_total = H + I_coupling * J * I0

        dS = -S / tau + gamma * (1.0 - S) * a * H_total
        dH = -H / tau + a * J * S * I0 - b * H

        return jnp.array([dS, dH])

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
            state: Current state (2, n_nodes)
            dt: Time step
            coupling: Coupling matrix
            parameters: Model parameters
            noise_scale: Noise amplitude
            key: Random key for noise

        Returns:
            Next state
        """
        det = self._wong_wang_dfun(state, 0.0, coupling, parameters)
        noise = jax.random.normal(key, shape=state.shape) * noise_scale
        return state + dt * det + jnp.sqrt(dt) * noise

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run Wong-Wang simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'J', 'a', 'b', 'D', 'tau', 'I0', 'gamma', 'sigma'
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        J = parameters["J"]
        a = parameters["a"]
        b = parameters["b"]
        D = parameters["D"]
        tau = parameters["tau"]
        I0 = parameters["I0"]
        gamma = parameters["gamma"]
        sigma = parameters["sigma"]

        params = (J, a, b, tau, I0, gamma)

        dt = config.dt
        n_steps = int(config.simulation_duration / dt)
        n_windows = config.num_windows
        n_nodes = coupling_matrix.shape[0]

        noise_scale = sigma * jnp.sqrt(D)

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

        x0 = jnp.zeros((2, n_nodes)) + 0.1
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            features = state_traj[0, :, :].var(
                axis=0
            )  # Variance of synaptic activity (S)
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * dt,
            parameters=parameters,
            metadata={"model": "wong-wang", "config": config},
        )


def make_wong_wang(config: Optional[SimulationConfig] = None) -> WongWangModel:
    """Factory function to create Wong-Wang model instance.

    Args:
        config: Simulation configuration

    Returns:
        WongWangModel instance
    """
    return WongWangModel(config)


__all__ = ["WongWangModel", "make_wong_wang"]
