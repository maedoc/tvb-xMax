"""Wilson-Cowan Neural Mass Dynamics Model

Implements Wilson-Cowan neural mass model as a DynamicsModel plugin.
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


@ModelRegistry.register("wilson-cowan")
class WilsonCowanModel:
    """Wilson-Cowan neural mass dynamics model.

    The Wilson-Cowan model describes the dynamics of interacting excitatory
    and inhibitory neural populations using a sigmoid activation function.

    State variables (per node):
        E: Excitatory population activity
        I: Inhibitory population activity

    Parameters:
        tau_e: Excitatory time constant
        tau_i: Inhibitory time constant
        c_ee: Excitatory-to-excitatory coupling
        c_ei: Inhibitory-to-excitatory coupling
        c_ie: Excitatory-to-inhibitory coupling
        c_ii: Inhibitory-to-inhibitory coupling
        a_e: Excitatory sigmoid slope
        a_i: Inhibitory sigmoid slope
        b_e: Excitatory sigmoid threshold
        b_i: Inhibitory sigmoid threshold
        P: External input (can be heterogeneous)
        D: Noise intensity
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize Wilson-Cowan model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "wilson-cowan"

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        return ParameterSpace(
            parameters={
                "tau_e": ParameterDefinition(
                    name="tau_e",
                    type="float",
                    bounds=(0.01, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Excitatory time constant",
                    default=0.1,
                    hetero=False,
                ),
                "tau_i": ParameterDefinition(
                    name="tau_i",
                    type="float",
                    bounds=(0.01, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Inhibitory time constant",
                    default=0.1,
                    hetero=False,
                ),
                "c_ee": ParameterDefinition(
                    name="c_ee",
                    type="float",
                    bounds=(0.0, 20.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Excitatory-to-excitatory coupling",
                    default=10.0,
                    hetero=False,
                ),
                "c_ei": ParameterDefinition(
                    name="c_ei",
                    type="float",
                    bounds=(0.0, 20.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Inhibitory-to-excitatory coupling",
                    default=10.0,
                    hetero=False,
                ),
                "c_ie": ParameterDefinition(
                    name="c_ie",
                    type="float",
                    bounds=(0.0, 20.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Excitatory-to-inhibitory coupling",
                    default=8.0,
                    hetero=False,
                ),
                "c_ii": ParameterDefinition(
                    name="c_ii",
                    type="float",
                    bounds=(0.0, 20.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Inhibitory-to-inhibitory coupling",
                    default=2.0,
                    hetero=False,
                ),
                "a_e": ParameterDefinition(
                    name="a_e",
                    type="float",
                    bounds=(0.1, 5.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Excitatory sigmoid slope",
                    default=1.2,
                    hetero=False,
                ),
                "a_i": ParameterDefinition(
                    name="a_i",
                    type="float",
                    bounds=(0.1, 5.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Inhibitory sigmoid slope",
                    default=1.0,
                    hetero=False,
                ),
                "b_e": ParameterDefinition(
                    name="b_e",
                    type="float",
                    bounds=(-5.0, 5.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Excitatory sigmoid threshold",
                    default=2.8,
                    hetero=False,
                ),
                "b_i": ParameterDefinition(
                    name="b_i",
                    type="float",
                    bounds=(-5.0, 5.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Inhibitory sigmoid threshold",
                    default=3.0,
                    hetero=False,
                ),
                "P": ParameterDefinition(
                    name="P",
                    type="array",
                    bounds=(0.0, 10.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="External input (heterogeneous)",
                    default=1.0,
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
            state_dim=2,  # E and I per node
            feature_dim=None,  # Depends on feature extractor
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="wilson-cowan",
            version="1.0.0",
            description="Wilson-Cowan neural mass model with excitatory/inhibitory populations",
            parameters=[
                "tau_e",
                "tau_i",
                "c_ee",
                "c_ei",
                "c_ie",
                "c_ii",
                "a_e",
                "a_i",
                "b_e",
                "b_i",
                "P",
                "D",
            ],
            state_dim=2,
            citation="Wilson HR, Cowan JD (1972). Excitatory and inhibitory interactions in localized populations...",
            references=[
                "Deco G, Jirsa VK (2012). Oscillations, resonances and brain dynamics.",
                "Breakspear M (2017). Dynamic models of large-scale brain activity.",
            ],
            author="APVBT Team",
            year=2024,
            tags=["neural-mass", "excitatory-inhibitory", "sigmoid"],
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

    def _sigmoid(self, x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
        """Sigmoid activation function.

        Args:
            x: Input
            a: Slope parameter
            b: Threshold parameter

        Returns:
            Sigmoid activation
        """
        return 1.0 / (1.0 + jnp.exp(-a * (x - b)))

    def _wilson_cowan_dfun(
        self,
        state: jnp.ndarray,
        t: float,
        coupling: jnp.ndarray,
        parameters: tuple,
    ) -> jnp.ndarray:
        """Wilson-Cowan dynamics function (JAX-compatible).

        Args:
            state: State vector (2, n_nodes) - [E, I]
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (tau_e, tau_i, c_ee, c_ei, c_ie, c_ii,
                              a_e, a_i, b_e, b_i, P)

        Returns:
            Derivative (2, n_nodes)
        """
        tau_e, tau_i, c_ee, c_ei, c_ie, c_ii, a_e, a_i, b_e, b_i, P = parameters
        E, I = state

        n_nodes = E.shape[0]

        W_E = coupling * c_ee  # Excitatory coupling matrix
        W_I = coupling * c_ei  # Inhibitory coupling matrix

        I_E = jnp.sum(W_E, axis=1) * E  # Total excitatory input
        I_I = jnp.sum(W_I, axis=1) * I  # Total inhibitory input

        dE = (-E + self._sigmoid(c_ee * I_E - c_ei * I + P, a_e, b_e)) / tau_e
        dI = (-I + self._sigmoid(c_ie * I_E - c_ii * I, a_i, b_i)) / tau_i

        return jnp.array([dE, dI])

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
        det = self._wilson_cowan_dfun(state, 0.0, coupling, parameters)
        noise = jax.random.normal(key, shape=state.shape) * noise_scale
        return state + dt * det + jnp.sqrt(dt) * noise

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run Wilson-Cowan simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'tau_e', 'tau_i', etc.
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        tau_e = parameters["tau_e"]
        tau_i = parameters["tau_i"]
        c_ee = parameters["c_ee"]
        c_ei = parameters["c_ei"]
        c_ie = parameters["c_ie"]
        c_ii = parameters["c_ii"]
        a_e = parameters["a_e"]
        a_i = parameters["a_i"]
        b_e = parameters["b_e"]
        b_i = parameters["b_i"]
        D = parameters["D"]

        P = parameters.get("P", 1.0)
        if jnp.isscalar(P):
            P = jnp.full(coupling_matrix.shape[0], P)

        params = (tau_e, tau_i, c_ee, c_ei, c_ie, c_ii, a_e, a_i, b_e, b_i, P)

        dt = config.dt
        n_steps = int(config.simulation_duration / dt)
        n_windows = config.num_windows
        n_nodes = coupling_matrix.shape[0]

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

        x0 = jnp.zeros((2, n_nodes)) + 0.1
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            features = state_traj[:, 0, :].mean(axis=0)  # Mean of excitatory activity
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * dt,
            parameters=parameters,
            metadata={"model": "wilson-cowan", "config": config},
        )


def make_wilson_cowan(config: Optional[SimulationConfig] = None) -> WilsonCowanModel:
    """Factory function to create Wilson-Cowan model instance.

    Args:
        config: Simulation configuration

    Returns:
        WilsonCowanModel instance
    """
    return WilsonCowanModel(config)


__all__ = ["WilsonCowanModel", "make_wilson_cowan"]
