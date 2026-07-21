"""FitzHugh-Nagumo Excitable Media Model

Implements FitzHugh-Nagumo model as a DynamicsModel plugin.
Based on FitzHugh (1961) and Nagumo et al. (1962) as a 2D reduction
of Hodgkin-Huxley neuron model for excitable media dynamics.
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


@ModelRegistry.register("fitzhugh-nagumo")
class FitzHughNagumoModel:
    """FitzHugh-Nagumo excitable media dynamics model.

    The FitzHugh-Nagumo model is a 2D reduction of the Hodgkin-Huxley
    neuron model, capturing excitable and oscillatory behavior with minimal
    parameters.

    State variables (per node):
        v: Membrane potential (fast variable)
        w: Recovery variable (slow variable)

    Parameters:
        a: Shape parameter (cubic nonlinearity)
        b: Time scale ratio (recovery speed)
        tau: Recovery time constant
        I: External input current
        D: Noise intensity

    Dynamics:
        dv/dt = v - v^3/3 - w + I + coupling + noise
        dw/dt = (v + a - b*w) / tau
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize FitzHugh-Nagumo model.

        Args:
            config: Simulation configuration
        """
        self._config = config or SimulationConfig()
        self._key = jax.random.PRNGKey(self._config.seed or 42)

    def get_name(self) -> str:
        """Return unique model identifier."""
        return "fitzhugh-nagumo"

    def get_parameter_space(self) -> ParameterSpace:
        """Define model parameter space with priors."""
        return ParameterSpace(
            parameters={
                "a": ParameterDefinition(
                    name="a",
                    type="float",
                    bounds=(-1.0, 1.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Shape parameter (cubic nonlinearity)",
                    default=0.7,
                    hetero=False,
                ),
                "b": ParameterDefinition(
                    name="b",
                    type="float",
                    bounds=(0.1, 2.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Time scale ratio (recovery speed)",
                    default=0.8,
                    hetero=False,
                ),
                "tau": ParameterDefinition(
                    name="tau",
                    type="float",
                    bounds=(1.0, 20.0),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Recovery time constant",
                    default=12.5,
                    hetero=False,
                ),
                "I": ParameterDefinition(
                    name="I",
                    type="float",
                    bounds=(0.0, 1.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="External input current",
                    default=0.5,
                    hetero=False,
                ),
                "D": ParameterDefinition(
                    name="D",
                    type="float",
                    bounds=(0.0, 0.5),
                    prior_type=DistributionType.UNIFORM,
                    prior_params={},
                    description="Noise intensity",
                    default=0.01,
                    hetero=False,
                ),
            },
            state_dim=2,
            feature_dim=None,
        )

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata."""
        return ModelMetadata(
            name="fitzhugh-nagumo",
            version="1.0.0",
            description="FitzHugh-Nagumo excitable media model as 2D reduction of Hodgkin-Huxley",
            parameters=["a", "b", "tau", "I", "D"],
            state_dim=2,
            citation="FitzHugh R (1961). Impulses and physiological states in theoretical models of nerve membrane. Biophysical Journal 1: 445-466.",
            references=[
                "Nagumo J, Arimoto S, Yoshizawa S (1962). An active pulse transmission line simulating nerve axon. Proceedings of the IRE 50(10): 2061-2070.",
                "Izhikevich EM (2007). Dynamical Systems in Neuroscience: The Geometry of Excitability and Bursting.",
            ],
            author="APVBT Team",
            year=2024,
            tags=["excitable-media", "neuron", "action-potential", "reduction"],
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

    def _fhn_dfun(
        self,
        state: jnp.ndarray,
        t: float,
        coupling: jnp.ndarray,
        parameters: tuple,
    ) -> jnp.ndarray:
        """FitzHugh-Nagumo dynamics function (JAX-compatible).

        Args:
            state: State vector (n_nodes, 2) - [v, w]
            t: Time (not used, autonomous system)
            coupling: Coupling matrix (n_nodes, n_nodes)
            parameters: Tuple of (a, b, tau, I, n_nodes)

        Returns:
            Derivative (n_nodes, 2) - [dv/dt, dw/dt]
        """
        a, b, tau, I, n_nodes = parameters
        v = state[:, 0]
        w = state[:, 1]

        dv = v - (v**3) / 3.0 - w + I
        dw = (v + a - b * w) / tau

        coupling_term = jnp.sum(coupling * state[:, 0][:, None], axis=1)

        dv = dv + coupling_term

        return jnp.stack([dv, dw], axis=1)

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
            state: Current state (n_nodes, 2)
            dt: Time step
            coupling: Coupling matrix
            parameters: Model parameters
            noise_scale: Noise amplitude
            key: Random key for noise

        Returns:
            Next state
        """
        det = self._fhn_dfun(state, 0.0, coupling, parameters)
        noise = jax.random.normal(key, shape=state.shape) * noise_scale
        new_state = state + dt * det + jnp.sqrt(dt) * noise

        return new_state

    def simulate(
        self,
        coupling_matrix: jnp.ndarray,
        parameters: Dict[str, Any],
        feature_extractor: Optional[Callable] = None,
        config: Optional[SimulationConfig] = None,
    ) -> SimulationResult:
        """Run FitzHugh-Nagumo simulation.

        Args:
            coupling_matrix: Coupling/connectivity matrix (n_nodes, n_nodes)
            parameters: Model parameters dict with keys 'a', 'b', 'tau', 'I', 'D'
            feature_extractor: Function to extract features from state trajectory
            config: Simulation configuration (uses default if not provided)

        Returns:
            SimulationResult with features and metadata
        """
        config = config or self._config

        a = parameters["a"]
        b = parameters["b"]
        tau = parameters["tau"]
        I = parameters["I"]
        D = parameters["D"]

        n_nodes = coupling_matrix.shape[0]

        params = (a, b, tau, I, n_nodes)

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

        x0 = jnp.zeros((n_nodes, 2))
        keys = jax.random.split(self._key, n_windows)
        x0, state_traj = jax.lax.scan(simulate_window, x0, keys)

        if feature_extractor is None:
            v = state_traj[:, :, 0]
            spike_rate = jnp.mean(jnp.abs(v) > 1.0)
            v_mean = jnp.mean(v)
            w_mean = jnp.mean(state_traj[:, :, 1])
            features = jnp.array([spike_rate, v_mean, w_mean])
        else:
            features = feature_extractor(state_traj)

        return SimulationResult(
            features=features,
            state_trajectory=state_traj,
            time_points=jnp.arange(state_traj.shape[0]) * dt,
            parameters=parameters,
            metadata={"model": "fitzhugh-nagumo", "config": config},
        )


def make_fitzhugh_nagumo(
    config: Optional[SimulationConfig] = None,
) -> FitzHughNagumoModel:
    """Factory function to create FitzHugh-Nagumo model instance.

    Args:
        config: Simulation configuration

    Returns:
        FitzHughNagumoModel instance
    """
    return FitzHughNagumoModel(config)


__all__ = ["FitzHughNagumoModel", "make_fitzhugh_nagumo"]
