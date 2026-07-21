"""Brain Dynamics Models

This module provides brain dynamics models for APVBT.
Includes Hopf oscillator and Multi-Population Rate (MPR) models.
"""

from typing import Optional, Callable
from collections.abc import Callable as CallableABC

# Import new plugin-based models
from .models import (
    DynamicsModel,
    ModelRegistry,
    ParameterDefinition,
    ParameterSpace,
    ModelMetadata,
    SimulationConfig,
    SimulationResult,
    ValidationResult,
    DistributionType,
    validate_parameter_space,
    get_model,
    list_models,
)

# Import model implementations
from .models.hopf import HopfModel, make_hopf
from .models.mpr import MPRModel, make_mpr
from .models.wilson_cowan import WilsonCowanModel, make_wilson_cowan
from .models.wong_wang import WongWangModel, make_wong_wang
from .models.kuramoto import KuramotoModel, make_kuramoto
from .models.fitzhugh_nagumo import FitzHughNagumoModel, make_fitzhugh_nagumo


# Legacy compatibility: keep old DynaModel class
class DynaModel:
    def __init__(self, name, dfun, features, dt=1e-3, adhoc=None, key=None):
        import vbjax as vb, jax

        self.name = name
        self.dfun = dfun
        self.dt = dt
        self.g = lambda x, p: p[1]
        _, loop = vb.make_sde(self.dt, self.dfun, self.g, adhoc=adhoc)
        self.loop = loop
        self.features = features
        self.key = key if key is not None else jax.random.PRNGKey(42)

    def run_w(self, w, k, D, nwin, key, eta_mu=1.0, omega_scl=1.0):
        import jax, jax.numpy as jp, vbjax as vb

        w = w / w.max()
        if self.name == "hopf":  # lame but
            import numpy as np

            eta = eta_mu + 0.1 * vb.random.normal(self.key, shape=(w.shape[1],))
            omega = (
                2.0
                * np.pi
                * jax.random.uniform(
                    self.key, shape=(w.shape[1],), minval=0.9, maxval=1.1
                )
                * omega_scl
            )
            p = k, D, eta, omega, w
        else:
            p = k, D, w

        def win(x0, key):
            z = vb.randn(1000, 2, w.shape[0], key=key)
            x = self.loop(x0, z, p)
            return x[-1], self.features(x)

        x0 = jp.zeros((2, w.shape[0])) + jp.c_[0.0, 0.0].T + 1e-4
        keys = jax.random.split(key, nwin)
        x0, xf = jax.lax.scan(win, x0, keys)
        return xf

    def run_ws(self, w, k, D, nwin=10, key=None, use_pmap=True):
        import jax, jax.numpy as jp, vbjax as vb

        key = key if key is not None else jax.random.PRNGKey(42)

        def f(w, key, k, D):
            return self.run_w(w, k, D, nwin, key)

        if use_pmap:
            w_ = w.reshape(
                (
                    vb.cores,
                    -1,
                )
                + w.shape[1:]
            )
            keys_ = jax.random.split(key, w_.shape[:2])
            k_ = jp.array(k).reshape(w_.shape[:2])
            D_ = jp.array(D).reshape(w_.shape[:2])
            xf = jax.pmap(jax.vmap(f))(w_, keys_, k_, D_)
            xf = xf.reshape((-1,) + xf.shape[2:])
        else:
            keys = jax.random.split(key, w.shape[0])
            xf = jax.jit(jax.vmap(f))(w, keys, k, D)
        return xf[:, nwin // 2 :].mean(axis=1)


def make_dynamics(name: str, features=None, key=None):
    """Legacy factory function to create dynamics model for simulations.

    This maintains backward compatibility with existing code.
    For new code, use make_hopf() or make_mpr() directly.

    Args:
        name: Model name ('hopf' or 'mpr')
        features: Feature extraction function (optional)
        key: Random seed (optional)

    Returns:
        Model run_ws function compatible with existing simulation code
    """
    import jax, jax.numpy as jp, vbjax as vb

    key = key if key is not None else jax.random.PRNGKey(42)
    # call make_dyamics('hopf') then pass it as "model" arg to other functions
    if name == "mpr":
        if features is None:

            def features(x):
                return x[:, 1].mean(axis=0)

        model = DynaModel(
            "mpr", mpr.mpr_dfun, features, adhoc=vb.mpr_r_positive, key=key
        )
    elif name == "hopf":
        if features is None:

            def features(x):
                return x[:, 0].var(axis=0)

        model = DynaModel("hopf", hopf.hopf_dfun, features, key=key)
    else:
        raise ValueError("model not implemented")
    return model.run_ws


# Export old dynamics functions for backward compatibility
from . import hopf, mpr


__all__ = [
    # New interfaces
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
    # Model implementations
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
    # Legacy compatibility
    "make_dynamics",
    "DynaModel",
    # Old modules
    "hopf",
    "mpr",
]
