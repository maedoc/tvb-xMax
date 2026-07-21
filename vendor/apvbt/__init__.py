"""APVBT - Amortizing Personalization in Virtual Brain Twins

This package provides tools for:
- Dataset loading with pluggable loaders
- Brain dynamics models with plugin architecture
- Simulation-based inference (SBI)
- Cross-coder training for connectome parcellation mapping
"""

from .data import XCode
from . import crosscoder  # Adds training methods to XCode

# Dynamics (legacy compatibility)
from .dynamics import make_dynamics, DynaModel
from .dynamics.hopf import hopf_dfun
from .dynamics.mpr import mpr_dfun

# Dynamics (new interfaces)
from .dynamics import (
    DynamicsModel,
    ModelRegistry,
    ParameterDefinition,
    ParameterSpace,
    ModelMetadata,
    SimulationConfig,
    SimulationResult,
    validate_parameter_space,
    get_model as get_dynamics_model,
    list_models as list_dynamics_models,
    HopfModel,
    make_hopf,
    MPRModel,
    make_mpr,
    WilsonCowanModel,
    make_wilson_cowan,
    WongWangModel,
    make_wong_wang,
    KuramotoModel,
    make_kuramoto,
    FitzHughNagumoModel,
    make_fitzhugh_nagumo,
)

# Simulation and SBI
from .simulation import sample_model, sample_subj_model, bench_model, bench_cohort_model
from .inference import run_sbi, posterior_diags, to_torch, uniform_var

# Utilities
from .utils import triu_to_mat, load_pkl, MvNorm, all_conf_rates, apply, small

# Regime selection
from .regimes import assess_regime, covariance_based_metric, pca_variance_metric

# Datasets (NEW)
from .datasets import (
    DatasetLoader,
    DatasetConfig,
    DatasetMetadata,
    SubjectMetadata,
    ValidationResult,
    DatasetRegistry,
    load_dataset,
    validate_config,
    get_dataset_metadata,
)
from .datasets import local_file
from .datasets import ebrains_kg
from .datasets import openneuro

LocalFileLoader = local_file.LocalFileLoader
EbrainsKGLoader = ebrains_kg.EbrainsKGLoader
OpenNeuroLoader = openneuro.OpenNeuroLoader

# API (NEW)
from .api import create_app
from .api.models import (
    InferenceRequest,
    InferenceResponse,
    ModelInfo,
    ErrorResponse,
)
from .api.errors import (
    ApvbtError,
    ValidationError,
    ModelNotFoundError,
    InferenceError,
)

__version__ = "2.0.0"

__all__ = [
    # Data and training
    "XCode",
    "crosscoder",
    # Dynamics (legacy)
    "make_dynamics",
    "DynaModel",
    "hopf_dfun",
    "mpr_dfun",
    # Dynamics (new interfaces)
    "DynamicsModel",
    "ModelRegistry",
    "ParameterDefinition",
    "ParameterSpace",
    "ModelMetadata",
    "SimulationConfig",
    "SimulationResult",
    "validate_parameter_space",
    "get_dynamics_model",
    "list_dynamics_models",
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
    # Simulation
    "sample_model",
    "sample_subj_model",
    "bench_model",
    "bench_cohort_model",
    # SBI
    "run_sbi",
    "posterior_diags",
    "to_torch",
    "uniform_var",
    # Utils
    "triu_to_mat",
    "load_pkl",
    "MvNorm",
    "all_conf_rates",
    "apply",
    "small",
    # Regime selection
    "assess_regime",
    "covariance_based_metric",
    "pca_variance_metric",
    # Datasets
    "DatasetLoader",
    "DatasetConfig",
    "DatasetMetadata",
    "SubjectMetadata",
    "ValidationResult",
    "DatasetRegistry",
    "load_dataset",
    "validate_config",
    "get_dataset_metadata",
    "LocalFileLoader",
    "EbrainsKGLoader",
    "OpenNeuroLoader",
    # API
    "create_app",
    "InferenceRequest",
    "InferenceResponse",
    "ModelInfo",
    "ErrorResponse",
    "ApvbtError",
    "ValidationError",
    "ModelNotFoundError",
    "InferenceError",
]
