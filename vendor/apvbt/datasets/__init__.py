"""Dataset loader framework for APVBT.

This module provides a pluggable architecture for loading brain connectome
datasets from various sources (EBRAINS KG, local files, OpenNeuro, etc.).
"""

from apvbt.datasets._base import (
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

# Import loaders to register them in the registry
from apvbt.datasets import local_file
from apvbt.datasets import ebrains_kg
from apvbt.datasets import openneuro

__all__ = [
    "DatasetLoader",
    "DatasetConfig",
    "DatasetMetadata",
    "SubjectMetadata",
    "ValidationResult",
    "DatasetRegistry",
    "load_dataset",
    "validate_config",
    "get_dataset_metadata",
    "local_file",
    "ebrains_kg",
    "openneuro",
]
