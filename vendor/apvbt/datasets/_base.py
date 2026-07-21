"""Base classes for dataset loaders.

This module contains core interfaces and dataclasses used by
all dataset loaders, avoiding circular import issues.
"""

from typing import (
    Protocol,
    Dict,
    Any,
    Optional,
    List,
    Type,
    runtime_checkable,
    TYPE_CHECKING,
)
from dataclasses import dataclass, field
import warnings

if TYPE_CHECKING:
    from apvbt.data import XCode


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    Attributes:
        source_type: Type of data source (e.g., 'local-file', 'ebrains-kg')
        source_url: URL or path to data source
        format: Format of the data (e.g., 'pkl', 'ebrains-kg')
        parcellations: List of parcellation schemes to load
        train_test_split: Fraction of data to use for training
        normalization: Normalization method ('sqrt', 'zscore', 'none')
        metadata_fields: List of metadata fields to extract
        cache_path: Path to cache directory
        download_params: Additional parameters for download
    """

    source_type: str
    source_url: Optional[str] = None
    format: str = "pkl"
    parcellations: List[str] = field(default_factory=list)
    train_test_split: float = 0.8
    normalization: str = "sqrt"
    metadata_fields: List[str] = field(default_factory=list)
    cache_path: Optional[str] = None
    download_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    """Metadata about a dataset.

    Attributes:
        name: Dataset name
        description: Human-readable description
        n_subjects: Number of subjects in dataset
        n_parcellations: Number of parcellation schemes
        parcellation_names: List of parcellation names
        subjects: List of subject IDs
        metadata_dict: Dictionary of metadata fields (age, sex, diagnosis, etc.)
        source_info: Information about data source
    """

    name: str
    description: str
    n_subjects: int
    n_parcellations: int
    parcellation_names: List[str]
    subjects: List[str]
    metadata_dict: Dict[str, Any] = field(default_factory=dict)
    source_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubjectMetadata:
    """Metadata for a single subject.

    Stores subject-level information including demographics, clinical data,
    acquisition parameters, and quality metrics.

    Attributes:
        subject_id: Unique identifier for the subject
        dataset: Name of the dataset this subject belongs to
        parcellation: Parcellation scheme used for this subject
        demographics: Demographic information (age, sex, handedness, etc.)
        clinical: Clinical information (diagnosis, medications, etc.)
        acquisition: Acquisition parameters (scanner, sequence, etc.)
        quality_metrics: Quality metrics (SNR, motion, etc.)
    """

    subject_id: str
    dataset: str
    parcellation: str
    demographics: Dict[str, Any] = field(default_factory=dict)
    clinical: Dict[str, Any] = field(default_factory=dict)
    acquisition: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
        metrics: Dictionary of validation metrics
        details: Additional validation details
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetLoader(Protocol):
    """Interface for dataset loaders.

    Dataset loaders implement this protocol to provide a standardized
    way to load brain connectome datasets from various sources.
    """

    def load(self, config: DatasetConfig) -> "XCode":
        """Load dataset and return XCode object.

        Args:
            config: Dataset configuration

        Returns:
            XCode object containing loaded data

        Raises:
            FileNotFoundError: If source not found
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        ...

    def validate(self, config: DatasetConfig) -> ValidationResult:
        """Validate configuration before loading.

        Args:
            config: Dataset configuration

        Returns:
            ValidationResult indicating success or failure
        """
        ...

    def get_metadata(self, config: DatasetConfig) -> DatasetMetadata:
        """Get dataset metadata without loading.

        Args:
            config: Dataset configuration

        Returns:
            DatasetMetadata object
        """
        ...


class DatasetRegistry:
    """Registry for dataset loaders using decorator-based registration.

    Example:
        @DatasetRegistry.register('local-file')
        class LocalFileLoader:
            def load(self, config):
                ...

        loader_class = DatasetRegistry.get('local-file')
        loader = loader_class()
        data = loader.load(config)
    """

    _loaders: Dict[str, Type[DatasetLoader]] = {}

    @classmethod
    def register(cls, source_type: str):
        """Decorator for registering dataset loaders.

        Args:
            source_type: Unique identifier for this loader type

        Returns:
            Decorator function

        Raises:
            ValueError: If source_type already registered
        """

        def decorator(loader_class: Type[DatasetLoader]):
            if source_type in cls._loaders:
                warnings.warn(
                    f"Loader for '{source_type}' already registered. "
                    f"Overwriting with {loader_class.__name__}"
                )
            cls._loaders[source_type] = loader_class
            return loader_class

        return decorator

    @classmethod
    def get(cls, source_type: str) -> Type[DatasetLoader]:
        """Get loader class by source type.

        Args:
            source_type: Identifier for loader type

        Returns:
            Loader class

        Raises:
            KeyError: If source_type not found
        """
        if source_type not in cls._loaders:
            available = ", ".join(sorted(cls._loaders.keys()))
            raise KeyError(
                f"Unknown dataset loader type: '{source_type}'. Available: {available}"
            )
        return cls._loaders[source_type]

    @classmethod
    def list_loaders(cls) -> List[str]:
        """List all registered loader types.

        Returns:
            List of loader type identifiers
        """
        return sorted(cls._loaders.keys())

    @classmethod
    def clear(cls):
        """Clear all registered loaders (mainly for testing)."""
        cls._loaders.clear()


def load_dataset(config: DatasetConfig) -> "XCode":
    """Load dataset using registered loader.

    Convenience function that looks up appropriate loader
    based on config.source_type and loads data.

    Args:
        config: Dataset configuration

    Returns:
        XCode object containing loaded data

    Raises:
        KeyError: If source_type not registered
        ValueError: If configuration is invalid
        RuntimeError: If loading fails
    """
    loader_class = DatasetRegistry.get(config.source_type)
    loader = loader_class()
    return loader.load(config)


def validate_config(config: DatasetConfig) -> ValidationResult:
    """Validate dataset configuration using registered loader.

    Args:
        config: Dataset configuration

    Returns:
        ValidationResult indicating success or failure
    """
    loader_class = DatasetRegistry.get(config.source_type)
    loader = loader_class()
    return loader.validate(config)


def get_dataset_metadata(config: DatasetConfig) -> DatasetMetadata:
    """Get dataset metadata using registered loader.

    Args:
        config: Dataset configuration

    Returns:
        DatasetMetadata object
    """
    loader_class = DatasetRegistry.get(config.source_type)
    loader = loader_class()
    return loader.get_metadata(config)


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
]
