"""Local file loader for APVBT datasets.

This loader handles loading XCode datasets from local pickle files.
"""

from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from apvbt.datasets._base import (
    DatasetLoader,
    DatasetConfig,
    DatasetMetadata,
    ValidationResult,
    DatasetRegistry,
)

if TYPE_CHECKING:
    from apvbt.data import XCode


@DatasetRegistry.register("local-file")
class LocalFileLoader(DatasetLoader):
    """Loader for local XCode pickle files.

    This is simplest loader that reads pre-processed XCode
    datasets from local pickle files.
    """

    def load(self, config: DatasetConfig) -> "XCode":
        """Load dataset from local pickle file.

        Args:
            config: Dataset configuration with source_url pointing to .pkl file

        Returns:
            XCode object containing loaded data

        Raises:
            FileNotFoundError: If file not found
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        if not config.source_url:
            raise ValueError("source_url must be specified for local-file loader")

        path = Path(config.source_url)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {config.source_url}")

        if not path.is_file():
            raise ValueError(f"source_url must be a file: {config.source_url}")

        if not path.suffix == ".pkl":
            raise ValueError(
                f"source_url must point to a .pkl file: {config.source_url}"
            )

        try:
            with open(path, "rb") as f:
                xcode = pickle.load(f)

            if not hasattr(xcode, "conns") or not hasattr(xcode, "parcs"):
                raise ValueError(
                    f"Loaded object is not an XCode instance: {type(xcode)}"
                )

            return xcode

        except (pickle.PickleError, EOFError) as e:
            raise RuntimeError(f"Failed to load pickle file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading dataset: {e}") from e

    def validate(self, config: DatasetConfig) -> ValidationResult:
        """Validate configuration before loading.

        Checks that:
        - source_url is provided
        - source_url points to an existing .pkl file
        - File is readable

        Args:
            config: Dataset configuration

        Returns:
            ValidationResult indicating success or failure
        """
        errors = []
        warnings = []
        metrics = {}
        details = {}

        if not config.source_url:
            errors.append("source_url must be specified")
            return ValidationResult(is_valid=False, errors=errors)

        path = Path(config.source_url)

        if not path.exists():
            errors.append(f"Dataset file not found: {config.source_url}")
            return ValidationResult(is_valid=False, errors=errors)

        if not path.is_file():
            errors.append(f"source_url must be a file: {config.source_url}")
            return ValidationResult(is_valid=False, errors=errors)

        if path.suffix != ".pkl":
            warnings.append(
                f"source_url does not have .pkl extension: {config.source_url}"
            )

        if config.train_test_split <= 0 or config.train_test_split >= 1:
            errors.append(
                f"train_test_split must be between 0 and 1: {config.train_test_split}"
            )

        if config.normalization not in ["sqrt", "zscore", "none"]:
            errors.append(
                f"normalization must be 'sqrt', 'zscore', or 'none': {config.normalization}"
            )

        try:
            file_size = path.stat().st_size
            metrics["file_size_bytes"] = file_size
            metrics["file_size_mb"] = file_size / (1024 * 1024)
            details["path"] = str(path.resolve())

            if file_size < 1024:
                warnings.append("File size is very small, may not contain valid data")

        except OSError as e:
            errors.append(f"Cannot access file: {e}")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            details=details,
        )

    def get_metadata(self, config: DatasetConfig) -> DatasetMetadata:
        """Get dataset metadata without loading full data.

        For local files, this reads the pickle file and extracts
        metadata from the XCode object without loading large arrays.

        Args:
            config: Dataset configuration

        Returns:
            DatasetMetadata object
        """
        if not config.source_url:
            raise ValueError("source_url must be specified")

        path = Path(config.source_url)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {config.source_url}")

        try:
            with open(path, "rb") as f:
                xcode = pickle.load(f)

            if not hasattr(xcode, "conns") or not hasattr(xcode, "parcs"):
                raise ValueError(f"Loaded object is not an XCode instance")

            n_subjects = xcode.conns[0].shape[0] if xcode.conns else 0
            n_parcellations = len(xcode.parcs) if xcode.parcs else 0

            metadata = DatasetMetadata(
                name=path.stem,
                description=f"XCode dataset from {config.source_url}",
                n_subjects=n_subjects,
                n_parcellations=n_parcellations,
                parcellation_names=xcode.parcs or [],
                subjects=[f"sub{i:04d}" for i in range(n_subjects)],
                metadata_dict={},
                source_info={
                    "source_type": "local-file",
                    "path": str(path.resolve()),
                    "file_size": path.stat().st_size,
                },
            )

            return metadata

        except (pickle.PickleError, EOFError) as e:
            raise RuntimeError(f"Failed to load pickle file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error loading metadata: {e}") from e
