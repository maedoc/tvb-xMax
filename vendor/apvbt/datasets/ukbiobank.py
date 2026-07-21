"""UK Biobank dataset loader.

This module provides a loader for brain connectome datasets from the UK Biobank,
a large-scale biomedical database containing in-depth genetic and health information
from half a million UK participants.

Note: This loader requires pandas for reading participant metadata.
Install with: pip install pandas
"""

import os
import json
import numpy as np
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from apvbt.datasets._base import (
    DatasetLoader,
    DatasetConfig,
    DatasetMetadata,
    ValidationResult,
    DatasetRegistry,
    SubjectMetadata,
)

if TYPE_CHECKING:
    from apvbt.data import XCode


@DatasetRegistry.register("ukbiobank")
class UKBiobankLoader:
    """Loader for UK Biobank brain imaging datasets.

    Supports loading structural connectomes from UK Biobank imaging data
    along with rich demographic and health metadata.

    Expected directory structure:
        ukbiobank_dataset/
        ├── participants.csv          # Participant metadata (required)
        ├── connectomes/              # Connectome files (optional subdirectory)
        │   ├── sub_1/                # Subject subdirectory (subject ID)
        │   │   ├── parc_schaefer2018_connectome.npy
        │   │   └── parc_aal_connectome.npy
        │   └── sub_2/
        └── participants.json         # Optional additional metadata

    Connectome files can be in .npy, .mat, or .csv format.
    """

    def load(self, config: DatasetConfig) -> "XCode":
        """Load dataset from UK Biobank directory.

        Args:
            config: Dataset configuration with source_type='ukbiobank'
                - source_url: Local path to UK Biobank dataset directory
                - download_params: Dict with:
                    - include_subjects: List of subject IDs to include (optional)
                    - exclude_subjects: List of subject IDs to exclude (optional)

        Returns:
            XCode instance with loaded data

        Raises:
            FileNotFoundError: If UK Biobank directory not found
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        from apvbt.data import XCode
        import jax.numpy as jp

        validation = self.validate(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")

        source_path = Path(config.source_url) if config.source_url else None
        if not source_path or not source_path.exists():
            raise FileNotFoundError(f"UK Biobank directory not found: {source_path}")

        download_params = config.download_params
        include_subjects = download_params.get("include_subjects", [])
        exclude_subjects = download_params.get("exclude_subjects", [])

        conns = []
        means = []
        parcs = []
        subjects_metadata = {}

        # Parse participants.csv for demographic metadata
        participants_file = source_path / "participants.csv"
        demographics = self._load_participants_csv(participants_file) if participants_file.exists() else {}

        # Find all connectome files
        connectomes_by_parc = self._find_connectomes(
            source_path, include_subjects, exclude_subjects
        )

        for parc_name, connectomes in connectomes_by_parc.items():
            parcs.append(parc_name)
            conn_array = np.array(list(connectomes.values()), dtype="f")
            jax_conn = jp.array(conn_array)
            conns.append(jax_conn)
            means.append(jp.mean(jax_conn, axis=0))

            for subject_id in connectomes.keys():
                if subject_id not in subjects_metadata:
                    subjects_metadata[subject_id] = self._extract_subject_metadata(
                        source_path, subject_id, demographics
                    )

        xc = XCode()
        if conns:
            xc.conns = conns
            xc.means = means
            xc.parcs = parcs
        else:
            xc.conns = []
            xc.means = []
            xc.parcs = []

        n_subjects = len(subjects_metadata)
        xc.tts = (
            config.train_test_split
            if isinstance(config.train_test_split, int)
            else int(n_subjects * config.train_test_split)
            if n_subjects > 0
            else 0
        )
        xc.wbs = []

        xc.metadata = {}
        for subject_id, metadata in subjects_metadata.items():
            xc.metadata[subject_id] = SubjectMetadata(
                subject_id=subject_id,
                dataset="ukbiobank",
                parcellation=parcs[0] if parcs else "unknown",
                demographics=metadata.get("demographics", {}),
                clinical=metadata.get("clinical", {}),
                acquisition=metadata.get("acquisition", {}),
                quality_metrics=metadata.get("quality_metrics", {}),
            )

        return xc

    def validate(self, config: DatasetConfig) -> ValidationResult:
        """Validate configuration before loading.

        Args:
            config: Dataset configuration

        Returns:
            ValidationResult indicating success or failure
        """
        errors = []
        warnings = []

        if config.source_type != "ukbiobank":
            errors.append(
                f"Expected source_type 'ukbiobank', got '{config.source_type}'"
            )

        if not config.source_url:
            errors.append("source_url must be provided for UK Biobank loader")

        if config.source_url:
            source_path = Path(config.source_url)
            if not source_path.exists():
                errors.append(f"Source path not found: {config.source_url}")
            elif not source_path.is_dir():
                errors.append(f"Source path must be a directory: {config.source_url}")
            else:
                participants_file = source_path / "participants.csv"
                if not participants_file.exists():
                    warnings.append(
                        "No participants.csv found - demographic metadata unavailable"
                    )
                
                # Check for connectome files
                parcellations = self._detect_parcellations(source_path)
                if not parcellations:
                    warnings.append(
                        "No connectome files found - dataset will be empty"
                    )

        download_params = config.download_params
        include_subjects = download_params.get("include_subjects", [])
        exclude_subjects = download_params.get("exclude_subjects", [])

        if include_subjects and not isinstance(include_subjects, list):
            errors.append("download_params['include_subjects'] must be a list")

        if exclude_subjects and not isinstance(exclude_subjects, list):
            errors.append("download_params['exclude_subjects'] must be a list")

        if config.train_test_split is not None and config.train_test_split < 0:
            errors.append("train_test_split must be non-negative")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={"config_valid": len(errors) == 0},
        )

    def get_metadata(self, config: DatasetConfig) -> DatasetMetadata:
        """Get dataset metadata without loading.

        Args:
            config: Dataset configuration

        Returns:
            DatasetMetadata object with information about the dataset
        """
        if not config.source_url:
            return DatasetMetadata(
                name="ukbiobank",
                description="UK Biobank dataset (source not specified)",
                n_subjects=0,
                n_parcellations=0,
                parcellation_names=[],
                subjects=[],
                metadata_dict={"source_type": "ukbiobank"},
                source_info={"source_type": "ukbiobank", "source_url": None},
            )

        source_path = Path(config.source_url)
        if not source_path.exists():
            return DatasetMetadata(
                name="ukbiobank",
                description=f"UK Biobank dataset (path not found: {source_path})",
                n_subjects=0,
                n_parcellations=0,
                parcellation_names=[],
                subjects=[],
                metadata_dict={"source_type": "ukbiobank"},
                source_info={"source_type": "ukbiobank", "source_url": str(source_path)},
            )

        # List subjects from connectome directories
        subjects = self._list_subjects(source_path)
        parcellations = self._detect_parcellations(source_path)

        # Try to load participants.csv for demographic summary
        demographics_summary = {}
        participants_file = source_path / "participants.csv"
        if participants_file.exists():
            try:
                demographics_summary = self._summarize_demographics(participants_file)
            except Exception:
                pass

        metadata = DatasetMetadata(
            name="ukbiobank",
            description=f"UK Biobank dataset ({len(subjects)} subjects)",
            n_subjects=len(subjects),
            n_parcellations=len(parcellations),
            parcellation_names=parcellations,
            subjects=subjects,
            metadata_dict={
                "source_type": "ukbiobank",
                "has_demographic_metadata": participants_file.exists(),
                **demographics_summary,
            },
            source_info={
                "source_type": "ukbiobank",
                "source_url": str(source_path),
                "format": "directory",
            },
        )

        return metadata

    def _load_participants_csv(self, csv_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load participant demographics from CSV file.

        Args:
            csv_path: Path to participants.csv

        Returns:
            Dictionary mapping subject ID to demographic data
        """
        try:
            import pandas as pd
        except ImportError:
            return {}

        try:
            df = pd.read_csv(csv_path)
            # Expect columns: subject_id, age, sex, ...
            if "subject_id" not in df.columns:
                return {}

            demographics = {}
            for _, row in df.iterrows():
                subject_id = str(row["subject_id"])
                demographics[subject_id] = {
                    col: row[col] for col in df.columns if col != "subject_id"
                }
            return demographics
        except Exception:
            return {}

    def _find_connectomes(
        self,
        source_path: Path,
        include_subjects: List[str] = [],
        exclude_subjects: List[str] = [],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Find all connectome files in directory structure.

        Args:
            source_path: Root directory
            include_subjects: List of subject IDs to include
            exclude_subjects: List of subject IDs to exclude

        Returns:
            Dictionary mapping parcellation names to subject connectomes
        """
        connectomes_by_parc: Dict[str, Dict[str, np.ndarray]] = {}

        # Look for connectomes/ subdirectory
        connectomes_dir = source_path / "connectomes"
        if not connectomes_dir.exists():
            # Alternative: look for subject directories directly under source_path
            connectomes_dir = source_path

        for subject_dir in connectomes_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name
            if include_subjects and subject_id not in include_subjects:
                continue
            if subject_id in exclude_subjects:
                continue

            for conn_file in subject_dir.glob("*connectome*.npy"):
                parc_name = self._extract_parcellation_name(conn_file.name)
                if parc_name not in connectomes_by_parc:
                    connectomes_by_parc[parc_name] = {}
                try:
                    conn = np.load(str(conn_file))
                    connectomes_by_parc[parc_name][subject_id] = conn
                except Exception:
                    pass

            for conn_file in subject_dir.glob("*connectome*.mat"):
                parc_name = self._extract_parcellation_name(conn_file.name)
                if parc_name not in connectomes_by_parc:
                    connectomes_by_parc[parc_name] = {}
                try:
                    conn = self._load_mat_connectome(conn_file)
                    connectomes_by_parc[parc_name][subject_id] = conn
                except Exception:
                    pass

            for conn_file in subject_dir.glob("*connectome*.csv"):
                parc_name = self._extract_parcellation_name(conn_file.name)
                if parc_name not in connectomes_by_parc:
                    connectomes_by_parc[parc_name] = {}
                try:
                    conn = np.loadtxt(str(conn_file), delimiter=",")
                    connectomes_by_parc[parc_name][subject_id] = conn
                except Exception:
                    pass

        return connectomes_by_parc

    def _load_mat_connectome(self, file_path: Path) -> np.ndarray:
        """Load connectome from MATLAB .mat file.

        Args:
            file_path: Path to .mat file

        Returns:
            Connectome matrix

        Raises:
            RuntimeError: If file cannot be loaded
        """
        try:
            import scipy.io

            mat_data = scipy.io.loadmat(str(file_path))

            for key in mat_data.keys():
                if not key.startswith("__"):
                    data = mat_data[key]
                    if isinstance(data, np.ndarray) and data.ndim == 2:
                        return data

            raise RuntimeError("No valid connectome matrix found in .mat file")
        except ImportError:
            raise RuntimeError(
                "scipy package is required for loading .mat files. "
                "Install with: pip install scipy"
            )

    def _extract_parcellation_name(self, filename: str) -> str:
        """Extract parcellation name from connectome filename.

        Args:
            filename: Connectome filename

        Returns:
            Parcellation name
        """
        # Remove connectome suffix and extensions
        parts = (
            filename.replace("_connectome", "")
            .replace(".npy", "")
            .replace(".mat", "")
            .replace(".csv", "")
        )

        # Find the last segment (parcellation name)
        # Format examples: parc_schaefer2018_connectome.npy -> schaefer2018
        #                 aal_connectome.mat -> aal
        #                 sub_1_schaefer2018_connectome.csv -> schaefer2018

        segments = [s for s in parts.replace("_", " ").replace("-", " ").split() if s]
        if not segments:
            return parts

        # If it starts with "parc_", extract after that
        if segments[0].lower().startswith("parc_"):
            return segments[0][5:]

        # Return the last segment (parcellation name)
        return segments[-1]

    def _list_subjects(self, source_path: Path) -> List[str]:
        """List all subjects in dataset.

        Args:
            source_path: Root directory

        Returns:
            List of subject IDs
        """
        subjects = []

        # Check connectomes/ subdirectory
        connectomes_dir = source_path / "connectomes"
        if connectomes_dir.exists():
            for item in connectomes_dir.iterdir():
                if item.is_dir():
                    subjects.append(item.name)
        else:
            # Look for subject directories directly under source_path
            for item in source_path.iterdir():
                if item.is_dir():
                    subjects.append(item.name)

        return sorted(subjects)

    def _detect_parcellations(self, source_path: Path) -> List[str]:
        """Detect parcellations used in dataset.

        Args:
            source_path: Root directory

        Returns:
            List of parcellation names
        """
        parcellations = set()

        connectomes_dir = source_path / "connectomes"
        if not connectomes_dir.exists():
            connectomes_dir = source_path

        for subject_dir in connectomes_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            for conn_file in subject_dir.glob("*connectome*"):
                parc_name = self._extract_parcellation_name(conn_file.name)
                parcellations.add(parc_name)

        return sorted(list(parcellations))

    def _extract_subject_metadata(
        self,
        source_path: Path,
        subject_id: str,
        demographics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract metadata for a single subject.

        Args:
            source_path: Root directory
            subject_id: Subject ID
            demographics: Pre-loaded demographic data

        Returns:
            Dictionary with subject metadata
        """
        metadata = {
            "demographics": {},
            "clinical": {},
            "acquisition": {},
            "quality_metrics": {},
        }

        # Add demographic data if available
        if subject_id in demographics:
            metadata["demographics"] = demographics[subject_id].copy()

        # Look for subject-specific JSON metadata files
        subject_dir = source_path / "connectomes" / subject_id
        if not subject_dir.exists():
            subject_dir = source_path / subject_id

        if subject_dir.exists():
            for json_file in subject_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        json_data = json.load(f)
                        metadata["acquisition"].update(json_data)
                except Exception:
                    pass

        return metadata

    def _summarize_demographics(self, csv_path: Path) -> Dict[str, Any]:
        """Create summary statistics for demographic data.

        Args:
            csv_path: Path to participants.csv

        Returns:
            Dictionary with demographic summary
        """
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            summary = {}

            for col in df.columns:
                if col == "subject_id":
                    continue
                if df[col].dtype in [np.int64, np.float64]:
                    summary[f"{col}_mean"] = float(df[col].mean())
                    summary[f"{col}_std"] = float(df[col].std())
                    summary[f"{col}_min"] = float(df[col].min())
                    summary[f"{col}_max"] = float(df[col].max())
                else:
                    summary[f"{col}_counts"] = df[col].value_counts().to_dict()

            return summary
        except Exception:
            return {}


__all__ = ["UKBiobankLoader"]