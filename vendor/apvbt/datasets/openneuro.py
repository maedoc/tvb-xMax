"""OpenNeuro dataset loader.

This module provides a loader for brain connectome datasets from
OpenNeuro, a platform hosting BIDS-compliant neuroimaging data.
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


@DatasetRegistry.register("openneuro")
class OpenNeuroLoader:
    """Loader for OpenNeuro BIDS datasets.

    Supports downloading and parsing BIDS-compliant datasets from OpenNeuro,
    extracting structural connectomes from DWI data and subject metadata.

    Note: This loader requires the openneuro-py package for downloading datasets.
    Install with: pip install openneuro-py
    """

    def load(self, config: DatasetConfig) -> "XCode":
        """Load dataset from OpenNeuro.

        Args:
            config: Dataset configuration with source_type='openneuro'
                - source_url: Local path to existing BIDS dataset (optional)
                - download_params: Dict with:
                    - dataset_id: OpenNeuro dataset ID (e.g., 'ds000001')
                    - version: Dataset version tag (optional, defaults to latest)
                    - include_subjects: List of subject IDs to include (optional)
                    - exclude_subjects: List of subject IDs to exclude (optional)

        Returns:
            XCode instance with loaded data

        Raises:
            FileNotFoundError: If BIDS dataset not found
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        from apvbt.data import XCode
        import jax.numpy as jp

        validation = self.validate(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")

        download_params = config.download_params
        dataset_id = download_params.get("dataset_id")
        version = download_params.get("version")
        include_subjects = download_params.get("include_subjects", [])
        exclude_subjects = download_params.get("exclude_subjects", [])

        if config.source_url:
            bids_root = Path(config.source_url)
        else:
            bids_root = self._download_dataset(dataset_id, version)

        conns = []
        means = []
        parcs = []
        subjects_metadata = {}

        for parc_name, connectomes in self._parse_bids_connectomes(
            bids_root, include_subjects, exclude_subjects
        ).items():
            parcs.append(parc_name)
            conn_array = np.array(list(connectomes.values()), dtype="f")
            conns.append(jp.array(conn_array))
            means.append(conn_array.mean(axis=0))

            for subject_id in connectomes.keys():
                if subject_id not in subjects_metadata:
                    subjects_metadata[subject_id] = self._extract_subject_metadata(
                        bids_root, subject_id
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
                dataset=dataset_id,
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

        if config.source_type != "openneuro":
            errors.append(
                f"Expected source_type 'openneuro', got '{config.source_type}'"
            )

        download_params = config.download_params
        dataset_id = download_params.get("dataset_id")

        if not dataset_id and not config.source_url:
            errors.append(
                "Either dataset_id in download_params or source_url must be provided"
            )

        if config.source_url:
            source_path = Path(config.source_url)
            if not source_path.exists():
                errors.append(f"Source path not found: {config.source_url}")
            elif not source_path.is_dir():
                errors.append(f"Source path must be a directory: {config.source_url}")
            else:
                dataset_description = source_path / "dataset_description.json"
                if not dataset_description.exists():
                    warnings.append(
                        "No dataset_description.json found - may not be a BIDS dataset"
                    )

        if dataset_id and not isinstance(dataset_id, str):
            errors.append("download_params['dataset_id'] must be a string")

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
        download_params = config.download_params
        dataset_id = download_params.get("dataset_id", "unknown")

        if config.source_url:
            bids_root = Path(config.source_url)
        else:
            bids_root = Path(tempfile.gettempdir()) / dataset_id
            if not bids_root.exists():
                bids_root = Path(tempfile.gettempdir()) / "openneuro_temp"

        dataset_description = {}
        if bids_root.exists():
            desc_file = bids_root / "dataset_description.json"
            if desc_file.exists():
                with open(desc_file, "r") as f:
                    dataset_description = json.load(f)

        subjects = []
        if bids_root.exists():
            subjects = self._list_subjects(bids_root)

        n_subjects = len(subjects)
        parcellations = (
            self._detect_parcellations(bids_root) if bids_root.exists() else []
        )

        metadata = DatasetMetadata(
            name=dataset_description.get("Name", dataset_id),
            description=dataset_description.get(
                "Description",
                f"OpenNeuro BIDS dataset ({n_subjects} subjects)",
            ),
            n_subjects=n_subjects,
            n_parcellations=len(parcellations),
            parcellation_names=parcellations,
            subjects=subjects,
            metadata_dict={
                "dataset_id": dataset_id,
                "dataset_version": download_params.get("version", "latest"),
                "bids_version": dataset_description.get("BIDSVersion", "unknown"),
                "license": dataset_description.get("License", "unknown"),
                "has_demographic_metadata": any(
                    self._subject_has_demographics(bids_root, sub_id)
                    for sub_id in subjects
                )
                if subjects
                else False,
            },
            source_info={
                "source_type": "openneuro",
                "source_url": config.source_url or str(bids_root),
                "format": "bids",
                "download_params": download_params,
            },
        )

        return metadata

    def _download_dataset(self, dataset_id: str, version: Optional[str] = None) -> Path:
        """Download dataset from OpenNeuro.

        Args:
            dataset_id: OpenNeuro dataset ID (e.g., 'ds000001')
            version: Dataset version tag (optional, defaults to latest)

        Returns:
            Path to downloaded BIDS dataset

        Raises:
            ImportError: If openneuro-py is not installed
            RuntimeError: If download fails
        """
        try:
            from openneuro import download
        except ImportError:
            raise ImportError(
                "openneuro-py package is required. Install with: pip install openneuro-py"
            )

        temp_dir = Path(tempfile.gettempdir())
        download_dir = temp_dir / dataset_id

        try:
            if version:
                download(dataset=dataset_id, tag=version, target_dir=str(download_dir))
            else:
                download(dataset=dataset_id, target_dir=str(download_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {dataset_id}: {e}")

        return download_dir

    def _parse_bids_connectomes(
        self,
        bids_root: Path,
        include_subjects: List[str] = [],
        exclude_subjects: List[str] = [],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Parse structural connectomes from BIDS DWI data.

        Args:
            bids_root: Path to BIDS dataset root
            include_subjects: List of subject IDs to include (optional)
            exclude_subjects: List of subject IDs to exclude (optional)

        Returns:
            Dictionary mapping parcellation names to subject connectomes
        """
        connectomes_by_parc: Dict[str, Dict[str, np.ndarray]] = {}

        if not bids_root.exists():
            return connectomes_by_parc

        subjects = self._list_subjects(bids_root)

        for subject_id in subjects:
            if include_subjects and subject_id not in include_subjects:
                continue
            if subject_id in exclude_subjects:
                continue

            subject_path = bids_root / subject_id
            if not subject_path.exists():
                continue

            for dwi_file in subject_path.glob("dwi/*_connectome*.nii*"):
                parc_name = self._extract_parcellation_name(dwi_file.name)

                if parc_name not in connectomes_by_parc:
                    connectomes_by_parc[parc_name] = {}

                try:
                    conn = self._load_connectome_file(dwi_file)
                    connectomes_by_parc[parc_name][subject_id] = conn
                except Exception as e:
                    pass

            for dwi_file in subject_path.glob("dwi/*connectome*.mat"):
                parc_name = self._extract_parcellation_name(dwi_file.name)

                if parc_name not in connectomes_by_parc:
                    connectomes_by_parc[parc_name] = {}

                try:
                    conn = self._load_mat_connectome(dwi_file)
                    connectomes_by_parc[parc_name][subject_id] = conn
                except Exception as e:
                    pass

        return connectomes_by_parc

    def _load_connectome_file(self, file_path: Path) -> np.ndarray:
        """Load connectome from NIfTI or CIFTI file.

        Args:
            file_path: Path to connectome file

        Returns:
            Connectome matrix

        Raises:
            RuntimeError: If file cannot be loaded
        """
        try:
            import nibabel as nib

            img = nib.load(str(file_path))
            data = img.get_fdata()

            if data.ndim == 3:
                return data
            elif data.ndim == 2:
                return data
            else:
                raise RuntimeError(f"Unexpected connectome dimensions: {data.ndim}")
        except ImportError:
            raise RuntimeError(
                "nibabel package is required for loading connectome files. "
                "Install with: pip install nibabel"
            )

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
            filename.replace("_connectome", "").replace(".nii", "").replace(".gz", "")
        )
        parts = parts.replace(".mat", "")

        # Find the last segment (parcellation name)
        # Format examples: sub-01_dwi_schaefer2018_connectome.nii.gz -> schaefer2018
        #                 aal_connectome.nii -> aal
        #                 sub-01_dwi_100parcs_connectome.mat -> 100parcs
        #                 atlas-DesikanKilliany_connectome.nii -> DesikanKilliany

        # Split on common delimiters and get the last meaningful part
        segments = [s for s in parts.replace("_", " ").replace("-", " ").split() if s]
        if not segments:
            return parts

        # If it starts with "atlas-", extract the atlas name (everything after the dash)
        if segments[0].lower().startswith("atlas-"):
            # Return everything after "atlas-"
            return segments[0][6:]

        # Return the last segment (parcellation name)
        return segments[-1]

    def _list_subjects(self, bids_root: Path) -> List[str]:
        """List all subjects in BIDS dataset.

        Args:
            bids_root: Path to BIDS dataset root

        Returns:
            List of subject IDs (e.g., ['sub-01', 'sub-02'])
        """
        subjects = []

        if not bids_root.exists():
            return subjects

        for item in bids_root.iterdir():
            if item.is_dir() and item.name.startswith("sub-"):
                subjects.append(item.name)

        return sorted(subjects)

    def _detect_parcellations(self, bids_root: Path) -> List[str]:
        """Detect parcellations used in dataset.

        Args:
            bids_root: Path to BIDS dataset root

        Returns:
            List of parcellation names
        """
        parcellations = set()

        for subject_path in bids_root.glob("sub-*/"):
            for dwi_file in subject_path.glob("dwi/*connectome*"):
                parc_name = self._extract_parcellation_name(dwi_file.name)
                parcellations.add(parc_name)

        return sorted(list(parcellations))

    def _extract_subject_metadata(
        self, bids_root: Path, subject_id: str
    ) -> Dict[str, Any]:
        """Extract metadata for a single subject.

        Args:
            bids_root: Path to BIDS dataset root
            subject_id: Subject ID (e.g., 'sub-01')

        Returns:
            Dictionary with subject metadata
        """
        metadata = {
            "demographics": {},
            "clinical": {},
            "acquisition": {},
            "quality_metrics": {},
        }

        subject_path = bids_root / subject_id
        if not subject_path.exists():
            return metadata

        participants_file = bids_root / "participants.tsv"
        if participants_file.exists():
            try:
                import pandas as pd

                df = pd.read_csv(participants_file, sep="\t")
                participant_row = df[df["participant_id"] == subject_id]

                if not participant_row.empty:
                    row = participant_row.iloc[0]
                    for col in df.columns:
                        if col != "participant_id":
                            value = row[col]
                            if pd.notna(value):
                                metadata["demographics"][col] = value
            except ImportError:
                pass
            except Exception:
                pass

        for json_file in subject_path.glob("*_sessions.json"):
            try:
                with open(json_file, "r") as f:
                    session_data = json.load(f)
                    metadata["acquisition"].update(session_data)
            except Exception:
                pass

        for json_file in subject_path.glob("dwi/*_dwi.json"):
            try:
                with open(json_file, "r") as f:
                    dwi_data = json.load(f)
                    metadata["acquisition"].update(dwi_data)
            except Exception:
                pass

        return metadata

    def _subject_has_demographics(self, bids_root: Path, subject_id: str) -> bool:
        """Check if subject has demographic metadata.

        Args:
            bids_root: Path to BIDS dataset root
            subject_id: Subject ID (e.g., 'sub-01')

        Returns:
            True if subject has demographics
        """
        metadata = self._extract_subject_metadata(bids_root, subject_id)
        return len(metadata["demographics"]) > 0


__all__ = ["OpenNeuroLoader"]
