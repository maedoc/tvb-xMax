"""EBRAINS Knowledge Graph dataset loader.

This module provides a loader for brain connectome datasets from the
EBRAINS Knowledge Graph (HCP and 1000 Brains datasets).
"""

import numpy as np
import tempfile
import tqdm
import urllib.request
import zipfile
from typing import TYPE_CHECKING

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


@DatasetRegistry.register("ebrains-kg")
class EbrainsKGLoader:
    """Loader for EBRAINS Knowledge Graph datasets.

    Supports HCP and 1000 Brains datasets with configurable download
    and parsing options.
    """

    def load(self, config: DatasetConfig) -> "XCode":
        """Load dataset from EBRAINS Knowledge Graph.

        Args:
            config: Dataset configuration with source_type='ebrains-kg'
                - source_url: Path to local zip file (None to download)
                - download_params: Dict with 'hcp' (bool) and 'skip_parcs' (str)

        Returns:
            XCode instance with loaded data

        Raises:
            FileNotFoundError: If source_url file not found
            ValueError: If configuration is invalid
            RuntimeError: If loading fails
        """
        from apvbt.data import XCode
        import jax.numpy as jp

        validation = self.validate(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.errors}")

        hcp = config.download_params.get("hcp", False)
        skip_parcs = config.download_params.get("skip_parcs", "")

        if config.source_url:
            parsed = self._parse(config.source_url, hcp=hcp, skip_parcs=skip_parcs)
        else:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                self._download(temp_file.name, hcp=hcp)
                parsed = self._parse(temp_file.name, hcp=hcp)

        parcs, means, conns = parsed
        xc = XCode()
        xc.conns = [jp.array(_.astype("f")) for _ in conns]
        xc.means = [jp.array(_.astype("f")) for _ in means]
        xc.parcs = parcs
        xc.tts = config.train_test_split or (conns[0].shape[0] // 2)
        xc.wbs = []

        n_subjects = conns[0].shape[0]

        xc.metadata = {}

        dataset_name = "HCP" if hcp else "1000 Brains"

        for i in range(n_subjects):
            subject_id = f"subj_{i:04d}"
            xc.metadata[subject_id] = SubjectMetadata(
                subject_id=subject_id,
                dataset=dataset_name,
                parcellation=parcs[0] if parcs else "unknown",
                demographics={},
                clinical={},
                acquisition={},
                quality_metrics={},
            )

        import warnings

        warnings.warn(
            "EBRAINS KG zip files do not contain demographic/clinical metadata. "
            "Metadata dictionaries are empty. To populate metadata, load from external sources."
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

        if config.source_type != "ebrains-kg":
            errors.append(
                f"Expected source_type 'ebrains-kg', got '{config.source_type}'"
            )

        if config.source_url:
            import os

            if not os.path.exists(config.source_url):
                errors.append(f"Source file not found: {config.source_url}")
            if not config.source_url.endswith(".zip"):
                warnings.append(f"Source file should be .zip, got {config.source_url}")

        if config.train_test_split is not None and config.train_test_split <= 0:
            errors.append("train_test_split must be positive")

        download_params = config.download_params
        if "hcp" in download_params and not isinstance(download_params["hcp"], bool):
            errors.append("download_params['hcp'] must be boolean")

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

        Note:
            This may need to download the data if source_url is not provided.
            For faster metadata retrieval without downloading, provide source_url.

            Warning: EBRAINS KG zip files do not contain demographic/clinical metadata.
            Subject metadata will be initialized but empty. To populate metadata,
            load from external sources and attach to XCode.metadata.
        """
        import tempfile
        import os

        hcp = config.download_params.get("hcp", False)
        skip_parcs = config.download_params.get("skip_parcs", "")

        if config.source_url and os.path.exists(config.source_url):
            zip_fname = config.source_url
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                zip_fname = temp_file.name
            try:
                self._download(zip_fname, hcp=hcp)
                parcs, means, conns = self._parse(
                    zip_fname, hcp=hcp, skip_parcs=skip_parcs, return_mean=False
                )
            finally:
                if os.path.exists(zip_fname):
                    os.remove(zip_fname)

        if config.source_url and os.path.exists(config.source_url):
            parcs, means, conns = self._parse(
                config.source_url, hcp=hcp, skip_parcs=skip_parcs, return_mean=False
            )

        n_subjects = conns[0].shape[0] if conns else 0
        n_parcellations = len(parcs)

        metadata = DatasetMetadata(
            name="HCP" if hcp else "1000 Brains",
            description=f"EBRAINS Knowledge Graph connectome dataset ({n_subjects} subjects, {n_parcellations} parcellations)",
            n_subjects=n_subjects,
            n_parcellations=n_parcellations,
            parcellation_names=parcs,
            subjects=[f"subj_{i:04d}" for i in range(n_subjects)],
            metadata_dict={
                "hcp": hcp,
                "skip_parcs": skip_parcs,
                "dataset_type": "connectome",
                "source": "EBRAINS Knowledge Graph",
                "has_demographic_metadata": False,
                "has_clinical_metadata": False,
            },
            source_info={
                "source_type": "ebrains-kg",
                "source_url": config.source_url or "downloaded",
                "format": "ebrains-kg",
                "metadata_source": "none (zip files do not contain metadata)",
            },
        )

        return metadata

    def _download(self, dl_fname: str, hcp: bool = False) -> None:
        """Download dataset from EBRAINS Knowledge Graph.

        Args:
            dl_fname: Destination file path
            hcp: Whether to download HCP dataset (vs 1000 Brains)
        """
        if hcp:
            url = "https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000059_Atlas_based_HCP_connectomes_v1.1_pub"
        else:
            url = "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/public/buckets/d-3f179784-194d-4795-9d8d-301b524ca00a"
        urllib.request.urlretrieve(url, dl_fname)

    def _parse(
        self,
        zip_fname: str,
        hcp: bool = False,
        skip_parcs: str = "",
        return_mean: bool = True,
    ) -> tuple:
        """Parse connectome count matrices from Knowledge Graph zip file.

        Args:
            zip_fname: Path to zip file
            hcp: Whether parsing HCP dataset (different structure)
            skip_parcs: Comma-separated parcellation names to skip
            return_mean: Whether to return mean-centered data

        Returns:
            Tuple of (parcs, means, conns)
        """
        kb_zip = zipfile.ZipFile(zip_fname)
        parc_zip_fnames = [
            _.filename for _ in kb_zip.filelist if _.filename.endswith(".zip")
        ]
        conns = []
        parcs = []
        means = []
        skip_list = [s.strip() for s in skip_parcs.split(",") if s.strip()]

        for l, parc_zip_fname in enumerate(tqdm.tqdm(parc_zip_fnames, ncols=60)):
            parc, _ = parc_zip_fname.split(".zip")
            nreg = int(parc.split("-")[0])

            if parc in skip_list:
                continue

            parcs.append(parc)
            with kb_zip.open(parc_zip_fname) as parc_zip_fd:
                parc_zip = zipfile.ZipFile(parc_zip_fd)
                ti, tj = np.triu_indices(nreg, k=1)
                ws = np.zeros((200 if hcp else 261, ti.size), "f")
                for i in range(ws.shape[0]):
                    counts_fname = f"{parc}/SC/{i + 1:04d}_1_Counts.csv"
                    if hcp:
                        counts_fname = (
                            f"{parc}/1StructuralConnectivity/{i:03d}/Counts.csv"
                        )
                    with parc_zip.open(counts_fname) as fd:
                        txt = fd.read().decode("ascii")
                        delim = "," if "," in txt else " "
                        mat = [
                            [float(_) for _ in line.split(delim)]
                            for line in txt.strip().split("\n")
                        ]
                        for line in mat:
                            assert len(line) == nreg
                        assert len(mat) == nreg
                        w = np.array(mat)
                        assert w.shape == (nreg, nreg), (
                            f"wrong shape {w.shape}!=({nreg},{nreg})"
                        )
                        ws[i] = w[ti, tj]
                        ws[i] -= ws[i].min()
                        ws[i] /= np.percentile(ws[i], 99)
                        ws[i] = np.clip(ws[i], 0.0, 1.0)
                        ws[i] = np.sqrt(ws[i])
                u = ws.mean(axis=0)
                ws -= u
                means.append(u)
                conns.append(ws)
        return parcs, means, conns


__all__ = ["EbrainsKGLoader"]
