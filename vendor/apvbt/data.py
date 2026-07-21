"""Data loading and management for cross-coder training.

This module handles loading connectome data from various sources,
combining datasets, and saving/loading XCode objects.
"""

import numpy as np
import pickle
import tempfile
import tqdm
import urllib.request
import zipfile
from typing import Dict, Any, List
from apvbt.datasets._base import SubjectMetadata


class XCode:
    """Cross-coder for multi-parcellation connectome data.

    This class manages connectome data across multiple brain parcellations
    and provides methods for loading, combining, and persisting data.

    Attributes:
        conns: List of connectome triu arrays per parcellation (mean-centered)
        means: Mean connectome per parcellation
        parcs: Parcellation names (e.g., '079-Shen2013', '150-Destrieux')
        tts: Train/test split index (number of training subjects)
        wbs: List of trained encoder-decoder weight/bias tuples per architecture
        metadata: Dictionary mapping subject_id to SubjectMetadata objects
    """

    wbs = None
    conns = None
    means = None
    parcs = None
    tts = None
    metadata = None

    @classmethod
    def from_conns_npz(cls, fname, tts=200):
        """Load connectomes from an npz file containing multiple parcellations.

        Args:
            fname: Path to npz file
            tts: Train/test split index

        Returns:
            XCode instance with loaded data
        """
        import jax.numpy as jp

        KB = np.load(fname, allow_pickle=True)
        parcs = []
        conns = []
        means = []
        for parc in KB.keys():
            if KB[parc].ndim == 3 and parc not in ("031-MIST",):
                parcs.append(parc)
                i, j = jp.triu_indices(KB[parc].shape[1], k=1)
                ctri = KB[parc][:, i, j]
                assert ctri.shape[0] == 274
                ctri = jp.sqrt(ctri)  # Spase scaling
                means.append(ctri.mean(axis=0))
                ctri -= ctri.mean(axis=0)
                conns.append(ctri)
        self = cls()
        self.conns = [jp.array(_.astype("f")) for _ in conns]
        self.means = means
        self.parcs = parcs
        self.tts = tts
        self.wbs = []
        return self

    @classmethod
    def from_kg(cls, zip_fname=None, tts=None, hcp=False, skip_parc=""):
        """Load connectomes from EBRAINS Knowledge Graph datasets.

        Args:
            zip_fname: Path to downloaded zip file (if None, will download)
            tts: Train/test split index (if None, uses half the subjects)
            hcp: Whether to load HCP dataset (vs 1000 Brains dataset)
            skip_parc: Comma-separated list of parcellations to skip

        Returns:
            XCode instance with loaded data
        """
        import jax.numpy as jp

        if zip_fname:
            parsed = cls._parse_counts_from_kg_zip(
                zip_fname, hcp=hcp, skip_parcs=skip_parc
            )
        else:
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                cls._download_kg_zip(temp_file.name, hcp=hcp)
                parsed = cls._parse_counts_from_kg_zip(temp_file.name, hcp=hcp)
        parcs, means, conns = parsed
        self = cls()
        self.conns = [jp.array(_.astype("f")) for _ in conns]
        self.means = [jp.array(_.astype("f")) for _ in means]
        self.parcs = parcs
        self.tts = tts or (conns[0].shape[0] // 2)
        self.wbs = []
        return self

    @staticmethod
    def _download_kg_zip(dl_fname, hcp=False):
        """Download dataset from EBRAINS Knowledge Graph.

        Args:
            dl_fname: Destination file path
            hcp: Whether to download HCP dataset (vs 1000 Brains)
        """
        # https://search.kg.ebrains.eu/instances/3f179784-194d-4795-9d8d-301b524ca00a
        if hcp:
            url = "https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000059_Atlas_based_HCP_connectomes_v1.1_pub"
        else:
            url = "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/public/buckets/d-3f179784-194d-4795-9d8d-301b524ca00a"
        urllib.request.urlretrieve(url, dl_fname)

    @staticmethod
    def _parse_counts_from_kg_zip(zip_fname, hcp=False, skip_parcs=""):
        """Parse connectome count matrices from Knowledge Graph zip file.

        Args:
            zip_fname: Path to zip file
            hcp: Whether parsing HCP dataset (different structure)
            skip_parcs: Comma-separated parcellation names to skip

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
        for l, parc_zip_fname in enumerate(tqdm.tqdm(parc_zip_fnames, ncols=60)):
            parc, _ = parc_zip_fname.split(".zip")
            nreg = int(parc.split("-")[0])
            # if parc in skip_parcs or nreg > 120:
            #     print('skip', parc)
            #     continue
            # if parc in ('031-MIST', '294-Julich-Brain'):  # inconsistent matrices
            #     continue
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
                        # np.loadtxt doesn't load from zip fd correctly?
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

    @classmethod
    def from_old_pkl(cls, fname="xencode.pkl"):
        """Load from old pickle format (backwards compatibility).

        Args:
            fname: Path to old pickle file

        Returns:
            XCode instance
        """
        self = cls()
        with open(fname, "rb") as fd:
            p = pickle.load(fd)
        self.wbs = p["all_wb1"]
        self.conns = p["conns"]
        self.means = p["means"]
        self.parcs = p["parcs"]
        self.tts = p["tts"]
        return self

    @classmethod
    def combine_xc(cls, xc1, xc2, shuffle=True):
        """Combine two XCode instances (e.g., HCP and 1KB datasets).

        Args:
            xc1: First XCode instance
            xc2: Second XCode instance
            shuffle: Whether to shuffle combined connectomes

        Returns:
            Combined XCode instance
        """
        import jax, jax.numpy as jp

        xch = cls()
        n1, n2 = xc1.conns[0].shape[0], xc2.conns[0].shape[0]
        nh = n1 + n2
        xch.conns = [jp.concat([_1, _2]) for _1, _2 in zip(xc1.conns, xc2.conns)]
        # shuffle connectomes for training
        xch.permkey = jax.random.PRNGKey(42)
        xch.permidx = jax.random.permutation(xch.permkey, jp.r_[:nh], independent=True)
        for i in range(len(xch.conns)):
            xch.conns[i] = xch.conns[i][xch.permidx]
        # combine the rest
        xch.means = [(_1 * n2 + _2 * n2) / nh for _1, _2 in zip(xc1.means, xc2.means)]
        xch.parcs = xc1.parcs
        xch.tts = nh // 2
        xch.wbs = []
        return xch

    def to_pkl(self, fname="xcode.pkl"):
        """Save XCode instance to pickle file.

        Args:
            fname: Destination file path
        """
        stuff = dict(
            conns=self.conns,
            means=self.means,
            parcs=self.parcs,
            tts=self.tts,
            wbs=self.wbs,
            metadata=self.metadata,
        )
        with open(fname, "wb") as fd:
            pickle.dump(stuff, fd)

    @classmethod
    def from_pkl(cls, fname="xcode.pkl"):
        """Load XCode instance from pickle file.

        Args:
            fname: Path to pickle file

        Returns:
            XCode instance
        """
        with open(fname, "rb") as fd:
            stuff = pickle.load(fd)
        self = cls()
        for key, val in stuff.items():
            setattr(self, key, val)
        return self

    def filter_by_field(self, field: str, value: Any) -> List[str]:
        """Filter subjects by a metadata field value.

        Args:
            field: Metadata field name (e.g., 'demographics.sex')
            value: Value to match

        Returns:
            List of subject IDs matching the criteria

        Example:
            >>> male_subjects = xcode.filter_by_field('demographics.sex', 'M')
        """
        if self.metadata is None:
            return []

        result = []
        field_parts = field.split(".")
        for subject_id, meta in self.metadata.items():
            try:
                current = meta
                for part in field_parts:
                    if part in ("demographics", "clinical", "acquisition"):
                        current = getattr(meta, part)
                    elif isinstance(current, dict):
                        current = current.get(part)
                if current == value:
                    result.append(subject_id)
            except (AttributeError, KeyError):
                continue
        return result

    def filter_by_range(self, field: str, min_val: float, max_val: float) -> List[str]:
        """Filter subjects by a numeric metadata field range.

        Args:
            field: Metadata field name (e.g., 'demographics.age')
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            List of subject IDs matching the criteria

        Example:
            >>> adult_subjects = xcode.filter_by_range('demographics.age', 18, 65)
        """
        if self.metadata is None:
            return []

        result = []
        field_parts = field.split(".")
        for subject_id, meta in self.metadata.items():
            try:
                current = meta
                for part in field_parts:
                    if part in ("demographics", "clinical", "acquisition"):
                        current = getattr(meta, part)
                    elif isinstance(current, dict):
                        current = current.get(part)
                if isinstance(current, (int, float)) and min_val <= current <= max_val:
                    result.append(subject_id)
            except (AttributeError, KeyError):
                continue
        return result

    def get_subject(self, subject_id: str) -> "SubjectMetadata":
        """Get metadata for a specific subject.

        Args:
            subject_id: Subject identifier

        Returns:
            SubjectMetadata object for the subject

        Raises:
            KeyError: If subject not found
        """
        if self.metadata is None:
            raise KeyError(f"No metadata available. Subject '{subject_id}' not found")
        if subject_id not in self.metadata:
            raise KeyError(f"Subject '{subject_id}' not found in metadata")
        return self.metadata[subject_id]

    def get_demographics_summary(self) -> Dict[str, Any]:
        """Get summary statistics for demographic information.

        Returns:
            Dictionary with demographic summaries:
            - n_subjects: Total number of subjects with metadata
            - age_stats: Mean, std, min, max age (if available)
            - sex_counts: Counts by sex (if available)
            - fields: List of available demographic fields

        Example:
            >>> summary = xcode.get_demographics_summary()
            >>> print(f"Mean age: {summary['age_stats']['mean']}")
        """
        if self.metadata is None:
            return {"n_subjects": 0, "fields": []}

        n_subjects = len(self.metadata)
        ages = []
        sex_counts = {}
        fields = set()

        for meta in self.metadata.values():
            fields.update(meta.demographics.keys())

            if "age" in meta.demographics:
                age = meta.demographics["age"]
                if isinstance(age, (int, float)):
                    ages.append(age)

            if "sex" in meta.demographics:
                sex = meta.demographics["sex"]
                sex_counts[sex] = sex_counts.get(sex, 0) + 1

        summary = {
            "n_subjects": n_subjects,
            "fields": sorted(list(fields)),
        }

        if ages:
            import numpy as np

            summary["age_stats"] = {
                "mean": float(np.mean(ages)),
                "std": float(np.std(ages)),
                "min": float(np.min(ages)),
                "max": float(np.max(ages)),
            }

        if sex_counts:
            summary["sex_counts"] = sex_counts

        return summary
