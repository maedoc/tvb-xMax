"""Dataset comparison utilities for APVBT.

This module provides functions for comparing datasets loaded via the dataset
loader framework. Includes metadata comparison, connectivity statistics,
and overlap analysis.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

from apvbt.datasets._base import DatasetMetadata, SubjectMetadata


@dataclass
class ComparisonResult:
    """Result of dataset comparison.

    Attributes:
        metadata_comparison: Dictionary comparing metadata fields
        subject_overlap: Number of overlapping subjects (if IDs available)
        parcellation_overlap: Number of overlapping parcellations
        connectivity_stats: Statistics of connectivity differences
        summary: Human-readable summary of differences
    """
    metadata_comparison: Dict[str, Any] = field(default_factory=dict)
    subject_overlap: Optional[int] = None
    parcellation_overlap: Optional[int] = None
    connectivity_stats: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


def compare_metadata(
    metadata1: DatasetMetadata,
    metadata2: DatasetMetadata,
) -> Dict[str, Any]:
    """Compare two dataset metadata objects.

    Args:
        metadata1: First dataset metadata
        metadata2: Second dataset metadata

    Returns:
        Dictionary with comparison results:
            - name_diff: Whether names differ
            - n_subjects_diff: Difference in subject counts
            - n_parcellations_diff: Difference in parcellation counts
            - parcellation_overlap: List of overlapping parcellation names
            - subject_overlap: List of overlapping subject IDs (if available)
            - field_comparison: Comparison of other metadata fields
    """
    # Basic field comparisons
    result = {
        "name_diff": metadata1.name != metadata2.name,
        "n_subjects_diff": metadata1.n_subjects - metadata2.n_subjects,
        "n_parcellations_diff": metadata1.n_parcellations - metadata2.n_parcellations,
        "parcellation_overlap": [],
        "subject_overlap": [],
        "field_comparison": {},
    }

    # Compare parcellation lists
    parcs1 = set(metadata1.parcellation_names)
    parcs2 = set(metadata2.parcellation_names)
    result["parcellation_overlap"] = list(parcs1.intersection(parcs2))
    result["parcellation_only_in_1"] = list(parcs1 - parcs2)
    result["parcellation_only_in_2"] = list(parcs2 - parcs1)

    # Compare subject lists (if available)
    if metadata1.subjects and metadata2.subjects:
        subjects1 = set(metadata1.subjects)
        subjects2 = set(metadata2.subjects)
        result["subject_overlap"] = list(subjects1.intersection(subjects2))
        result["subject_only_in_1"] = list(subjects1 - subjects2)
        result["subject_only_in_2"] = list(subjects2 - subjects1)

    # Compare metadata_dict fields
    all_keys = set(metadata1.metadata_dict.keys()) | set(metadata2.metadata_dict.keys())
    for key in all_keys:
        val1 = metadata1.metadata_dict.get(key)
        val2 = metadata2.metadata_dict.get(key)
        if val1 != val2:
            result["field_comparison"][key] = {
                "value1": val1,
                "value2": val2,
                "equal": val1 == val2,
            }

    return result


def compare_datasets(
    xc1: XCode,
    xc2: XCode,
    parcellation: Optional[str] = None,
) -> ComparisonResult:
    """Compare two XCode datasets.

    Compares metadata, subject overlap, parcellation overlap, and
    connectivity statistics for matching parcellations.

    Args:
        xc1: First XCode dataset
        xc2: Second XCode dataset
        parcellation: Specific parcellation to compare (if None, compare all)

    Returns:
        ComparisonResult with detailed comparison
    """
    # Validate inputs
    if xc1.parcs is None or xc2.parcs is None:
        raise ValueError("XCode objects must have parcs attribute (not None)")
    if xc1.conns is None or xc2.conns is None:
        raise ValueError("XCode objects must have conns attribute (not None)")
    # Type narrowing for type checker
    assert xc1.parcs is not None and xc2.parcs is not None
    assert xc1.conns is not None and xc2.conns is not None
    
    # Extract metadata from XCode objects
    # XCode doesn't have direct DatasetMetadata, so we create simplified versions
    metadata1 = DatasetMetadata(
        name="Dataset 1",
        description="",
        n_subjects=xc1.conns[0].shape[0] if xc1.conns else 0,
        n_parcellations=len(xc1.parcs) if xc1.parcs else 0,
        parcellation_names=xc1.parcs if xc1.parcs else [],
        subjects=list(xc1.metadata.keys()) if xc1.metadata else [],
        metadata_dict={},
        source_info={},
    )
    
    metadata2 = DatasetMetadata(
        name="Dataset 2",
        description="",
        n_subjects=xc2.conns[0].shape[0] if xc2.conns else 0,
        n_parcellations=len(xc2.parcs) if xc2.parcs else 0,
        parcellation_names=xc2.parcs if xc2.parcs else [],
        subjects=list(xc2.metadata.keys()) if xc2.metadata else [],
        metadata_dict={},
        source_info={},
    )
    
    # Compare metadata
    metadata_comparison = compare_metadata(metadata1, metadata2)
    
    # Determine which parcellations to compare
    if parcellation:
        if parcellation in xc1.parcs and parcellation in xc2.parcs:
            parcellations_to_compare = [parcellation]
        else:
            raise ValueError(f"Parcellation {parcellation} not found in both datasets")
    else:
        parcellations_to_compare = list(set(xc1.parcs).intersection(set(xc2.parcs)))
    
    # Compare connectivity statistics for overlapping parcellations
    connectivity_stats = {}
    for parc in parcellations_to_compare:
        idx1 = xc1.parcs.index(parc)
        idx2 = xc2.parcs.index(parc)
        
        conns1 = xc1.conns[idx1]
        conns2 = xc2.conns[idx2]
        
        # Basic statistics
        mean1 = np.mean(conns1, axis=0)
        mean2 = np.mean(conns2, axis=0)
        var1 = np.var(conns1, axis=0)
        var2 = np.var(conns2, axis=0)
        
        # Compute differences
        mean_diff = np.mean(np.abs(mean1 - mean2))
        var_diff = np.mean(np.abs(var1 - var2))
        correlation = np.corrcoef(mean1, mean2)[0, 1] if len(mean1) > 1 else 1.0
        
        connectivity_stats[parc] = {
            "mean_difference": float(mean_diff),
            "variance_difference": float(var_diff),
            "mean_correlation": float(correlation),
            "n_subjects_1": conns1.shape[0],
            "n_subjects_2": conns2.shape[0],
            "n_features": conns1.shape[1],
        }
    
    # Create summary
    summary_parts = []
    if metadata_comparison["n_subjects_diff"] != 0:
        diff = metadata_comparison["n_subjects_diff"]
        summary_parts.append(f"Subject count difference: {diff:+d}")
    
    if metadata_comparison["n_parcellations_diff"] != 0:
        diff = metadata_comparison["n_parcellations_diff"]
        summary_parts.append(f"Parcellation count difference: {diff:+d}")
    
    if parcellations_to_compare:
        summary_parts.append(f"Comparing {len(parcellations_to_compare)} overlapping parcellations")
        for parc, stats in connectivity_stats.items():
            summary_parts.append(
                f"  {parc}: mean diff={stats['mean_difference']:.3f}, "
                f"var diff={stats['variance_difference']:.3f}"
            )
    else:
        summary_parts.append("No overlapping parcellations to compare")
    
    summary = "\n".join(summary_parts)
    
    return ComparisonResult(
        metadata_comparison=metadata_comparison,
        subject_overlap=len(metadata_comparison.get("subject_overlap", [])),
        parcellation_overlap=len(metadata_comparison["parcellation_overlap"]),
        connectivity_stats=connectivity_stats,
        summary=summary,
    )


def compute_dataset_overlap(
    xc1: XCode,
    xc2: XCode,
) -> Dict[str, Any]:
    """Compute overlap between two datasets.

    Args:
        xc1: First XCode dataset
        xc2: Second XCode dataset

    Returns:
        Dictionary with overlap metrics:
            - parcellation_overlap: List of overlapping parcellation names
            - subject_overlap: List of overlapping subject IDs (if metadata available)
            - parcellation_jaccard: Jaccard index for parcellation overlap
            - subject_jaccard: Jaccard index for subject overlap (if available)
    """
    result = {}
    
    # Parcellation overlap
    parcs1 = set(xc1.parcs) if xc1.parcs else set()
    parcs2 = set(xc2.parcs) if xc2.parcs else set()
    
    overlap_parcs = list(parcs1.intersection(parcs2))
    union_parcs = list(parcs1.union(parcs2))
    
    result["parcellation_overlap"] = overlap_parcs
    result["parcellation_only_in_1"] = list(parcs1 - parcs2)
    result["parcellation_only_in_2"] = list(parcs2 - parcs1)
    result["parcellation_jaccard"] = (
        len(overlap_parcs) / len(union_parcs) if union_parcs else 0.0
    )
    
    # Subject overlap (if metadata available)
    if xc1.metadata and xc2.metadata:
        subjects1 = set(xc1.metadata.keys())
        subjects2 = set(xc2.metadata.keys())
        
        overlap_subjects = list(subjects1.intersection(subjects2))
        union_subjects = list(subjects1.union(subjects2))
        
        result["subject_overlap"] = overlap_subjects
        result["subject_only_in_1"] = list(subjects1 - subjects2)
        result["subject_only_in_2"] = list(subjects2 - subjects1)
        result["subject_jaccard"] = (
            len(overlap_subjects) / len(union_subjects) if union_subjects else 0.0
        )
    else:
        result["subject_overlap"] = []
        result["subject_jaccard"] = None
    
    return result


def compare_connectivity_distributions(
    conns1: np.ndarray,
    conns2: np.ndarray,
) -> Dict[str, float]:
    """Compare connectivity distributions between two arrays.

    Args:
        conns1: First connectivity array (n_subjects, n_features)
        conns2: Second connectivity array (n_subjects, n_features)

    Returns:
        Dictionary with comparison metrics:
            - mean_absolute_difference: Mean absolute difference between means
            - variance_ratio: Ratio of variances (var1 / var2)
            - correlation: Pearson correlation between means
            - kl_divergence: Approximate KL divergence (if similar dimensions)
    """
    # Ensure arrays are 2D
    if conns1.ndim == 1:
        conns1 = conns1.reshape(-1, 1)
    if conns2.ndim == 1:
        conns2 = conns2.reshape(-1, 1)
    
    # Compute statistics
    mean1 = np.mean(conns1, axis=0)
    mean2 = np.mean(conns2, axis=0)
    var1 = np.var(conns1, axis=0)
    var2 = np.var(conns2, axis=0)
    
    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(mean1 - mean2))
    
    # Variance ratio (avoid division by zero)
    var_ratio = np.mean(var1) / np.mean(var2) if np.mean(var2) > 0 else np.inf
    
    # Correlation between means
    if len(mean1) > 1 and np.std(mean1) > 0 and np.std(mean2) > 0:
        correlation = np.corrcoef(mean1, mean2)[0, 1]
    else:
        correlation = 1.0 if np.allclose(mean1, mean2) else 0.0
    
    # Approximate KL divergence (assuming Gaussian distributions)
    # KL = 0.5 * (tr(Σ2⁻¹Σ1) + (μ2-μ1)ᵀΣ2⁻¹(μ2-μ1) - k + ln(|Σ2|/|Σ1|))
    # Simplified: use diagonal covariance approximation
    eps = 1e-10
    kl = 0.5 * (
        np.sum(var1 / (var2 + eps)) +
        np.sum((mean2 - mean1)**2 / (var2 + eps)) -
        len(mean1) +
        np.sum(np.log((var2 + eps) / (var1 + eps)))
    )
    
    return {
        "mean_absolute_difference": float(mean_abs_diff),
        "variance_ratio": float(var_ratio),
        "correlation": float(correlation),
        "kl_divergence": float(kl),
        "n_subjects_1": conns1.shape[0],
        "n_subjects_2": conns2.shape[0],
        "n_features": conns1.shape[1],
    }


def get_demographics_comparison(
    xc1: XCode,
    xc2: XCode,
    field: str,
) -> Dict[str, Any]:
    """Compare demographics for a specific field across datasets.

    Args:
        xc1: First XCode dataset
        xc2: Second XCode dataset
        field: Demographics field to compare (e.g., 'age', 'sex')

    Returns:
        Dictionary with comparison metrics:
            - values_1: List of values from dataset 1
            - values_2: List of values from dataset 2
            - mean_1, mean_2: Means (if numeric)
            - distribution_1, distribution_2: Value counts (if categorical)
            - statistical_test: Result of appropriate statistical test
    """
    if not xc1.metadata or not xc2.metadata:
        return {"error": "No metadata available"}
    assert xc1.metadata is not None and xc2.metadata is not None
    
    # Extract values
    values1 = []
    values2 = []
    
    for meta in xc1.metadata.values():
        if field in meta.demographics:
            values1.append(meta.demographics[field])
    
    for meta in xc2.metadata.values():
        if field in meta.demographics:
            values2.append(meta.demographics[field])
    
    if not values1 or not values2:
        return {"error": f"Field '{field}' not found in both datasets"}
    
    # Determine if numeric or categorical
    float_vals1: List[float] = []
    float_vals2: List[float] = []
    is_numeric = False
    try:
        # Try to convert to float
        float_vals1 = [float(v) for v in values1]
        float_vals2 = [float(v) for v in values2]
        is_numeric = True
    except (ValueError, TypeError):
        pass
    
    result = {
        "field": field,
        "n_values_1": len(values1),
        "n_values_2": len(values2),
        "is_numeric": is_numeric,
    }
    
    if is_numeric:
        result["mean_1"] = float(np.mean(float_vals1))
        result["mean_2"] = float(np.mean(float_vals2))
        result["std_1"] = float(np.std(float_vals1))
        result["std_2"] = float(np.std(float_vals2))
        result["median_1"] = float(np.median(float_vals1))
        result["median_2"] = float(np.median(float_vals2))
        
        # T-test if enough samples
        if len(float_vals1) > 1 and len(float_vals2) > 1:
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(float_vals1, float_vals2)
                result["t_test"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                }
            except ImportError:
                result["t_test"] = "scipy not available"
    else:
        # Categorical data
        from collections import Counter
        counts1 = Counter(values1)
        counts2 = Counter(values2)
        
        result["distribution_1"] = dict(counts1)
        result["distribution_2"] = dict(counts2)
        
        # Chi-square test if enough samples
        all_categories = set(counts1.keys()) | set(counts2.keys())
        if len(all_categories) > 1:
            try:
                from scipy import stats
                # Create contingency table
                contingency = np.zeros((2, len(all_categories)))
                for i, cat in enumerate(all_categories):
                    contingency[0, i] = counts1.get(cat, 0)
                    contingency[1, i] = counts2.get(cat, 0)
                
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                result["chi_square_test"] = {
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                    "degrees_of_freedom": int(dof),
                }
            except ImportError:
                result["chi_square_test"] = "scipy not available"
    
    return result


__all__ = [
    "ComparisonResult",
    "compare_metadata",
    "compare_datasets",
    "compute_dataset_overlap",
    "compare_connectivity_distributions",
    "get_demographics_comparison",
]