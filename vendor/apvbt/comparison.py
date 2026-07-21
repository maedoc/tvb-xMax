"""
Statistical comparison utilities for algorithm and parcellation comparison.

This module provides functions for comparing SBI algorithms, distributions,
and computing statistical metrics like OK% and median statistics.
"""

import numpy as np
from typing import Tuple, Dict, Any
from scipy import stats


def compute_ok_metric(
    shrinkage: np.ndarray,
    z_score: np.ndarray
) -> float:
    """
    Compute OK% metric (parameters correctly inferred).

    OK% = percentage of parameters with:
        - Positive shrinkage (s > 0)
        - Within 95% CI (z < 1.96)

    Args:
        shrinkage: Posterior shrinkage array
        z_score: Posterior z-score array

    Returns:
        ok_percent: Percentage of parameters meeting both criteria (0-100)

    From: ppt0-workflow.ipynb, ppt1-bench-parcs.ipynb
    """
    ok = ((shrinkage > 0) * (z_score < 1.96)).mean() * 100
    return ok


def compute_median_metrics(
    shrinkage: np.ndarray,
    z_score: np.ndarray
) -> np.ndarray:
    """
    Compute median shrinkage and z-score (robust summary statistics).

    Args:
        shrinkage: Posterior shrinkage array
        z_score: Posterior z-score array

    Returns:
        Array of [median_shrinkage, median_z_score]

    From: ppt0-workflow.ipynb, ppt1-bench-parcs.ipynb
    """
    median_shrinkage = np.percentile(shrinkage.ravel(), 50)
    median_z_score = np.percentile(z_score.ravel(), 50)
    return np.array([median_shrinkage, median_z_score])


def paired_difference_test(
    data1: np.ndarray,
    data2: np.ndarray,
    test: str = 'wilcoxon',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test if two paired samples have different distributions.

    Args:
        data1: First sample array
        data2: Second sample array (paired with data1)
        test: Statistical test ('wilcoxon', 'ttest_rel', 'sign')
        alpha: Significance level

    Returns:
        Dict with:
            'statistic': Test statistic
            'p_value': P-value
            'is_significant': Whether difference is significant
            'effect_size': Effect size (if applicable)
            'confidence_interval': CI for effect size

    From: None (new capability for statistical testing)
    """
    data1 = np.asarray(data1).ravel()
    data2 = np.asarray(data2).ravel()

    if test == 'wilcoxon':
        stat, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
        effect_size = stat / float(len(data1)) if len(data1) > 0 else 0.0
        ci = stats.norm.interval(0.95, loc=effect_size, scale=1.0/np.sqrt(float(len(data1))))
    elif test == 'ttest_rel':
        stat, p_value = stats.ttest_rel(data1, data2)
        cohen_d = np.mean(data1 - data2) / np.std(data1 - data2, ddof=1)
        effect_size = cohen_d
        se = np.std(data1 - data2, ddof=1) / np.sqrt(float(len(data1)))
        ci = (effect_size - 1.96*se, effect_size + 1.96*se)
    elif test == 'sign':
        data_diff = data1 - data2
        n_pos = int(np.sum(data_diff > 0))
        n_neg = int(np.sum(data_diff < 0))
        stat = n_pos - n_neg
        min_count = min(n_pos, n_neg)
        binom_dist = stats.binom(n=len(data_diff), p=0.5)
        p_value = 2 * binom_dist.cdf(min_count)
        effect_size = stat / float(len(data_diff)) if len(data_diff) > 0 else 0.0
        ci = stats.norm.interval(0.95, loc=effect_size, scale=1.0/np.sqrt(float(len(data_diff))))
    else:
        raise ValueError(f"Unknown test: {test}. Must be 'wilcoxon', 'ttest_rel', or 'sign'")

    return {
        'statistic': stat,
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'effect_size': effect_size,
        'confidence_interval': ci,
        'test_used': test
    }


def compare_distributions(
    dist1: np.ndarray,
    dist2: np.ndarray,
    tests: list = ['ks', 'ttest', 'mannwhitney']
) -> Dict[str, Any]:
    """
    Compare two distributions using multiple tests.

    Args:
        dist1: First distribution
        dist2: Second distribution
        tests: List of tests to run

    Returns:
        Dict mapping test names to results with:
            'statistic', 'p_value', 'interpretation'

    From: None (new capability for distribution comparison)
    """
    dist1 = np.asarray(dist1).ravel()
    dist2 = np.asarray(dist2).ravel()
    results = {}

    if 'ks' in tests:
        stat, p_value = stats.ks_2samp(dist1, dist2)
        interpretation = 'different' if p_value < 0.05 else 'similar'
        results['ks'] = {
            'statistic': stat,
            'p_value': p_value,
            'interpretation': interpretation
        }

    if 'ttest' in tests:
        stat, p_value = stats.ttest_ind(dist1, dist2)
        interpretation = 'different' if p_value < 0.05 else 'similar'
        results['ttest'] = {
            'statistic': stat,
            'p_value': p_value,
            'interpretation': interpretation
        }

    if 'mannwhitney' in tests:
        stat, p_value = stats.mannwhitneyu(dist1, dist2, alternative='two-sided')
        interpretation = 'different' if p_value < 0.05 else 'similar'
        results['mannwhitney'] = {
            'statistic': stat,
            'p_value': p_value,
            'interpretation': interpretation
        }

    return results


def correct_multiple_comparisons(
    p_values: np.ndarray,
    method: str = 'fdr',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Correct p-values for multiple comparisons.

    Args:
        p_values: Array of p-values
        method: Correction method ('fdr', 'bonferroni', 'holm')
        alpha: Significance level

    Returns:
        Dict with:
            'corrected_p_values': Adjusted p-values
            'rejected_hypotheses': Boolean array of rejected tests
            'num_significant': Number of significant tests
            'method_used': Correction method applied

    From: None (new capability for multiple testing correction)
    """
    p_values = np.asarray(p_values).ravel()

    if method == 'fdr':
        try:
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        except ImportError:
            raise ImportError("statsmodels is required for FDR correction. Install with: pip install statsmodels")
    elif method == 'bonferroni':
        p_corrected = p_values * float(len(p_values))
        p_corrected = np.minimum(p_corrected, 1.0)
        rejected = p_corrected < alpha
    elif method == 'holm':
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        p_corrected = np.zeros_like(sorted_p)
        rejected = np.zeros_like(sorted_p, dtype=bool)
        for i, p in enumerate(sorted_p):
            p_corrected[i] = p * (len(p_values) - i)
            has_no_previous_rejection = not bool(np.any(rejected[:i]))
            if has_no_previous_rejection and float(p_corrected[i]) < alpha:
                rejected[i] = True
        p_corrected[sorted_indices] = p_corrected
        rejected[sorted_indices] = rejected
    else:
        raise ValueError(f"Unknown correction method: {method}. Must be 'fdr', 'bonferroni', or 'holm'")

    return {
        'corrected_p_values': p_corrected,
        'rejected_hypotheses': rejected,
        'num_significant': int(np.sum(rejected)),
        'method_used': method
    }


def compute_effect_size(
    data1: np.ndarray,
    data2: np.ndarray,
    method: str = 'cohens_d'
) -> Dict[str, Any]:
    """
    Compute effect size between two samples.

    Args:
        data1: First sample
        data2: Second sample
        method: Effect size method ('cohens_d', 'cliffs_delta')

    Returns:
        Dict with:
            'effect_size': Effect size value
            'magnitude': Interpretation ('small', 'medium', 'large')
            'confidence_interval': CI for effect size

    From: None (new capability for effect size)
    """
    data1 = np.asarray(data1).ravel()
    data2 = np.asarray(data2).ravel()

    if method == 'cohens_d':
        diff = np.mean(data1) - np.mean(data2)
        pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) +
                            (len(data2) - 1) * np.var(data2, ddof=1)) /
                           (len(data1) + len(data2) - 2))
        effect_size = diff / pooled_std if pooled_std > 0 else 0.0

        if abs(effect_size) < 0.2:
            magnitude = 'small'
        elif abs(effect_size) < 0.5:
            magnitude = 'medium'
        elif abs(effect_size) < 0.8:
            magnitude = 'large'
        else:
            magnitude = 'very large'

        se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
        ci = (effect_size - 1.96*se, effect_size + 1.96*se)

    elif method == 'cliffs_delta':
        n1 = len(data1)
        n2 = len(data2)

        def cliffs_delta(x, y):
            return (np.sum(x[:, None] > y[None, :]) -
                   np.sum(x[:, None] < y[None, :])) / (n1 * n2)

        effect_size = cliffs_delta(data1, data2)

        if abs(effect_size) < 0.147:
            magnitude = 'negligible'
        elif abs(effect_size) < 0.33:
            magnitude = 'small'
        elif abs(effect_size) < 0.474:
            magnitude = 'medium'
        else:
            magnitude = 'large'

        se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        ci = (effect_size - 1.96*se, effect_size + 1.96*se)
    else:
        raise ValueError(f"Unknown effect size method: {method}. Must be 'cohens_d' or 'cliffs_delta'")

    return {
        'effect_size': effect_size,
        'magnitude': magnitude,
        'confidence_interval': ci,
        'method_used': method
    }
