"""Dynamical regime selection for SBI.

This module provides metrics to assess whether parameter variations
meaningfully affect simulation features - a critical prerequisite for SBI.

The goal is to identify parameter ranges where the parameter→feature mapping
is sensitive (not saturated), ensuring parameter identifiability.
"""

import numpy as np


def covariance_based_metric(features, plot=False):
    """Assess regime quality via feature covariance distribution.

    High correlation between features (→1.0) suggests parameters aren't
    creating diverse feature patterns (saturated regime).
    Broad distribution suggests good sensitivity.

    Args:
        features: Feature array (n_samples, n_features)
        plot: Whether to plot histogram (requires matplotlib)

    Returns:
        Dictionary with:
        - cov_matrix: Covariance matrix
        - cov_triu: Upper triangle values
        - mean_cov: Mean covariance
        - std_cov: Std of covariances
        - is_saturated: Boolean (True if mean > 0.9 and std < 0.1)
    """
    # Compute covariance matrix
    cov = np.cov(features.T)

    # Extract upper triangle (excluding diagonal)
    n = cov.shape[0]
    i, j = np.triu_indices(n, k=1)
    cov_triu = cov[i, j]

    # Statistics
    mean_cov = np.mean(cov_triu)
    std_cov = np.std(cov_triu)

    # Heuristic: saturated if narrow distribution near 1
    is_saturated = (mean_cov > 0.9) and (std_cov < 0.1)

    result = {
        'cov_matrix': cov,
        'cov_triu': cov_triu,
        'mean_cov': mean_cov,
        'std_cov': std_cov,
        'is_saturated': is_saturated,
    }

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        plt.hist(cov_triu, bins=50, density=True)
        plt.axvline(mean_cov, color='r', linestyle='--',
                   label=f'mean={mean_cov:.3f}')
        plt.xlabel('Covariance')
        plt.ylabel('Density')
        plt.title(f'Feature Covariance Distribution\n'
                 f'({"SATURATED" if is_saturated else "GOOD"} regime)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(122)
        plt.imshow(cov, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(label='Covariance')
        plt.title('Covariance Matrix')

        plt.tight_layout()
        plt.show()

    return result


def gradient_based_sensitivity(model, params, features_fn, eps=1e-5):
    """Assess regime via gradient magnitudes (requires JAX model).

    Small gradient norms indicate saturated regime where parameter
    changes don't affect features.

    Args:
        model: JAX-based model function
        params: Parameter values to evaluate at
        features_fn: Function to extract features from model output
        eps: Finite difference epsilon (if autodiff not available)

    Returns:
        Dictionary with gradient norms and sensitivity metrics

    Note: This is a stub - needs implementation with JAX autodiff
    """
    # TODO: Implement with jax.grad
    raise NotImplementedError(
        "Gradient-based sensitivity requires JAX autodiff implementation. "
        "Use jax.grad to compute ∂features/∂params and check gradient norms."
    )


def mutual_information_metric(params, features, k=5):
    """Estimate mutual information between parameters and features.

    Higher MI indicates more informative regime where features
    contain information about parameters.

    Args:
        params: Parameter samples (n_samples, n_params)
        features: Feature samples (n_samples, n_features)
        k: Number of neighbors for k-NN MI estimator

    Returns:
        Dictionary with MI estimate

    Note: This is a stub - requires implementation with MI estimator
    """
    # TODO: Implement with sklearn.feature_selection.mutual_info_regression
    # or dedicated MI library
    raise NotImplementedError(
        "MI-based metric requires implementation. "
        "Consider using sklearn.feature_selection.mutual_info_regression "
        "or dedicated MI estimation libraries."
    )


def pca_variance_metric(features, threshold=0.9):
    """Assess regime via PCA variance explained.

    If first PC explains >threshold variance with small magnitude,
    features aren't varying enough.

    Args:
        features: Feature array (n_samples, n_features)
        threshold: Threshold for cumulative variance explained

    Returns:
        Dictionary with PCA statistics
    """
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(features)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    # Find number of PCs needed to explain threshold variance
    n_components = np.searchsorted(cum_var, threshold) + 1

    # Check if variance is too concentrated
    is_saturated = (explained_var[0] > threshold) and (
        np.std(features) < 0.1)

    result = {
        'explained_variance_ratio': explained_var,
        'cumulative_variance': cum_var,
        'n_components_for_threshold': n_components,
        'first_pc_variance': explained_var[0],
        'is_saturated': is_saturated,
    }

    return result


def assess_regime(params, features, method='covariance', **kwargs):
    """Convenience function to assess dynamical regime quality.

    Args:
        params: Parameter samples (n_samples, n_params)
        features: Feature samples (n_samples, n_features)
        method: Which metric to use ('covariance', 'pca', 'mi', 'gradient')
        **kwargs: Additional arguments for specific methods

    Returns:
        Dictionary with assessment results

    Raises:
        ValueError: If method is not recognized
    """
    if method == 'covariance':
        return covariance_based_metric(features, **kwargs)
    elif method == 'pca':
        return pca_variance_metric(features, **kwargs)
    elif method == 'mi':
        return mutual_information_metric(params, features, **kwargs)
    elif method == 'gradient':
        return gradient_based_sensitivity(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from: covariance, pca, mi, gradient")
