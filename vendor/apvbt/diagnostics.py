"""
Feature analysis and data quality validation module.

This module provides functions for analyzing feature covariance, sensitivity,
data quality validation, subject identification accuracy, and crosscoder
latent space quality assessment.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING, Union
from scipy import stats, linalg

if TYPE_CHECKING:
    from .data import XCode


def compute_feature_covariance(
    features: np.ndarray,
    params: Optional[np.ndarray] = None,
    by_param: bool = False
) -> Dict[str, Any]:
    """
    Compute feature covariance matrix and analyze structure.

    Args:
        features: Array of shape (n_samples, n_features)
        params: Optional parameter array (n_samples, n_params) for param-specific analysis
        by_param: If True, compute covariance per parameter quantile

    Returns:
        Dict with:
            cov_matrix: (n_features, n_features) covariance matrix
            eigenvalues: Eigenvalues of covariance matrix
            mean_cov: Mean of off-diagonal elements
            std_cov: Std of off-diagonal elements
            condition_number: Ratio of largest to smallest eigenvalue
            param_specific_cov: (if by_param) Covariance per parameter bin

    From: ppt2-k-per-parc.ipynb (cell 4-11)
    """
    features = np.asarray(features)

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    n_samples, n_features = features.shape

    # Compute covariance matrix
    cov_matrix = np.cov(features, rowvar=False)

    # Handle 1D input (np.cov returns scalar)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])

    # Eigenvalue decomposition
    if cov_matrix.shape[0] == 1 and cov_matrix.shape[1] == 1:
        # 1x1 matrix has single eigenvalue
        eigenvalues = np.array([cov_matrix[0, 0]])
    else:
        eigenvalues = linalg.eigvalsh(cov_matrix)

    # Mean and std of off-diagonal elements
    i_upper, j_upper = np.triu_indices(n_features, k=1)
    off_diag_cov = cov_matrix[i_upper, j_upper]
    if len(off_diag_cov) > 0:
        mean_cov = float(np.mean(off_diag_cov))
        std_cov = float(np.std(off_diag_cov))
    else:
        mean_cov = 0.0
        std_cov = 0.0

    # Condition number (largest/smallest eigenvalue ratio)
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    condition_number = max_eig / min_eig if min_eig > 1e-10 else np.inf

    result = {
        'cov_matrix': cov_matrix,
        'eigenvalues': eigenvalues,
        'mean_cov': float(mean_cov),
        'std_cov': float(std_cov),
        'condition_number': float(condition_number),
    }

    # Parameter-specific covariance if requested
    if by_param and params is not None:
        params = np.asarray(params)
        if params.ndim == 1:
            params = params.reshape(-1, 1)

        n_params = params.shape[1]

        # Bin parameters into quartiles
        result['param_specific_cov'] = {}
        for p_idx in range(n_params):
            param_vals = params[:, p_idx]
            quartiles = np.percentile(param_vals, [0, 25, 50, 75, 100])

            for q_idx in range(4):
                q_low, q_high = quartiles[q_idx], quartiles[q_idx + 1]
                mask = (param_vals >= q_low) & (param_vals < q_high)
                if np.sum(mask) > 1:
                    features_q = features[mask]
                    cov_q = np.cov(features_q, rowvar=False)
                    off_diag_q = cov_q[i_upper, j_upper]
                    result['param_specific_cov'][f'param_{p_idx}_q{q_idx}'] = {
                        'mean_off_diag': float(np.mean(off_diag_q)),
                        'std_off_diag': float(np.std(off_diag_q)),
                        'condition_number': float(np.linalg.cond(cov_q)),
                    }

    return result


def analyze_feature_sensitivity(
    features: np.ndarray,
    params: np.ndarray,
    method: str = 'correlation'
) -> Dict[str, Any]:
    """
    Analyze how features vary with parameters.

    Args:
        features: Array of shape (n_samples, n_features)
        params: Array of shape (n_samples, n_params)
        method: 'correlation' or 'regression' or 'mutual_info'

    Returns:
        Dict with:
            sensitivity_matrix: (n_params, n_features) sensitivity scores
            most_sensitive_features: Indices of most sensitive features per param
            param_ranking: Which params most affect features
            importance_scores: Overall feature importance

    From: ppt2-k-per-parc.ipynb (k vs ccsd analysis)
    """
    features = np.asarray(features)
    params = np.asarray(params)

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if params.ndim == 1:
        params = params.reshape(-1, 1)

    n_samples, n_features = features.shape
    n_params = params.shape[1]

    # Ensure shapes match
    if params.shape[0] != n_samples:
        raise ValueError(f"params and features must have same number of samples: {params.shape[0]} vs {n_samples}")

    sensitivity_matrix = np.zeros((n_params, n_features))

    if method == 'correlation':
        for p_idx in range(n_params):
            for f_idx in range(n_features):
                corr, _ = stats.pearsonr(params[:, p_idx], features[:, f_idx])
                sensitivity_matrix[p_idx, f_idx] = np.abs(corr)

    elif method == 'regression':
        from sklearn.linear_model import LinearRegression
        for p_idx in range(n_params):
            reg = LinearRegression()
            reg.fit(params[:, p_idx:p_idx+1], features)
            sensitivity_matrix[p_idx, :] = np.abs(reg.coef_.ravel())

    elif method == 'mutual_info':
        try:
            from sklearn.feature_selection import mutual_info_regression
            for f_idx in range(n_features):
                mi_scores = mutual_info_regression(params, features[:, f_idx])
                sensitivity_matrix[:, f_idx] = mi_scores
        except ImportError:
            raise ImportError("sklearn is required for mutual_info method")

    else:
        raise ValueError(f"Unknown method: {method}. Must be 'correlation', 'regression', or 'mutual_info'")

    # Most sensitive features per parameter
    most_sensitive_features = np.argmax(sensitivity_matrix, axis=1)

    # Parameter ranking (which params most affect features)
    param_ranking = np.argsort(np.mean(np.abs(sensitivity_matrix), axis=1))[::-1]

    # Overall feature importance
    importance_scores = np.mean(np.abs(sensitivity_matrix), axis=0)

    return {
        'sensitivity_matrix': sensitivity_matrix,
        'most_sensitive_features': most_sensitive_features,
        'param_ranking': param_ranking,
        'importance_scores': importance_scores,
        'method_used': method,
    }


def validate_feature_informativeness(
    features: np.ndarray,
    params: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Validate that features contain information about parameters.

    Args:
        features: Array of shape (n_samples, n_features)
        params: Array of shape (n_samples, n_params)
        threshold: Minimum mutual information or R² to be informative

    Returns:
        Dict with:
            mutual_info: MI between each feature and parameter
            variance_explained: PCA variance explained ratio
            is_informative: Boolean array per feature
            recommendations: List of recommendations

    From: ppt0-workflow.ipynb (comparing prior vs posterior distributions)
    """
    features = np.asarray(features)
    params = np.asarray(params)

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if params.ndim == 1:
        params = params.reshape(-1, 1)

    n_samples, n_features = features.shape
    n_params = params.shape[1]

    # Ensure shapes match
    if params.shape[0] != n_samples:
        raise ValueError(f"params and features must have same number of samples: {params.shape[0]} vs {n_samples}")

    # Compute mutual information between features and parameters
    mutual_info = np.zeros((n_features, n_params))
    try:
        from sklearn.feature_selection import mutual_info_regression
        for p_idx in range(n_params):
            mutual_info[:, p_idx] = mutual_info_regression(params[:, p_idx:p_idx+1], features[:, p_idx])
    except ImportError:
        # Fallback to correlation-based approximation
        for p_idx in range(n_params):
            for f_idx in range(n_features):
                corr, _ = stats.pearsonr(params[:, p_idx], features[:, f_idx])
                mutual_info[f_idx, p_idx] = corr ** 2

    # PCA variance explained
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(features)
    variance_explained = pca.explained_variance_ratio_

    # Determine if features are informative
    is_informative = np.any(mutual_info > threshold, axis=1)

    # Generate recommendations
    recommendations = []
    if np.mean(mutual_info) < threshold:
        recommendations.append("Features have low mutual information with parameters. Consider using different features.")
    if variance_explained[0] < 0.5:
        recommendations.append("First PC explains less than 50% variance. Feature space may be redundant.")
    if np.all(is_informative):
        recommendations.append("Features are informative. Proceed with SBI.")
    else:
        recommendations.append("Some features are not informative. Consider feature selection.")

    return {
        'mutual_info': mutual_info,
        'variance_explained': variance_explained,
        'is_informative': is_informative,
        'recommendations': recommendations,
        'threshold_used': threshold,
    }


def check_data_quality(
    data: np.ndarray,
    name: str = 'data',
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Check data for NaN, Inf, outliers, and consistency issues.

    Args:
        data: Input array to check
        name: Name of data for error messages
        threshold: Z-score threshold for outlier detection

    Returns:
        Dict with:
            nan_count: Number of NaN values
            inf_count: Number of Inf values
            outlier_count: Number of outliers
            outlier_indices: Indices of outliers
            quality_score: Quality score (0-1)
            warnings: List of warnings
            errors: List of errors

    From: step2-id-conn-parc.ipynb (NaN warnings from SBI)
    """
    data = np.asarray(data)
    nan_mask = np.isnan(data)
    inf_mask = np.isinf(data)
    total_elements = data.size

    # Count issues
    nan_count = np.sum(nan_mask)
    inf_count = np.sum(inf_mask)

    # Detect outliers using z-score
    warnings_list = []
    errors_list = []

    if total_elements > 0:
        valid_mask = ~(nan_mask | inf_mask)
        valid_data = data[valid_mask]

        if valid_data.size > 1:
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            if std_val > 1e-10:
                z_scores = np.abs((valid_data - mean_val) / std_val)
                outlier_mask = z_scores > threshold
                outlier_count = np.sum(outlier_mask)
                outlier_indices = np.where(valid_mask)[0][outlier_mask]

                if outlier_count > 0:
                    outlier_percent = 100 * outlier_count / valid_data.size
                    if outlier_percent > 10:
                        warnings_list.append(f"{name}: {outlier_percent:.1f}% outliers detected (threshold={threshold})")
            else:
                outlier_count = 0
                outlier_indices = np.array([], dtype=int)
        else:
            outlier_count = 0
            outlier_indices = np.array([], dtype=int)
    else:
        outlier_count = 0
        outlier_indices = np.array([], dtype=int)

    # Generate warnings
    if nan_count > 0:
        nan_percent = 100 * nan_count / total_elements
        if nan_percent > 1:
            errors_list.append(f"{name}: {nan_percent:.1f}% NaN values ({nan_count}/{total_elements})")
        else:
            warnings_list.append(f"{name}: {nan_count} NaN values")

    if inf_count > 0:
        inf_percent = 100 * inf_count / total_elements
        if inf_percent > 1:
            errors_list.append(f"{name}: {inf_percent:.1f}% Inf values ({inf_count}/{total_elements})")
        else:
            warnings_list.append(f"{name}: {inf_count} Inf values")

    # Compute quality score
    valid_count = total_elements - nan_count - inf_count
    if total_elements > 0:
        base_score = valid_count / total_elements
        outlier_penalty = outlier_count / valid_count if valid_count > 0 else 0
        quality_score = max(0.0, min(1.0, base_score - 0.5 * outlier_penalty))
    else:
        quality_score = 1.0

    return {
        'nan_count': int(nan_count),
        'inf_count': int(inf_count),
        'outlier_count': int(outlier_count),
        'outlier_indices': outlier_indices,
        'quality_score': float(quality_score),
        'warnings': warnings_list,
        'errors': errors_list,
        'total_elements': int(total_elements),
        'threshold_used': threshold,
    }


def compute_subject_identification_accuracy(
    corr_matrix: np.ndarray
) -> float:
    """
    Compute how well features can distinguish between different subjects.

    Args:
        corr_matrix: Inter-subject correlation matrix (n_subjects, n_subjects)

    Returns:
        Accuracy: Fraction of subjects correctly identified (0-1)

    From: ppt0-workflow.ipynb (cell 8)
    """
    corr_matrix = np.asarray(corr_matrix)

    if corr_matrix.ndim != 2 or corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("corr_matrix must be square with shape (n_subjects, n_subjects)")

    n_subjects = corr_matrix.shape[0]

    # Find max correlation for each subject (excluding diagonal)
    corr_matrix_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_no_diag, -np.inf)

    # Count how many subjects have max correlation on diagonal
    correctly_identified = np.sum(np.argmax(corr_matrix, axis=1) == np.arange(n_subjects))

    # Compute accuracy
    accuracy = correctly_identified / n_subjects if n_subjects > 0 else 0.0

    return float(accuracy)


def evaluate_latent_space_quality(
    xc: 'XCode',
    arch: int = 8,
    tts: Optional[float] = None,
    n_folds: int = 1,
    key: Any = None
) -> Dict[str, Any]:
    """
    Evaluate crosscoder latent space quality.

    This function assesses how well the crosscoder captures connectome
    structure in the latent space and how well it generalizes.

    Args:
        xc: XCode object with connectome data and trained crosscoder
        arch: Latent dimensionality (number of components)
        tts: Train/test split ratio (0.0-1.0). If None, uses xc.tts
        n_folds: Number of cross-validation folds (1 = single split)
        key: JAX random key for reproducibility

    Returns:
        Dict with:
            latent_variance: np.ndarray (arch,) - variance per component
            latent_explained_ratio: float - total variance explained
            reconstruction_error_train: float - MSE on training set
            reconstruction_error_test: float - MSE on test set
            generalization_gap: float - test - train error
            cross_parc_transfer: Dict[str, float] - transfer learning quality per parc
            optimal_dimensionality: int - suggested arch via elbow method
            subject_separability: float - mean pairwise latent distance
            quality_score: float - overall quality metric (0-1)

    From: ppt3-latent-components.ipynb (cells 4-12)
    """
    import jax.numpy as jp
    
    tts = tts if tts is not None else getattr(xc, 'tts', 0.8)
    
    # Check if crosscoder is trained for this architecture
    arch_list = getattr(xc, 'arch', [])
    if arch not in arch_list:
        raise ValueError(f"Architecture {arch} not trained. Available: {arch_list}")
    
    iarch = arch_list.index(arch)
    wbs = xc.wbs[iarch]
    
    # Encode all connectomes to latent space
    us_list = []
    for ((ew, eb), _), c in zip(wbs, xc.conns):
        us_list.append(jp.array(c[tts:] @ ew + eb))
    us = jp.array(us_list)
    us_flat = us.reshape(-1, us.shape[-1])
    
    # Compute latent variance per component
    latent_variance = np.array(jp.var(us_flat, axis=0))
    total_variance = np.sum(latent_variance)
    latent_explained_ratio = float(total_variance / (total_variance + 1e-10))
    
    # Compute reconstruction errors
    train_conns = [_[:tts] for _ in xc.conns]
    test_conns = [_[tts:] for _ in xc.conns]
    
    train_errors = []
    test_errors = []
    
    for ((ew, eb), (dw, db)), mean, c_train, c_test in zip(wbs, xc.means, train_conns, test_conns):
        # Train reconstruction
        u_train = c_train @ ew + eb
        recon_train = u_train @ dw + db + mean
        train_mse = float(np.mean((recon_train - c_train - mean) ** 2))
        train_errors.append(train_mse)
        
        # Test reconstruction
        u_test = c_test @ ew + eb
        recon_test = u_test @ dw + db + mean
        test_mse = float(np.mean((recon_test - c_test - mean) ** 2))
        test_errors.append(test_mse)
    
    reconstruction_error_train = float(np.mean(train_errors))
    reconstruction_error_test = float(np.mean(test_errors))
    generalization_gap = reconstruction_error_test - reconstruction_error_train
    
    # Cross-parcellation transfer quality
    cross_parc_transfer = {}
    for i, parc in enumerate(xc.parcs):
        cross_parc_transfer[parc] = 1.0 / (1.0 + test_errors[i])
    
    # Compute optimal dimensionality via elbow method
    sorted_variance = np.sort(latent_variance)[::-1]
    cumsum = np.cumsum(sorted_variance)
    cumsum_norm = cumsum / cumsum[-1] if cumsum[-1] > 0 else cumsum
    
    # Find elbow point (where cumulative explained variance reaches 90%)
    optimal_dimensionality = arch
    for i, cv in enumerate(cumsum_norm):
        if cv >= 0.9:
            optimal_dimensionality = i + 1
            break
    
    # Subject separability (mean pairwise latent distance)
    n_subjects = us_flat.shape[0]
    if n_subjects > 1:
        # Sample pairwise distances to avoid O(n^2) computation
        n_samples = min(100, n_subjects)
        indices = np.random.choice(n_subjects, size=min(n_samples * 2, n_subjects), replace=False)
        sampled_us = np.array(us_flat[indices[:n_samples]])
        distances = []
        for i in range(len(sampled_us)):
            for j in range(i + 1, len(sampled_us)):
                distances.append(np.linalg.norm(sampled_us[i] - sampled_us[j]))
        subject_separability = float(np.mean(distances)) if distances else 0.0
    else:
        subject_separability = 0.0
    
    # Overall quality score (0-1)
    # Based on: low generalization gap, high separability, good reconstruction
    quality_components = []
    quality_components.append(1.0 / (1.0 + generalization_gap * 100))  # Reconstruction quality
    quality_components.append(min(1.0, subject_separability))  # Separability
    quality_components.append(1.0 / (1.0 + reconstruction_error_test * 10))  # Test error
    quality_score = float(np.mean(quality_components))
    
    return {
        'latent_variance': latent_variance,
        'latent_explained_ratio': latent_explained_ratio,
        'reconstruction_error_train': reconstruction_error_train,
        'reconstruction_error_test': reconstruction_error_test,
        'generalization_gap': generalization_gap,
        'cross_parc_transfer': cross_parc_transfer,
        'optimal_dimensionality': int(optimal_dimensionality),
        'subject_separability': subject_separability,
        'quality_score': quality_score,
        'architecture': arch,
        'n_parcellations': len(xc.parcs),
    }


def compute_latent_component_importance(
    wbs: List,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Analyze importance of each latent component.

    Args:
        wbs: Trained crosscoder weights from make_wbs() for a single architecture
             Format: List of ((enc_w, enc_b), (dec_w, dec_b)) per parcellation
        threshold: Minimum importance to be considered significant

    Returns:
        Dict with:
            component_importance: np.ndarray - importance per component
            active_components: List[int] - indices of important components
            effective_dimensionality: int - number of active components
            cumulative_importance: np.ndarray - cumulative explained variance
            importance_per_parc: Dict[int, np.ndarray] - importance per parcellation

    From: ppt3-latent-components.ipynb (component analysis)
    """
    # Get latent dimension from first decoder weights
    _, (dw, _) = wbs[0]
    n_components = dw.shape[0]
    
    # Compute importance per parcellation
    importance_per_parc = {}
    for i, ((_, _), (dw, _)) in enumerate(wbs):
        # Use decoder weight norms as importance measure
        # Higher norm = more contribution to reconstruction
        norms = np.linalg.norm(np.array(dw), axis=1)
        importance_per_parc[i] = norms
    
    # Aggregate importance across parcellations
    importance_matrix = np.array([importance_per_parc[i] for i in range(len(wbs))])
    component_importance = np.mean(importance_matrix, axis=0)
    
    # Normalize to sum to 1
    total_importance = np.sum(component_importance)
    if total_importance > 0:
        component_importance = component_importance / total_importance
    
    # Find active components
    active_mask = component_importance >= threshold
    active_components = np.where(active_mask)[0].tolist()
    effective_dimensionality = len(active_components)
    
    # Cumulative importance
    sorted_importance = np.sort(component_importance)[::-1]
    cumulative_importance = np.cumsum(sorted_importance)
    
    return {
        'component_importance': component_importance,
        'active_components': active_components,
        'effective_dimensionality': effective_dimensionality,
        'cumulative_importance': cumulative_importance,
        'importance_per_parc': importance_per_parc,
        'threshold_used': threshold,
        'n_components': n_components,
    }


def validate_regime_selection(
    params: np.ndarray,
    features: np.ndarray,
    method: str = 'covariance',
    **kwargs
) -> Dict[str, Any]:
    """
    Validate that selected regime enables parameter identification.

    This function wraps assess_regime() to provide validation with
    recommendations for parameter adjustments.

    Args:
        params: Parameter samples (n_samples, n_params)
        features: Feature samples (n_samples, n_features)
        method: Assessment method ('covariance', 'pca', 'mi', 'gradient')
        **kwargs: Additional arguments for specific assessment methods

    Returns:
        Dict with:
            is_valid: Whether regime is suitable for SBI
            metric_value: Primary regime quality metric
            recommendation: Suggested parameter adjustments
            confidence: Confidence in assessment (0-1)
            details: Full assessment details from assess_regime()

    From: ppt0-workflow.ipynb (regime checks before SBI)
    """
    from .regimes import assess_regime

    params = np.asarray(params)
    features = np.asarray(features)

    if params.ndim == 1:
        params = params.reshape(-1, 1)
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Ensure features has at least 2 columns for covariance computation
    if features.shape[1] < 2:
        features = np.column_stack([features, features + np.random.randn(*features.shape) * 1e-6])

    n_samples, n_params = params.shape
    n_features = features.shape[1]

    details = assess_regime(params, features, method=method, **kwargs)

    is_valid = False
    metric_value = 0.0
    confidence = 0.0
    recommendation = ""

    if method == 'covariance':
        mean_cov = details.get('mean_cov', 0)
        std_cov = details.get('std_cov', 0)
        is_saturated = details.get('is_saturated', True)

        metric_value = std_cov
        is_valid = not is_saturated and std_cov > 0.1

        if is_valid:
            confidence = min(1.0, std_cov / 0.3)
            recommendation = "Regime is suitable for SBI. Features show good variability."
        else:
            if is_saturated:
                confidence = 0.9
                recommendation = "Regime is SATURATED. Increase noise (D) or reduce coupling (k) to create more diverse dynamics."
            else:
                confidence = 0.5
                recommendation = "Feature variability is low. Consider adjusting dynamical parameters."

    elif method == 'pca':
        first_pc_var = details.get('first_pc_variance', 1.0)
        is_saturated = details.get('is_saturated', True)

        metric_value = 1.0 - first_pc_var
        is_valid = not is_saturated

        if is_valid:
            confidence = min(1.0, metric_value / 0.5)
            recommendation = "Regime shows good feature diversity. Ready for SBI."
        else:
            confidence = 0.8
            recommendation = "Features dominated by single component. Adjust parameters to increase diversity."

    elif method == 'mi':
        mi_mean = details.get('mi_mean', 0)
        is_saturated = details.get('is_saturated', True)
        threshold = details.get('threshold', 0.1)

        metric_value = mi_mean
        is_valid = not is_saturated

        if is_valid:
            confidence = min(1.0, mi_mean / threshold)
            recommendation = "Good mutual information between parameters and features. Proceed with SBI."
        else:
            confidence = 0.7
            recommendation = "Low mutual information. Features may not be informative about parameters."

    elif method == 'gradient':
        mean_grad_norm = details.get('mean_gradient_norm', 0)
        is_saturated = details.get('is_saturated', True)

        metric_value = mean_grad_norm
        is_valid = not is_saturated

        if is_valid:
            confidence = min(1.0, mean_grad_norm / 0.01)
            recommendation = "Good sensitivity to parameter changes. Regime is suitable."
        else:
            confidence = 0.6
            recommendation = "Low gradient sensitivity. Parameters have minimal effect on features."

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'is_valid': is_valid,
        'metric_value': float(metric_value),
        'recommendation': recommendation,
        'confidence': float(confidence),
        'details': details,
        'method': method,
        'n_samples': n_samples,
        'n_params': n_params,
        'n_features': n_features,
    }


def search_optimal_regime(
    xc: 'XCode',
    model,
    parc: str,
    k_range: Tuple[float, float] = (0.01, 10.0),
    D_range: Tuple[float, float] = (0.001, 1.0),
    n_samples: int = 20,
    n_sims: int = 50,
    metric: str = 'ccsd',
    method: str = 'covariance',
    key: Any = None
) -> Dict[str, Any]:
    """
    Search for optimal (k, D) regime that maximizes feature variability.

    This function performs a grid search over (k, D) parameter space to find
    the regime that produces the most discriminable features for SBI.

    Args:
        xc: XCode object with connectome data
        model: Dynamics model function
        parc: Parcellation name
        k_range: (k_min, k_max) range for coupling strength (log space)
        D_range: (D_min, D_max) range for noise intensity
        n_samples: Number of samples per dimension for grid search
        n_sims: Number of simulations per (k, D) point
        metric: Optimization metric ('ccsd', 'subject_id_accuracy', 'regime_quality')
        method: Regime assessment method for 'regime_quality' metric
        key: JAX random key

    Returns:
        Dict with:
            optimal_k: Optimal coupling strength
            optimal_D: Optimal noise level
            metric_surface: (n_k, n_D) metric values
            regime_quality: Quality assessment at optimum
            k_values: Array of tested k values
            D_values: Array of tested D values
            search_summary: Summary of search results

    From: ppt2-k-per-parc.ipynb (k optimization)
    """
    import jax.numpy as jp

    k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_samples)
    D_values = np.logspace(np.log10(D_range[0]), np.log10(D_range[1]), n_samples)

    metric_surface = np.zeros((len(k_values), len(D_values)))

    w = xc.get_conn(parc)
    n_test = min(n_sims, len(w))

    best_metric = -np.inf
    optimal_k = k_values[0]
    optimal_D = D_values[0]

    for i, k in enumerate(k_values):
        for j, D in enumerate(D_values):
            try:
                xf = model(w[:n_test], k, D, use_pmap=False)

                xf_np = np.array(xf)

                if metric == 'ccsd':
                    corr = np.corrcoef(xf_np)
                    if corr.ndim == 2 and corr.shape[0] > 1:
                        np.fill_diagonal(corr, 0)
                        metric_surface[i, j] = np.std(corr[np.triu_indices(len(corr), k=1)])
                    else:
                        metric_surface[i, j] = 0

                elif metric == 'subject_id_accuracy':
                    corr = np.corrcoef(xf_np)
                    if corr.ndim == 2 and corr.shape[0] > 1:
                        metric_surface[i, j] = compute_subject_identification_accuracy(corr)
                    else:
                        metric_surface[i, j] = 0

                elif metric == 'regime_quality':
                    params = np.column_stack([
                        np.full(n_test, k),
                        np.full(n_test, D)
                    ])
                    validation = validate_regime_selection(params, xf_np, method=method)
                    metric_surface[i, j] = validation['metric_value']

                else:
                    raise ValueError(f"Unknown metric: {metric}")

                if metric_surface[i, j] > best_metric:
                    best_metric = metric_surface[i, j]
                    optimal_k = k
                    optimal_D = D

            except Exception:
                metric_surface[i, j] = 0

    params = np.column_stack([
        np.full(n_test, optimal_k),
        np.full(n_test, optimal_D)
    ])
    try:
        xf = model(w[:n_test], optimal_k, optimal_D, use_pmap=False)
        xf_np = np.array(xf)
        regime_quality = validate_regime_selection(params, xf_np, method='covariance')
    except Exception:
        regime_quality = {'is_valid': False, 'metric_value': 0.0}

    search_summary = {
        'best_metric': float(best_metric),
        'optimal_k': float(optimal_k),
        'optimal_D': float(optimal_D),
        'search_space_size': int(n_samples * n_samples),
        'metric_used': metric,
    }

    return {
        'optimal_k': float(optimal_k),
        'optimal_D': float(optimal_D),
        'metric_surface': metric_surface,
        'regime_quality': regime_quality,
        'k_values': k_values,
        'D_values': D_values,
        'search_summary': search_summary,
        'parcellation': parc,
    }


def benchmark_crosscoder_architectures(
    xc: 'XCode',
    arch_range: Optional[List[int]] = None,
    n_iter: int = 200,
    lr: float = 3e-4,
    metric: str = 'reconstruction_error',
    tts: Optional[float] = None
) -> Dict[str, Any]:
    """
    Compare crosscoder performance across architectures.

    This function trains crosscoders with different latent dimensions
    and evaluates their performance.

    Args:
        xc: XCode object with connectome data
        arch_range: List of latent dimensions to test (default: [4, 8, 16, 32])
        n_iter: Number of training iterations per architecture
        lr: Learning rate
        metric: Metric to optimize ('reconstruction_error', 'quality_score')
        tts: Train/test split ratio. If None, uses xc.tts

    Returns:
        Dict with:
            architectures: List[int] - tested architectures
            train_errors: Dict[int, float] - train error per arch
            test_errors: Dict[int, float] - test error per arch
            quality_scores: Dict[int, float] - overall quality per arch
            optimal_arch: int - best performing architecture
            elbow_point: int - elbow method suggestion
            recommendation: str - architectural recommendation
            traces: Dict[int, List] - training traces per architecture

    From: ppt3-latent-components.ipynb (architecture sweep)
    """
    import jax.numpy as jp
    
    arch_range = arch_range or [4, 8, 16, 32]
    tts = tts if tts is not None else getattr(xc, 'tts', 0.8)
    
    # Store original wbs state
    original_wbs = list(xc.wbs) if hasattr(xc, 'wbs') and xc.wbs else []
    original_arch = list(xc.arch) if hasattr(xc, 'arch') else []
    
    results = {
        'architectures': [],
        'train_errors': {},
        'test_errors': {},
        'quality_scores': {},
        'optimal_arch': None,
        'elbow_point': None,
        'recommendation': '',
        'traces': {},
    }
    
    try:
        for arch in arch_range:
            # Check if already trained
            if arch in original_arch:
                # Use existing trained model
                iarch = original_arch.index(arch)
                wbs = original_wbs[iarch]
                trace = []
            else:
                # Train new model
                trace, wbs, _ = xc.train(arch, lr=lr, niter=n_iter, tts=tts)
            
            # Evaluate
            train_conns = [_[:tts] for _ in xc.conns]
            test_conns = [_[tts:] for _ in xc.conns]
            
            train_errors = []
            test_errors = []
            
            for ((ew, eb), (dw, db)), mean, c_train, c_test in zip(wbs, xc.means, train_conns, test_conns):
                # Train error
                u_train = c_train @ ew + eb
                recon_train = u_train @ dw + db + mean
                train_mse = float(np.mean((recon_train - c_train - mean) ** 2))
                train_errors.append(train_mse)
                
                # Test error
                u_test = c_test @ ew + eb
                recon_test = u_test @ dw + db + mean
                test_mse = float(np.mean((recon_test - c_test - mean) ** 2))
                test_errors.append(test_mse)
            
            mean_train_error = float(np.mean(train_errors))
            mean_test_error = float(np.mean(test_errors))
            
            # Compute quality score
            generalization_gap = mean_test_error - mean_train_error
            quality_score = 1.0 / (1.0 + generalization_gap * 100 + mean_test_error * 10)
            
            results['architectures'].append(arch)
            results['train_errors'][arch] = mean_train_error
            results['test_errors'][arch] = mean_test_error
            results['quality_scores'][arch] = quality_score
            results['traces'][arch] = trace
        
        # Find optimal architecture
        if metric == 'reconstruction_error':
            optimal_arch = min(results['test_errors'], key=results['test_errors'].get)
        else:  # quality_score
            optimal_arch = max(results['quality_scores'], key=results['quality_scores'].get)
        
        results['optimal_arch'] = optimal_arch
        
        # Find elbow point (diminishing returns)
        test_errors_list = [results['test_errors'][a] for a in arch_range]
        improvements = np.diff(test_errors_list)
        
        elbow_point = arch_range[0]
        for i, imp in enumerate(improvements):
            # Elbow when improvement is less than 10% of previous error
            if i > 0 and abs(imp) < 0.1 * abs(test_errors_list[i]):
                elbow_point = arch_range[i]
                break
            elbow_point = arch_range[i + 1]
        
        results['elbow_point'] = elbow_point
        
        # Generate recommendation
        if optimal_arch == elbow_point:
            results['recommendation'] = f"Use architecture {optimal_arch} (optimal and elbow point match)"
        elif optimal_arch < elbow_point:
            results['recommendation'] = f"Use architecture {optimal_arch} (optimal by {metric})"
        else:
            results['recommendation'] = f"Consider architecture {elbow_point} (elbow) or {optimal_arch} (optimal)"
        
    finally:
        # Restore original state
        xc.wbs = original_wbs
        # Note: arch property is computed from wbs, so it auto-restores
    
    return results
