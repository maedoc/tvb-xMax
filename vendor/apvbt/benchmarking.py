"""Multi-scale benchmarking workflows for brain dynamics models.

This module provides comprehensive benchmarking capabilities for:
- Multi-parcellation benchmarking across brain atlases
- Multi-algorithm comparison (MAF vs MDN)
- Parameter space optimization
- Regime analysis across dynamical parameters

Functions implement workflows from notebooks (ppt0-workflow, ppt1-bench-parcs, 
ppt2-k-per-parc) to ensure reproducibility of research results.
"""

import numpy as np
import tqdm
from typing import Dict, Any, List, Tuple, Optional, Union
from collections.abc import Callable

from .inference import run_sbi, posterior_diags
from .comparison import compute_ok_metric


def bench_model_multi_parc(
    xc,
    model: Callable,
    parcs: List[str],
    arch: int = 8,
    num_batch: int = 32,
    batch_size: int = 128,
    do_subjets: bool = False,
    num_postcd: int = 1000,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """Run benchmark across multiple parcellations.
    
    This function benchmarks SBI performance across multiple brain parcellations,
    comparing cohort-level and (optionally) subject-level inference quality.
    
    Args:
        xc: XCode object with connectome data
        model: Dynamics model function (e.g., hopf_dfun, mpr_dfun)
        parcs: List of parcellation names to benchmark
        arch: Number of latent components for cross-coder
        num_batch: Number of batches for SBI training
        batch_size: Batch size per training batch
        do_subjets: Whether to run subject-level SBI (computationally expensive)
        num_postcd: Number of posterior samples for diagnostics
        **kwargs: Additional arguments for bench_model
        Note: SBI algorithm is hardcoded to 'maf' in bench_model.
    
    Returns:
        Dict mapping parcellation names to benchmark results:
            {
                '079-Shen2013': {
                    'diags_cd': (shrinkage, z_score, ci90),
                    'diags_subj': (shrinkage, z_score, ci90) if do_subjets else None,
                    'metrics': {'ok_percent': ..., 'median_shrinkage': ..., 'median_z': ...}
                },
                ...
            }
    
    From: step2-id-conn-parc.ipynb (loop over parcellations)
    """
    results = {}
    
    for parc in tqdm.tqdm(parcs, desc="Benchmarking parcellations"):
        # Run benchmark for this parcellation
        # Note: bench_model uses default 'maf' algorithm
        diags_cd, diags_subj = bench_model(
            xc, model, parc=parc, arch=arch,
            num_batch=num_batch, batch_size=batch_size,
            do_subjets=do_subjets, num_postcd=num_postcd,
            prog=False, **kwargs
        )
        
        # Compute additional metrics
        shrinkage, z_score, ci90 = diags_cd
        ok_percent = compute_ok_metric(shrinkage, z_score)
        median_shrinkage = np.median(shrinkage)
        median_z = np.median(z_score)
        
        # Store results
        results[parc] = {
            'diags_cd': diags_cd,
            'diags_subj': diags_subj,
            'metrics': {
                'ok_percent': ok_percent,
                'median_shrinkage': median_shrinkage,
                'median_z': median_z,
                'mean_shrinkage': np.mean(shrinkage),
                'mean_z': np.mean(z_score),
                'ci90_coverage': np.mean(ci90)
            }
        }
    
    return results


def bench_model_multi_algo(
    theta: np.ndarray,
    features: np.ndarray,
    algos: List[str] = ['maf', 'mdn'],
    num_post_samples: int = 1000,
    test_theta: Optional[np.ndarray] = None,
    test_features: Optional[np.ndarray] = None,
    **sbi_kwargs
) -> Dict[str, Any]:
    """Train and compare multiple SBI algorithms on same data.
    
    This function trains multiple SBI algorithms (e.g., MAF and MDN) on the
    same training data and compares their performance on test data.
    
    Args:
        theta: Training parameter samples (n_samples, n_params)
        features: Training features (n_samples, n_features)
        algos: List of algorithms to compare
        num_post_samples: Number of posterior samples for evaluation
        test_theta: Test parameters for evaluation (default: use training set)
        test_features: Test features for evaluation (default: use training set)
        **sbi_kwargs: Additional arguments for run_sbi
    
    Returns:
        Dict with:
            'algorithms': List of algorithm names
            'posteriors': Dict of trained posteriors
            'diagnostics': Dict of diagnostic results per algorithm
                {
                    'maf': {'shrinkage': ..., 'z_score': ..., 'ci90': ..., ...},
                    'mdn': {'shrinkage': ..., 'z_score': ..., 'ci90': ..., ...}
                }
            'comparison': Statistical comparison between algorithms
    
    From: ppt1-bench-parcs.ipynb (running both MAF and MDN)
    """
    posteriors = {}
    diagnostics = {}
    
    # Use test data if provided, otherwise use training data
    if test_theta is None:
        test_theta = theta
    if test_features is None:
        test_features = features
    
    for algo in tqdm.tqdm(algos, desc="Training algorithms"):
        # Train posterior
        result = run_sbi(theta, features, prog=False, algo=algo, **sbi_kwargs)
        # Handle both old return (posterior) and new return (posterior, metadata)
        posterior = result[0] if isinstance(result, tuple) else result
        posteriors[algo] = posterior

        # Evaluate on test data
        # Sample from posterior for each test feature
        if hasattr(posterior, 'sample_batched'):
            samples = posterior.sample_batched(
                (num_post_samples,),
                x=np.array(test_features),
                show_progress_bars=False
            )
        else:
            samples = posterior.sample(
                (num_post_samples, len(test_features)),
                x=np.array(test_features),
                show_progress_bars=False
            )
        
        # Compute diagnostics
        shrinkage, z_score, ci90 = posterior_diags(test_theta, samples, test_theta)
        ok_percent = compute_ok_metric(shrinkage, z_score)
        
        diagnostics[algo] = {
            'shrinkage': shrinkage,
            'z_score': z_score,
            'ci90': ci90,
            'ok_percent': ok_percent,
            'median_shrinkage': np.median(shrinkage),
            'median_z': np.median(z_score),
            'mean_shrinkage': np.mean(shrinkage),
            'mean_z': np.mean(z_score)
        }
    
    return {
        'algorithms': algos,
        'posteriors': posteriors,
        'diagnostics': diagnostics,
        'comparison': {
            'comparison': 'algorithms compared',
            'metrics': list(diagnostics[algos[0]].keys())
        }
    }


def aggregate_parc_results(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['shrinkage', 'z_score']
) -> Dict[str, Any]:
    """Aggregate benchmark results across parcellations.
    
    This function summarizes benchmark results across multiple parcellations,
    computing means, standard deviations, medians, and rankings.
    
    Args:
        results_dict: Results from bench_model_multi_parc
            {
                '079-Shen2013': {'metrics': {'shrinkage': ..., 'z_score': ...}},
                ...
            }
        metrics: List of metrics to aggregate
    
    Returns:
        Dict with:
            'means': Mean per metric across parcellations
            'stds': Std per metric across parcellations
            'medians': Median per metric across parcellations
            'best_parc': Best parcellation per metric
            'worst_parc': Worst parcellation per metric
            'rankings': Ranked list per metric
    
    From: step2-id-conn-parc.ipynb (computing average z-scores and shrinkage)
    """
    aggregated = {
        'means': {},
        'stds': {},
        'medians': {},
        'best_parc': {},
        'worst_parc': {},
        'rankings': {}
    }
    
    for metric in metrics:
        # Extract values for this metric
        values = {}
        for parc, result in results_dict.items():
            if metric in result['metrics']:
                values[parc] = result['metrics'][metric]
        
        if not values:
            continue
        
        # Compute statistics
        vals_array = np.array(list(values.values()))
        aggregated['means'][metric] = np.mean(vals_array)
        aggregated['stds'][metric] = np.std(vals_array)
        aggregated['medians'][metric] = np.median(vals_array)
        
        # Find best and worst (assuming higher is better for most metrics)
        parcs_list = list(values.keys())
        sorted_indices = np.argsort(vals_array)
        aggregated['worst_parc'][metric] = parcs_list[sorted_indices[0]]
        aggregated['best_parc'][metric] = parcs_list[sorted_indices[-1]]
        
        # Create ranking
        rankings = sorted(
            values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        aggregated['rankings'][metric] = [
            {'parcellation': p, 'value': v, 'rank': i+1}
            for i, (p, v) in enumerate(rankings)
        ]
    
    return aggregated


def rank_parcellations(
    results_dict: Dict[str, Dict[str, Any]],
    metric: str = 'z_score',
    ascending: bool = True
) -> List[Tuple[str, float, int]]:
    """Rank parcellations by performance metric.
    
    Args:
        results_dict: Results from bench_model_multi_parc
        metric: Metric to rank by ('ok_percent', 'shrinkage', 'z_score')
        ascending: Sort direction (True for ascending, False for descending)
    
    Returns:
        List of (parc_name, metric_value, rank) tuples
    
    From: step2-id-conn-parc.ipynb (identifying best/worst parcellations)
    """
    # Extract metric values
    values = []
    for parc, result in results_dict.items():
        if metric in result['metrics']:
            values.append((parc, result['metrics'][metric]))
    
    # Sort by metric value
    values.sort(key=lambda x: x[1], reverse=not ascending)
    
    # Add ranks
    ranked = [
        (parc, val, i+1)
        for i, (parc, val) in enumerate(values)
    ]
    
    return ranked


def bench_parc_with_regimes(
    xc,
    model: Callable,
    parc: str,
    k_range: Tuple[float, float],
    D_range: Tuple[float, float],
    n_samples: int = 100,
    nwin: int = 100,
    key = None
) -> Dict[str, Any]:
    """Analyze performance across dynamical regimes (k, D parameter space).
    
    This function evaluates model performance across a grid of (k, D) parameter
    values to identify optimal dynamical regimes for a given parcellation.
    
    Args:
        xc: XCode object
        model: Dynamics model function
        parc: Parcellation name
        k_range: (k_min, k_max) range for coupling strength
        D_range: (D_min, D_max) range for noise intensity
        n_samples: Number of k and D values to test
        nwin: Simulation window size
        key: JAX random key (if None, uses default from vbjax)
    
    Returns:
        Dict with:
            'regime_grid': (nk, nD) grid of (k, D) values
            'performance_map': (nk, nD) performance metric (e.g., CCSD)
            'optimal_regime': (k_opt, D_opt) best performing regime
            'regime_diags': Diagnostics per regime
    
    From: ppt2-k-per-parc.ipynb (k vs ccsd optimization)
    """
    import jax, jax.numpy as jp, vbjax as vb
    
    if key is None:
        key = vb.key
    
    # Create parameter grids
    k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_samples)
    D_values = np.linspace(D_range[0], D_range[1], n_samples)
    
    # Get test connectome
    w = xc.get_conn(parc)[:1]  # Use first subject for evaluation
    
    performance_map = np.zeros((len(k_values), len(D_values)))
    
    # Evaluate each (k, D) regime
    for i, k in enumerate(tqdm.tqdm(k_values, desc="Evaluating k values")):
        for j, D in enumerate(D_values):
            # Run simulation with these parameters
            try:
                xf = model(w + jp.zeros((1, 1, 1)), k, D, use_pmap=False)
                
                # Compute performance metric (e.g., CCSD - correlation std deviation)
                # CCSD measures inter-subject variability in features
                # Higher CCSD means more discriminable features
                corr = np.corrcoef(xf[0].numpy() if hasattr(xf[0], 'numpy') else xf[0])
                np.fill_diagonal(corr, 0)
                performance_map[i, j] = np.std(corr)
            except Exception as e:
                performance_map[i, j] = 0
    
    # Find optimal regime
    opt_idx = np.unravel_index(np.argmax(performance_map), performance_map.shape)
    optimal_regime = (k_values[opt_idx[0]], D_values[opt_idx[1]])
    
    return {
        'regime_grid': (k_values, D_values),
        'performance_map': performance_map,
        'optimal_regime': optimal_regime,
        'max_performance': performance_map[opt_idx]
    }


def optimize_k_per_parcellation(
    xc,
    model: Callable,
    k_range: Tuple[float, float] = (0.01, 10.0),
    D: float = 0.2,
    nwin: int = 100,
    n_k: int = 128,
    key = None
) -> Dict[str, Any]:
    """Find optimal coupling parameter (k) for each parcellation.
    
    This function finds the optimal coupling strength (k) for each parcellation
    by maximizing the CCSD metric (correlation standard deviation), which measures
    inter-subject variability in features.
    
    Args:
        xc: XCode object
        model: Dynamics model function
        k_range: Range of k values to search (log space)
        D: Fixed noise parameter
        nwin: Simulation window size
        n_k: Number of k values to test
        key: JAX random key (if None, uses default from vbjax)
    
    Returns:
        Dict with:
            'optimal_k': Dict of parcellation -> optimal k
            'max_ccsd': Dict of parcellation -> max CCSD
            'traces': Dict of parcellation -> list of (k, ccsd)
    
    From: ppt2-k-per-parc.ipynb (optimizing k per parcellation)
    """
    import jax, jax.numpy as jp, vbjax as vb
    
    if key is None:
        key = vb.key
    
    # Get available parcellations (skip '031-MIST' as in notebooks)
    parcs = [p for p in xc.parcs if p != '031-MIST']
    
    optimal_k = {}
    max_ccsd = {}
    traces = {}
    
    for parc in tqdm.tqdm(parcs, desc="Optimizing k per parcellation"):
        # Create k values in log space
        k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_k)
        ccsd_values = []
        
        # Get test connectome
        w = xc.get_conn(parc)[:1]
        
        # Evaluate each k value
        for k in k_values:
            try:
                xf = model(w + jp.zeros((1, 1, 1)), k, D, use_pmap=False)
                
                # Compute CCSD
                corr = np.corrcoef(xf[0].numpy() if hasattr(xf[0], 'numpy') else xf[0])
                np.fill_diagonal(corr, 0)
                ccsd = np.std(corr)
                ccsd_values.append(ccsd)
            except Exception as e:
                ccsd_values.append(0)
        
        # Find optimal k
        max_idx = np.argmax(ccsd_values)
        optimal_k[parc] = k_values[max_idx]
        max_ccsd[parc] = ccsd_values[max_idx]
        traces[parc] = list(zip(k_values, ccsd_values))
    
    return {
        'optimal_k': optimal_k,
        'max_ccsd': max_ccsd,
        'traces': traces
    }


def bench_model_with_regime_validation(
    xc,
    model: Callable,
    parc: str = '079-Shen2013',
    arch: int = 8,
    auto_select_regime: bool = True,
    k_range: Tuple[float, float] = (0.01, 10.0),
    D_range: Tuple[float, float] = (0.001, 1.0),
    num_batch: int = 32,
    batch_size: int = 128,
    num_postcd: int = 1000,
    regime_metric: str = 'ccsd',
    use_pmap: bool = True,
    prog: bool = True
) -> Dict[str, Any]:
    """Run benchmark with automatic regime validation.
    
    This function validates the dynamical regime before running SBI benchmark.
    If the regime is invalid and auto_select_regime=True, it searches for
    optimal (k, D) parameters.
    
    Args:
        xc: XCode object with connectome data
        model: Dynamics model function
        parc: Parcellation name
        arch: Latent dimension for crosscoder
        auto_select_regime: If True, search for optimal regime when invalid
        k_range: (k_min, k_max) range for coupling strength search
        D_range: (D_min, D_max) range for noise intensity search
        num_batch: Number of batches for SBI training
        batch_size: Batch size per training batch
        num_postcd: Number of posterior samples for diagnostics
        regime_metric: Metric for regime optimization ('ccsd', 'regime_quality')
        use_pmap: Whether to use pmap for parallelization
        prog: Whether to show progress bars
    
    Returns:
        Dict with:
            'diags_cd': Cohort-level diagnostics (shrinkage, z_score, ci90)
            'diags_subj': Subject-level diagnostics (None in this version)
            'regime_validation': Initial regime validation results
            'regime_search': Regime search results (if auto_select_regime=True)
            'optimal_k': Coupling strength used
            'optimal_D': Noise intensity used
            'metrics': Summary metrics (ok_percent, median_shrinkage, etc.)
    
    From: Extension Point 5 (Crosscoder and Regime Diagnostics)
    """
    from .diagnostics import validate_regime_selection, search_optimal_regime, compute_subject_identification_accuracy
    from . import simulation
    import jax.numpy as jp
    import vbjax as vb
    
    w = xc.get_conn(parc)
    n_test = min(batch_size, len(w))
    
    default_k = 0.3
    default_D = 0.2
    
    k_test = default_k
    D_test = default_D
    
    initial_xf = model(w[:n_test], k_test, D_test, use_pmap=False)
    initial_xf_np = np.array(initial_xf)
    
    params_test = np.column_stack([
        np.full(n_test, k_test),
        np.full(n_test, D_test)
    ])
    
    regime_validation = validate_regime_selection(params_test, initial_xf_np, method='covariance')
    
    regime_search = None
    if not regime_validation['is_valid'] and auto_select_regime:
        regime_search = search_optimal_regime(
            xc, model, parc,
            k_range=k_range,
            D_range=D_range,
            n_samples=15,
            n_sims=n_test,
            metric=regime_metric
        )
        k_test = regime_search['optimal_k']
        D_test = regime_search['optimal_D']
    
    mvn = xc.calc_mvn(arch)
    theta_cdhat, xf_cdhat = simulation.sample_model(
        xc, model, mvn, parc, num_batch, batch_size,
        prog=prog, use_pmap=use_pmap
    )
    
    posterior_result = run_sbi(theta_cdhat, xf_cdhat, prog=prog)
    posterior = posterior_result[0] if isinstance(posterior_result, tuple) else posterior_result
    
    xf_test = model(w, k_test, D_test, use_pmap=False)
    xf_test_np = np.array(xf_test)
    
    theta_true = jp.concat([
        xc.encode_conn(arch, parc),
        jp.c_[jp.full((len(w),), k_test), jp.full((len(w),), D_test)]
    ], axis=1)
    
    po_theta = posterior.sample_batched(
        (num_postcd,), x=np.array(xf_test_np[:batch_size]),
        show_progress_bars=prog
    )
    
    diags_cd = posterior_diags(
        theta_cdhat[:, arch:],
        po_theta[..., arch:],
        np.array(theta_true[:batch_size, arch:])
    )
    
    shrinkage, z_score, ci90 = diags_cd
    ok_percent = compute_ok_metric(shrinkage, z_score)
    
    metrics = {
        'ok_percent': ok_percent,
        'median_shrinkage': np.median(shrinkage),
        'median_z': np.median(z_score),
        'mean_shrinkage': np.mean(shrinkage),
        'mean_z': np.mean(z_score),
        'ci90_coverage': np.mean(ci90),
        'k_used': k_test,
        'D_used': D_test,
    }
    
    return {
        'diags_cd': diags_cd,
        'diags_subj': None,
        'regime_validation': regime_validation,
        'regime_search': regime_search,
        'optimal_k': k_test,
        'optimal_D': D_test,
        'metrics': metrics,
        'parcellation': parc,
        'architecture': arch,
    }


def bench_cohort_model(
    xc,
    model: Callable,
    parc: str = '079-Shen2013',
    arch: int = 8,
    num_batch: int = 32,
    batch_size: int = 128,
    use_pmap=True
):
    """Benchmark cohort-level SBI (in-sample).

    Args:
        xc: XCode instance
        model: Dynamics model function
        parc: Parcellation name
        arch: Latent dimension
        num_batch: Number of batches for training
        batch_size: Batch size
        use_pmap: Whether to use pmap

    Returns:
        Tuple of (mean_shrinkage, mean_zscore)

    Moved from: simulation.py (Module 7 refactoring)
    """
    # Lazy import to avoid circular dependency
    from . import simulation

    mvn = xc.calc_mvn(arch)
    thetas, xfs = simulation.sample_model(
        xc, model, mvn, parc, num_batch, batch_size,
        use_pmap=use_pmap)
    posterior = run_sbi(thetas, xfs)
    thetas_hat = posterior.sample_batched((200,), x=np.array(xfs[:batch_size]))
    ps, pz, ci = posterior_diags(thetas_hat, thetas[:batch_size])
    return ps.mean().item(), pz.mean()


def bench_model(xc, model, parc='079-Shen2013',
                arch=8, num_batch=32, batch_size=128,
                do_subjets=False, num_postcd=None, inflate=1,
                use_pmap=True, return_everything=False,
                prog=True
                ):
    """Comprehensive benchmark comparing cohort vs subject-level SBI.

    Args:
        xc: XCode instance
        model: Dynamics model function
        parc: Parcellation name
        arch: Latent dimension
        num_batch: Number of batches for training
        batch_size: Batch size
        do_subjets: Whether to also run subject-level SBI
        num_postcd: Number of posterior samples (default: same as training)
        inflate: Inflation factor for covariance
        use_pmap: Whether to use pmap
        return_everything: If True, return locals() dict instead
        prog: Whether to show progress bars

    Returns:
        Tuple of (diags_cd, diags_subj) or locals() dict if return_everything=True
        - diags_cd: Cohort-level diagnostics
        - diags_subj: Subject-level diagnostics (or None if do_subjets=False)

    Moved from: simulation.py (Module 7 refactoring)
    """
    import jax, jax.numpy as jp, vbjax as vb

    # Import sample functions from simulation module
    from . import simulation

    # setup ground truth
    w = xc.get_conn(parc)  # (n_test, n_parc, n_parc)
    u = xc.encode_conn(arch, parc)
    k = 0.2 + vb.rand(len(w))*0.2
    D = 0.2 + vb.rand(len(w))*0.2
    theta = jp.concat([u, jp.c_[k, D]], axis=1)
    xf = model(w, k, D, use_pmap=False)  # (n_test, n_parc)

    # build & apply cohort sbi
    mvn = xc.calc_mvn(arch)
    mvn.u_cov = mvn.u_cov * inflate  # inflate a bit
    theta_cdhat, xf_cdhat = simulation.sample_model(xc, model, mvn, parc, num_batch, batch_size,
                                         prog=prog, use_pmap=use_pmap)
    posterior_cd = run_sbi(theta_cdhat, xf_cdhat, prog=prog)
    # Handle both old return (posterior) and new return (posterior, metadata)
    posterior = posterior_cd[0] if isinstance(posterior_cd, tuple) else posterior_cd
    po_theta_cdhat = posterior.sample_batched(
        (num_postcd or theta_cdhat.shape[0],), x=np.array(xf), show_progress_bars=prog)
    # (nsamp, xf.shape[0], arch+2)
    diags_cd = posterior_diags(
        theta_cdhat[:, arch:], po_theta_cdhat[..., arch:], theta[:, arch:])

    if do_subjets:
        # sbi subject
        diags_subj = []
        for it in tqdm.trange(len(xf)):  # 74 subjects
            theta_hat, xf_hat = simulation.sample_subj_model(
                w[it], model, num_batch, batch_size, prog=prog, use_pmap=use_pmap)
            posterior_result = run_sbi(theta_hat, xf_hat, prog=prog)
            # Handle both old return (posterior) and new return (posterior, metadata)
            posterior = posterior_result[0] if isinstance(posterior_result, tuple) else posterior_result
            po_theta_hat = posterior.sample(
                (theta_hat.shape[0],), x=np.array(xf[it]), show_progress_bars=prog)
            diags_it = posterior_diags(
                theta_hat, po_theta_hat, theta[it, arch:])
            diags_subj.append(diags_it)
    else:
        diags_subj = None

    if return_everything:
        return locals()
    return diags_cd, diags_subj
