"""Result persistence and reporting for benchmarking workflows.

This module provides functionality for saving, loading, and generating reports
from benchmark results to ensure reproducibility and enable result querying.

Functions implement result persistence workflows from notebooks (step2-id-conn-parc,
ppt1-bench-parcs) to ensure reproducibility of research results.
"""

import numpy as np
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import warnings


def save_benchmark_results(
    results: Dict[str, Any],
    fname: str,
    format: str = 'npz',
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Save benchmark results to file with metadata.

    This function saves benchmark results (from bench_model_multi_parc,
    bench_model_multi_algo, etc.) to a file with standardized format
    and optional metadata for provenance tracking.

    Args:
        results: Benchmark results dictionary
        fname: Output filename (path)
        format: File format ('npz', 'json', 'pkl')
        metadata: Optional metadata (timestamp, config, etc.)

    Returns:
        Dict with:
            'saved_path': Full path to saved file
            'size': File size in bytes
            'checksum': SHA256 checksum
            'metadata': Metadata included in file

    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written

    From: step2-id-conn-parc.ipynb (np.savez for hopf_bench_parc.npz)
    """
    fname_path = Path(fname)
    fname_path.parent.mkdir(parents=True, exist_ok=True)

    # Add default metadata if not provided
    if metadata is None:
        metadata = {}
    if 'timestamp' not in metadata:
        metadata['timestamp'] = datetime.now().isoformat()
    if 'version' not in metadata:
        metadata['version'] = '1.0'

    # Combine results with metadata
    data = {
        'results': results,
        'metadata': metadata
    }

    if format == 'npz':
        # Save as NPZ (numpy compressed format)
        # Convert numpy arrays to saveable format
        save_dict = {}
        save_dict['_metadata'] = json.dumps(metadata)

        # Flatten results dictionary for npz saving
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f'{prefix}{key}' if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, f'{new_key}_')
                elif isinstance(value, (np.ndarray, int, float, str)):
                    save_dict[new_key] = value
                elif isinstance(value, tuple):
                    for i, item in enumerate(value):
                        if isinstance(item, np.ndarray):
                            save_dict[f'{new_key}_item_{i}'] = item
                        else:
                            save_dict[f'{new_key}_item_{i}'] = item

        flatten_dict(results)
        np.savez(fname_path, **save_dict)

    elif format == 'json':
        # Save as JSON (human-readable, but limited numpy support)
        # Convert numpy arrays to lists
        json_dict = _convert_to_json_serializable(data)
        with open(fname_path, 'w') as f:
            json.dump(json_dict, f, indent=2)

    elif format == 'pkl':
        # Save as pickle (supports all Python objects)
        with open(fname_path, 'wb') as f:
            pickle.dump(data, f)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported: 'npz', 'json', 'pkl'")

    # Compute file info
    file_size = fname_path.stat().st_size
    checksum = _compute_checksum(fname_path)

    return {
        'saved_path': str(fname_path.absolute()),
        'size': file_size,
        'checksum': checksum,
        'metadata': metadata
    }


def load_benchmark_results(
    fname: str,
    format: Optional[str] = None
) -> Dict[str, Any]:
    """Load benchmark results from file with provenance.

    This function loads benchmark results saved by save_benchmark_results
    and returns both the results and provenance metadata.

    Args:
        fname: Input filename (path)
        format: File format (auto-detect if None)

    Returns:
        Dict with:
            'results': Loaded benchmark results
            'metadata': Provenance metadata
            'provenance': File info (path, size, checksum)

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format cannot be determined or is unsupported

    From: None (new capability for standardized loading)
    """
    fname_path = Path(fname)

    if not fname_path.exists():
        raise FileNotFoundError(f"File not found: {fname}")

    # Auto-detect format if not specified
    if format is None:
        suffix = fname_path.suffix.lower()
        if suffix == '.npz':
            format = 'npz'
        elif suffix == '.json':
            format = 'json'
        elif suffix == '.pkl':
            format = 'pkl'
        else:
            raise ValueError(f"Cannot auto-detect format from suffix: {suffix}")

    # Compute file info for provenance
    file_size = fname_path.stat().st_size
    checksum = _compute_checksum(fname_path)

    # Load based on format
    if format == 'npz':
        # Load from NPZ
        data = np.load(fname_path, allow_pickle=True)
        metadata = json.loads(str(data['_metadata']))

        # Reconstruct results dictionary from flattened format
        results = _reconstruct_results_from_npz(dict(data))

    elif format == 'json':
        # Load from JSON
        with open(fname_path, 'r') as f:
            data = json.load(f)
        results = data['results']
        metadata = data.get('metadata', {})

    elif format == 'pkl':
        # Load from pickle
        with open(fname_path, 'rb') as f:
            data = pickle.load(f)
        results = data['results']
        metadata = data.get('metadata', {})

    else:
        raise ValueError(f"Unsupported format: {format}")

    return {
        'results': results,
        'metadata': metadata,
        'provenance': {
            'path': str(fname_path.absolute()),
            'size': file_size,
            'checksum': checksum
        }
    }


def generate_benchmark_summary(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['shrinkage', 'z_score', 'ok_percent'],
    format: str = 'markdown'
) -> str:
    """Generate summary table of benchmark results.

    This function generates a formatted summary table from multi-parcellation
    benchmark results, suitable for publication or documentation.

    Args:
        results_dict: Results from bench_model_multi_parc
            Format: {'parc_name': {'metrics': {'shrinkage': ..., 'z_score': ..., ...}}, ...}
        metrics: List of metrics to include in summary
        format: Output format ('markdown', 'csv', 'html', 'latex')

    Returns:
        Formatted summary table as string

    Raises:
        ValueError: If format is not supported
        KeyError: If required metrics are missing from results

    From: ppt1-bench-parcs.ipynb (manual table formatting)
    """
    # Extract parcellation names
    parcs = list(results_dict.keys())

    if not parcs:
        warnings.warn("Empty results dictionary provided")
        return ""

    # Build table header
    header = ['Parcellation'] + [f'{m.title()}' for m in metrics]

    # Build table rows
    rows = []
    for parc in parcs:
        row = [parc]
        parc_metrics = results_dict[parc].get('metrics', {})

        for metric in metrics:
            # Map metric names to possible keys in results
            if metric == 'shrinkage':
                value = parc_metrics.get('mean_shrinkage', parc_metrics.get('median_shrinkage', np.nan))
            elif metric == 'z_score':
                value = parc_metrics.get('mean_z', parc_metrics.get('median_z', np.nan))
            elif metric == 'ok_percent':
                value = parc_metrics.get('ok_percent', np.nan)
            else:
                value = parc_metrics.get(metric, np.nan)

            # Format value
            if isinstance(value, (float, np.floating)):
                row.append(f'{value:.4f}')
            else:
                row.append(str(value))

        rows.append(row)

    # Format output based on requested format
    if format == 'markdown':
        return _format_table_markdown(header, rows)
    elif format == 'csv':
        return _format_table_csv(header, rows)
    elif format == 'html':
        return _format_table_html(header, rows)
    elif format == 'latex':
        return _format_table_latex(header, rows)
    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_comparison_report(
    results1: Dict[str, Any],
    results2: Dict[str, Any],
    label1: str = 'MAF',
    label2: str = 'MDN',
    tests: List[str] = ['wilcoxon']
) -> Dict[str, Any]:
    """Generate comparison report between two sets of results.

    This function generates a comprehensive comparison report between
    two benchmark result sets (e.g., MAF vs MDN algorithms) including
    statistical tests and recommendations.

    Args:
        results1: First set of results (e.g., from MAF)
        results2: Second set of results (e.g., from MDN)
        label1: Label for first results (e.g., 'MAF')
        label2: Label for second results (e.g., 'MDN')
        tests: Statistical tests to run

    Returns:
        Dict with:
            'summary_text': Textual summary
            'significance_tests': Statistical test results
            'recommendations': Algorithm recommendations
            'metrics_comparison': Side-by-side metrics comparison

    From: ppt1-bench-parcs.ipynb (manual comparison of MAF vs MDN)
    """
    from .comparison import paired_difference_test, compute_ok_metric

    report = {
        'labels': (label1, label2),
        'summary_text': '',
        'significance_tests': {},
        'recommendations': '',
        'metrics_comparison': {}
    }

    # Extract metrics from results
    def extract_metrics(results):
        if 'metrics' in results:
            return results['metrics']
        # Handle diags_cd format
        elif 'diags_cd' in results:
            shrinkage, z_score, ci90 = results['diags_cd']
            ok = compute_ok_metric(shrinkage, z_score)
            return {
                'mean_shrinkage': np.mean(shrinkage),
                'median_shrinkage': np.median(shrinkage),
                'mean_z': np.mean(z_score),
                'median_z': np.median(z_score),
                'ok_percent': ok,
                'ci90_coverage': np.mean(ci90)
            }
        else:
            return {}

    metrics1 = extract_metrics(results1)
    metrics2 = extract_metrics(results2)

    # Build metrics comparison table
    for key in set(list(metrics1.keys()) + list(metrics2.keys())):
        report['metrics_comparison'][key] = {
            label1: metrics1.get(key, np.nan),
            label2: metrics2.get(key, np.nan)
        }

    # Run statistical tests if diags available
    if 'diags_cd' in results1 and 'diags_cd' in results2:
        shrinkage1, z_score1, _ = results1['diags_cd']
        shrinkage2, z_score2, _ = results2['diags_cd']

        for test_name in tests:
            # Test shrinkage difference
            result_shrink = paired_difference_test(shrinkage1, shrinkage2, test=test_name)
            report['significance_tests'][f'shrinkage_{test_name}'] = result_shrink

            # Test z-score difference
            result_z = paired_difference_test(z_score1, z_score2, test=test_name)
            report['significance_tests'][f'z_score_{test_name}'] = result_z

        # Generate recommendations based on tests
        report['recommendations'] = _generate_recommendations(
            metrics1, metrics2, report['significance_tests']
        )

    # Generate summary text
    report['summary_text'] = _generate_comparison_summary(
        label1, label2, metrics1, metrics2, report['significance_tests']
    )

    return report


def query_benchmark_results(
    results_dir: str,
    filters: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Query saved benchmark results by metadata.

    This function searches a directory for benchmark result files and
    filters them based on metadata criteria.

    Args:
        results_dir: Directory containing benchmark result files
        filters: Dict of metadata filters (e.g., {'model': 'hopf', 'algo': 'maf'})

    Returns:
        List of matching benchmark result dicts with file info

    From: None (new capability for result querying)
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Supported formats
    suffixes = ['.npz', '.json', '.pkl']
    result_files = []

    # Find all result files
    for suffix in suffixes:
        result_files.extend(results_path.glob(f'*{suffix}'))

    matching_results = []

    for result_file in result_files:
        try:
            # Load file to check metadata
            data = load_benchmark_results(str(result_file))

            # Check if metadata matches filters
            if _matches_filters(data['metadata'], filters):
                matching_results.append({
                    'results': data['results'],
                    'metadata': data['metadata'],
                    'provenance': data['provenance']
                })

        except Exception as e:
            # Skip files that cannot be loaded
            warnings.warn(f"Could not load {result_file}: {e}")
            continue

    return matching_results


# Helper functions

def _compute_checksum(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _convert_to_json_serializable(data: Any) -> Any:
    """Convert data to JSON-serializable format."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: _convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        return float(data)
    elif isinstance(data, (bool, str, int, float, type(None))):
        return data
    else:
        # Try to convert to string for unknown types
        return str(data)


def _reconstruct_results_from_npz(data: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct results dictionary from flattened NPZ format."""
    results = {}

    # Extract metadata (already handled separately)
    data = {k: v for k, v in data.items() if k != '_metadata'}

    # Reconstruct hierarchical structure
    # Keys are in format: 'parc_name_metrics_mean_shrinkage' etc.
    # We need to parse these into nested dictionaries

    # Find all top-level keys (parcellation names)
    top_level_keys = set()
    for key in data.keys():
        if '_' in key:
            parts = key.split('_')
            if parts:
                top_level_keys.add(parts[0])

    # Reconstruct each top-level entry
    for tl_key in top_level_keys:
        # Find all keys starting with this top-level key
        tl_keys = [k for k in data.keys() if k.startswith(tl_key + '_')]

        if not tl_keys:
            continue

        # Build nested structure
        results[tl_key] = {}

        # This is a simplified reconstruction - full reconstruction would need
        # to handle the exact key structure from flatten_dict
        for key in tl_keys:
            suffix = key[len(tl_key) + 1:]
            value = data[key]

            # Handle item_ format (for tuples)
            if 'item_' in suffix:
                # Skip item_ keys - handled differently in full implementation
                continue

            # Assign to nested dict
            parts = suffix.split('_')

            current = results[tl_key]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

    return results


def _format_table_markdown(header: List[str], rows: List[List[str]]) -> str:
    """Format table as Markdown."""
    lines = []
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '|'.join(['---' for _ in header]) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _format_table_csv(header: List[str], rows: List[List[str]]) -> str:
    """Format table as CSV."""
    lines = []
    lines.append(','.join(header))
    for row in rows:
        lines.append(','.join(row))
    return '\n'.join(lines)


def _format_table_html(header: List[str], rows: List[List[str]]) -> str:
    """Format table as HTML."""
    lines = ['<table>']
    lines.append('<thead><tr>' + ''.join(f'<th>{h}</th>' for h in header) + '</tr></thead>')
    lines.append('<tbody>')
    for row in rows:
        lines.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>')
    lines.append('</tbody></table>')
    return '\n'.join(lines)


def _format_table_latex(header: List[str], rows: List[List[str]]) -> str:
    """Format table as LaTeX."""
    n_cols = len(header)
    lines = [r'\begin{tabular}{' + 'l' * n_cols + '}']
    lines.append(' & '.join(header) + r' \\')
    lines.append(r'\midrule')
    for row in rows:
        lines.append(' & '.join(row) + r' \\')
    lines.append(r'\end{tabular}')
    return '\n'.join(lines)


def _generate_recommendations(
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    tests: Any
) -> str:
    """Generate algorithm recommendations based on metrics and tests."""
    lines = []

    # Compare OK% (primary metric)
    ok1 = metrics1.get('ok_percent', 0)
    ok2 = metrics2.get('ok_percent', 0)

    # Get labels from tests dict or use defaults
    labels = tests.get('labels')
    if isinstance(labels, tuple) and len(labels) >= 2:
        label1, label2 = labels[0], labels[1]
    elif isinstance(labels, list) and len(labels) >= 2:
        label1, label2 = labels[0], labels[1]
    else:
        label1, label2 = 'Algorithm 1', 'Algorithm 2'

    if ok1 > ok2:
        lines.append(f"- {label1} has higher OK% ({ok1:.1f}% vs {ok2:.1f}%)")
    elif ok2 > ok1:
        lines.append(f"- {label2} has higher OK% ({ok2:.1f}% vs {ok1:.1f}%)")
    else:
        lines.append(f"- Both algorithms have identical OK% ({ok1:.1f}%)")

    # Check statistical significance
    shrink_test = tests.get('shrinkage_wilcoxon')
    if shrink_test and shrink_test.get('is_significant', False):
        lines.append(f"- Shrinkage difference is statistically significant (p={shrink_test['p_value']:.4f})")

    z_test = tests.get('z_score_wilcoxon')
    if z_test and z_test.get('is_significant', False):
        lines.append(f"- Z-score difference is statistically significant (p={z_test['p_value']:.4f})")

    return '\n'.join(lines) if lines else "Unable to generate recommendations."


def _generate_comparison_summary(
    label1: str, label2: str,
    metrics1: Dict[str, float],
    metrics2: Dict[str, float],
    tests: Dict[str, Dict[str, Any]]
) -> str:
    """Generate textual comparison summary."""
    lines = [f"Comparison: {label1} vs {label2}"]
    lines.append("=" * 50)
    lines.append("")

    # Metrics comparison
    lines.append("Metrics:")
    for key in sorted(set(list(metrics1.keys()) + list(metrics2.keys()))):
        val1 = metrics1.get(key, np.nan)
        val2 = metrics2.get(key, np.nan)
        lines.append(f"  {key}:")
        lines.append(f"    {label1}: {val1:.4f}")
        lines.append(f"    {label2}: {val2:.4f}")
    lines.append("")

    # Statistical tests
    if tests:
        lines.append("Statistical Tests:")
        for test_name, test_result in tests.items():
            if 'is_significant' in test_result:
                sig = "significant" if test_result['is_significant'] else "not significant"
                pval = test_result.get('p_value', np.nan)
                lines.append(f"  {test_name}: {sig} (p={pval:.4f})")
    lines.append("")

    return '\n'.join(lines)


def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if metadata matches all filters."""
    for key, value in filters.items():
        if key not in metadata:
            return False
        if metadata[key] != value:
            return False
    return True
