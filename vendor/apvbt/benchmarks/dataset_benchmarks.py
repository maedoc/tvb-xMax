"""Dataset loader performance benchmarks.

This module provides performance benchmarks for dataset loaders, measuring:
- Loading time for different dataset sizes
- Memory usage during loading
- Validation performance
- Metadata extraction speed
"""

import time
import tempfile
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from apvbt.benchmarks.core import Benchmark, BenchmarkResult, timeit, measure_memory
from apvbt.datasets import (
    DatasetConfig,
    DatasetRegistry,
    DatasetMetadata,
    ValidationResult,
    load_dataset,
    validate_config,
    get_dataset_metadata,
)
from apvbt.data import XCode


def _create_test_xcode(n_subjects: int = 10, n_parcels: int = 3, n_regions: int = 50) -> XCode:
    """Create a test XCode object for benchmarking.
    
    Similar to test fixture but optimized for benchmarks.
    """
    import jax.numpy as jp
    
    xcode = XCode()
    
    n_triu = n_regions * (n_regions - 1) // 2
    
    conns = []
    means = []
    parcs = []
    
    for i in range(n_parcels):
        parc_name = f"{n_regions:03d}-TestParc{i}"
        parcs.append(parc_name)
        
        # Create random connectivity matrix for this parcellation
        conn = jp.array(np.random.randn(n_subjects, n_triu).astype(np.float32))
        conns.append(conn)
        
        # Compute mean across subjects
        mean = conn.mean(axis=0)
        means.append(mean)
    
    xcode.conns = conns
    xcode.means = means
    xcode.parcs = parcs
    xcode.tts = n_subjects // 2
    xcode.wbs = []
    
    return xcode


def benchmark_local_file_loader(bench: Benchmark, 
                               n_subjects: int = 100,
                               n_parcels: int = 5,
                               n_regions: int = 100) -> Dict[str, Any]:
    """Benchmark local file loader performance.
    
    Measures:
    - Time to serialize XCode to pickle
    - Time to load from pickle
    - Memory usage during loading
    - Validation speed
    """
    from apvbt.datasets.local_file import LocalFileLoader
    
    # Create test data
    xcode = _create_test_xcode(n_subjects=n_subjects, 
                               n_parcels=n_parcels, 
                               n_regions=n_regions)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Benchmark serialization
        @bench.timeit(repeats=3, warmup=1)
        def serialize_xcode():
            with open(temp_path, 'wb') as f:
                pickle.dump(xcode, f)
        
        serialize_xcode()
        
        # Benchmark loading
        loader = LocalFileLoader()
        config = DatasetConfig(
            source_type='local-file',
            source_url=temp_path,
            format='pkl'
        )
        
        @bench.timeit(repeats=3, warmup=1)
        def load_xcode():
            return loader.load(config)
        
        result = load_xcode()
        
        # Benchmark validation
        @bench.timeit(repeats=5, warmup=1)
        def validate_config():
            return loader.validate(config)
        
        validation_result = validate_config()
        
        # Benchmark metadata extraction
        @bench.timeit(repeats=5, warmup=1)
        def get_metadata():
            return loader.get_metadata(config)
        
        metadata = get_metadata()
        
        return {
            'n_subjects': n_subjects,
            'n_parcels': n_parcels,
            'n_regions': n_regions,
            'validation_success': validation_result.is_valid,
            'metadata_subjects': metadata.n_subjects
        }
        
    finally:
        # Clean up
        Path(temp_path).unlink(missing_ok=True)


def benchmark_loader_registry(bench: Benchmark) -> Dict[str, Any]:
    """Benchmark dataset loader registry operations.
    
    Measures:
    - Time to get loader from registry
    - Time to list available loaders
    - Registration performance
    """
    from apvbt.datasets import DatasetRegistry
    
    # Benchmark getting loader
    @bench.timeit(repeats=10, warmup=2)
    def get_loader():
        return DatasetRegistry.get('local-file')
    
    loader_class = get_loader()
    
    # Benchmark listing loaders
    @bench.timeit(repeats=10, warmup=2)
    def list_loaders():
        return DatasetRegistry.list_loaders()
    
    loader_list = list_loaders()
    
    # Benchmark creating loader instance
    @bench.timeit(repeats=10, warmup=2)
    def create_loader():
        cls = DatasetRegistry.get('local-file')
        return cls()
    
    loader = create_loader()
    
    return {
        'available_loaders': len(loader_list),
        'has_local_file': 'local-file' in loader_list,
        'loader_class': loader_class.__name__
    }


def benchmark_concurrent_loading(bench: Benchmark,
                                 n_concurrent: int = 3,
                                 small_size: Tuple[int, int, int] = (10, 2, 50),
                                 large_size: Tuple[int, int, int] = (100, 5, 100)) -> Dict[str, Any]:
    """Benchmark concurrent loading scenarios.
    
    Measures:
    - Time to load multiple small datasets
    - Time to load large dataset
    - Memory usage patterns
    """
    from apvbt.datasets.local_file import LocalFileLoader
    
    # Create temporary files
    temp_files = []
    try:
        # Create small datasets
        small_configs = []
        for i in range(n_concurrent):
            xcode = _create_test_xcode(n_subjects=small_size[0],
                                       n_parcels=small_size[1],
                                       n_regions=small_size[2])
            with tempfile.NamedTemporaryFile(suffix=f'_{i}.pkl', delete=False) as f:
                temp_path = f.name
                pickle.dump(xcode, f)
                temp_files.append(temp_path)
                
                config = DatasetConfig(
                    source_type='local-file',
                    source_url=temp_path,
                    format='pkl'
                )
                small_configs.append(config)
        
        # Create large dataset
        xcode_large = _create_test_xcode(n_subjects=large_size[0],
                                         n_parcels=large_size[1],
                                         n_regions=large_size[2])
        with tempfile.NamedTemporaryFile(suffix='_large.pkl', delete=False) as f:
            large_path = f.name
            pickle.dump(xcode_large, f)
            temp_files.append(large_path)
            large_config = DatasetConfig(
                source_type='local-file',
                source_url=large_path,
                format='pkl'
            )
        
        loader = LocalFileLoader()
        
        # Benchmark loading small datasets sequentially
        @bench.timeit(repeats=3, warmup=1)
        def load_small_sequential():
            results = []
            for config in small_configs:
                results.append(loader.load(config))
            return results
        
        small_results = load_small_sequential()
        
        # Benchmark loading large dataset
        @bench.timeit(repeats=3, warmup=1)
        def load_large():
            return loader.load(large_config)
        
        large_result = load_large()
        
        return {
            'n_concurrent': n_concurrent,
            'small_size': small_size,
            'large_size': large_size,
            'small_loaded': len(small_results),
            'large_subjects': large_result.conns[0].shape[0] if large_result.conns else 0
        }
        
    finally:
        # Clean up
        for path in temp_files:
            Path(path).unlink(missing_ok=True)


def run_dataset_benchmarks(benchmark_name: str = 'dataset-loaders',
                          n_subjects: int = 100,
                          n_parcels: int = 5,
                          n_regions: int = 100) -> Benchmark:
    """Run comprehensive dataset loader benchmarks.
    
    Args:
        benchmark_name: Name for this benchmark run
        n_subjects: Number of subjects in test datasets
        n_parcels: Number of parcellations in test datasets
        n_regions: Number of regions per parcellation
        
    Returns:
        Benchmark object with results
    """
    bench = Benchmark(benchmark_name)
    
    # Run individual benchmarks
    local_file_results = benchmark_local_file_loader(
        bench, n_subjects, n_parcels, n_regions
    )
    
    registry_results = benchmark_loader_registry(bench)
    
    concurrent_results = benchmark_concurrent_loading(bench)
    
    # Add metadata about benchmarks
    bench.results.append(BenchmarkResult(
        name='benchmark_metadata',
        elapsed=0.0,
        metadata={
            'local_file_results': local_file_results,
            'registry_results': registry_results,
            'concurrent_results': concurrent_results,
            'parameters': {
                'n_subjects': n_subjects,
                'n_parcels': n_parcels,
                'n_regions': n_regions
            }
        }
    ))
    
    return bench