"""Performance benchmarking utilities for APVBT.

This module provides tools for measuring runtime and memory usage of key
APVBT operations, enabling performance regression detection and optimization.
"""

from .core import Benchmark, timeit, measure_memory, get_default_benchmark
from .dataset_benchmarks import (
    benchmark_local_file_loader,
    benchmark_loader_registry,
    benchmark_concurrent_loading,
    run_dataset_benchmarks,
)
from .model_benchmarks import (
    benchmark_model_initialization,
    benchmark_simulation_speed,
    benchmark_model_registry,
    benchmark_parameter_sampling,
    benchmark_cross_model_comparison,
    run_model_benchmarks,
)

__all__ = [
    'Benchmark',
    'timeit',
    'measure_memory',
    'get_default_benchmark',
    # Dataset benchmarks
    'benchmark_local_file_loader',
    'benchmark_loader_registry',
    'benchmark_concurrent_loading',
    'run_dataset_benchmarks',
    # Model benchmarks
    'benchmark_model_initialization',
    'benchmark_simulation_speed',
    'benchmark_model_registry',
    'benchmark_parameter_sampling',
    'benchmark_cross_model_comparison',
    'run_model_benchmarks',
]