"""Dynamics model performance benchmarks.

This module provides performance benchmarks for brain dynamics models, measuring:
- Simulation speed for different model types
- Memory usage during simulation
- Parameter space sampling performance
- Model initialization time
"""

import time
import numpy as np
import jax
import jax.numpy as jp
from typing import Dict, Any, List, Optional, Tuple

from apvbt.benchmarks.core import Benchmark, BenchmarkResult, timeit, measure_memory
from apvbt.dynamics.models import (
    ModelRegistry,
    get_model,
    list_models,
    ParameterSpace,
    SimulationConfig,
    ValidationResult,
    ParameterDefinition,
    sample_parameter_space,
    validate_parameter_space,
)
from apvbt.dynamics.models.hopf import HopfModel
from apvbt.dynamics.models.mpr import MPRModel
from apvbt.dynamics.models.wilson_cowan import WilsonCowanModel
from apvbt.dynamics.models.wong_wang import WongWangModel
from apvbt.dynamics.models.kuramoto import KuramotoModel
from apvbt.dynamics.models.fitzhugh_nagumo import FitzHughNagumoModel


def get_default_parameters(param_space: ParameterSpace) -> Dict[str, Any]:
    """Get default parameter values from parameter space.
    
    Uses param_def.default if set, otherwise uses midpoint of bounds.
    """
    defaults = {}
    for name, param_def in param_space.parameters.items():
        if param_def.default is not None:
            defaults[name] = param_def.default
        else:
            # Use midpoint of bounds
            midpoint = (param_def.bounds[0] + param_def.bounds[1]) / 2
            defaults[name] = midpoint
    return defaults


def create_test_connectivity(n_regions: int = 50, seed: int = 42) -> jp.ndarray:
    """Create test connectivity matrix for benchmarking."""
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    
    # Create random symmetric connectivity matrix
    w = np.random.randn(n_regions, n_regions).astype(np.float32)
    w = (w + w.T) / 2  # Symmetrize
    w = w - np.diag(np.diag(w))  # Zero diagonal
    w = np.abs(w)  # Ensure non-negative
    
    # Normalize
    w = w / w.max()
    
    return jp.array(w)


def benchmark_model_initialization(bench: Benchmark, 
                                  model_name: str,
                                  n_regions: int = 50) -> Dict[str, Any]:
    """Benchmark model initialization performance.
    
    Measures:
    - Time to create model instance
    - Time to validate default parameter space
    - Memory usage during initialization
    """
    # Benchmark getting model from registry
    @bench.timeit(repeats=5, warmup=1)
    def get_model_from_registry():
        return get_model(model_name)
    
    model_class = get_model_from_registry()
    
    # Benchmark creating model instance
    @bench.timeit(repeats=5, warmup=1)
    def create_model_instance():
        return model_class()
    
    model = create_model_instance()
    
    # Benchmark getting default parameter space
    @bench.timeit(repeats=5, warmup=1)
    def get_parameter_space():
        return model.get_parameter_space()
    
    param_space = get_parameter_space()
    
    # Benchmark validating parameter space
    @bench.timeit(repeats=5, warmup=1)
    def validate_parameters():
        return validate_parameter_space(param_space)
    
    validation_result = validate_parameters()
    
    return {
        'model_name': model_name,
        'model_class': model_class.__name__,
        'n_parameters': len(param_space.parameters),
        'validation_success': validation_result.is_valid,
        'n_regions': n_regions
    }


def benchmark_simulation_speed(bench: Benchmark,
                              model_name: str,
                              n_regions: int = 50,
                              simulation_steps: int = 1000,
                              n_simulations: int = 10) -> Dict[str, Any]:
    """Benchmark simulation performance.
    
    Measures:
    - Time to run single simulation
    - Time to run multiple simulations (batched)
    - Memory usage during simulation
    """
    # Get model
    model_class = get_model(model_name)
    model = model_class()
    
    # Create test connectivity
    w = create_test_connectivity(n_regions=n_regions)
    
    # Get parameter space and default parameters
    param_space = model.get_parameter_space()
    default_params = get_default_parameters(param_space)
    
    # Create simulation config
    config = SimulationConfig(
        dt=0.001,
        simulation_duration=simulation_steps * 0.001,
        transient_duration=100 * 0.001,  # burn_in steps
        num_windows=1,
        use_pmap=False,
        seed=42
    )
    
    # Benchmark single simulation
    @bench.timeit(repeats=3, warmup=1)
    def run_single_simulation():
        return model.simulate(coupling_matrix=w, parameters=default_params, config=config)
    
    result = run_single_simulation()
    
    # Benchmark batched simulations
    @bench.timeit(repeats=3, warmup=1)
    def run_batched_simulations():
        # Create multiple parameter sets using sample_parameter_space
        param_samples_dict = sample_parameter_space(
            param_space, 
            n_samples=n_simulations,
            node_count=n_regions,
            rng_key=jax.random.PRNGKey(42)
        )
        # Convert dict of arrays to list of dicts
        param_samples = []
        for i in range(n_simulations):
            params = {}
            for param_name, param_array in param_samples_dict.items():
                if param_array.ndim == 2:  # heterogeneous, shape (n_samples, n_regions)
                    params[param_name] = param_array[i]
                else:  # scalar, shape (n_samples,)
                    params[param_name] = param_array[i]
            param_samples.append(params)
        
        # Run simulations sequentially (in real usage would be batched)
        results = []
        for params in param_samples:
            config = SimulationConfig(
                dt=0.001,
                simulation_duration=simulation_steps * 0.001,
                transient_duration=100 * 0.001,
                num_windows=1,
                use_pmap=False,
                seed=42 + len(results)
            )
            results.append(model.simulate(coupling_matrix=w, parameters=params, config=config))
        
        return results
    
    batched_results = run_batched_simulations()
    
    # Measure memory usage for single simulation
    def memory_intensive_simulation():
        config = SimulationConfig(
            dt=0.001,
            simulation_duration=simulation_steps * 10 * 0.001,  # Longer simulation
            transient_duration=100 * 0.001,
            num_windows=1,
            use_pmap=False,
            seed=99
        )
        return model.simulate(coupling_matrix=w, parameters=default_params, config=config)
    
    memory_result = bench.measure_memory(memory_intensive_simulation)
    
    return {
        'model_name': model_name,
        'n_regions': n_regions,
        'simulation_steps': simulation_steps,
        'n_simulations': n_simulations,
        'single_sim_time_ms': bench._timings.get('run_single_simulation', [0])[0] * 1000,
        'batched_sim_time_ms': bench._timings.get('run_batched_simulations', [0])[0] * 1000,
        'simulation_success': result is not None,
        'n_batched_results': len(batched_results)
    }


def benchmark_model_registry(bench: Benchmark) -> Dict[str, Any]:
    """Benchmark model registry operations.
    
    Measures:
    - Time to list all models
    - Time to get each model
    - Registration performance
    """
    # Benchmark listing models
    @bench.timeit(repeats=10, warmup=2)
    def list_all_models():
        return list_models()
    
    model_list = list_all_models()
    
    # Benchmark getting each model
    model_times = {}
    for model_name in model_list:
        @bench.timeit(repeats=5, warmup=1)
        def get_specific_model(name=model_name):
            return get_model(name)
        
        model_class = get_specific_model()
        model_times[model_name] = bench._timings.get('get_specific_model', [0])[-1]
    
    # Benchmark creating instances
    instance_times = {}
    for model_name in model_list:
        model_class = get_model(model_name)
        
        @bench.timeit(repeats=5, warmup=1)
        def create_instance(cls=model_class):
            return cls()
        
        instance = create_instance()
        instance_times[model_name] = bench._timings.get('create_instance', [0])[-1]
    
    return {
        'n_models': len(model_list),
        'models': model_list,
        'model_get_times': {k: v * 1000 for k, v in model_times.items()},
        'instance_create_times': {k: v * 1000 for k, v in instance_times.items()}
    }


def benchmark_parameter_sampling(bench: Benchmark,
                                model_name: str,
                                n_samples: int = 100) -> Dict[str, Any]:
    """Benchmark parameter space sampling performance.
    
    Measures:
    - Time to sample parameters from space
    - Memory usage during sampling
    - Parameter validation speed
    """
    # Get model
    model_class = get_model(model_name)
    model = model_class()
    
    # Get parameter space
    param_space = model.get_parameter_space()
    
    # Benchmark sampling parameters
    @bench.timeit(repeats=5, warmup=1)
    def sample_parameters():
        samples = []
        for _ in range(n_samples):
            sample_dict = sample_parameter_space(param_space, n_samples=1, node_count=1)
            sample = {k: v[0] for k, v in sample_dict.items()}
            samples.append(sample)
        return samples
    
    parameter_samples = sample_parameters()
    
    # Benchmark validating samples
    @bench.timeit(repeats=5, warmup=1)
    def validate_samples():
        results = []
        for params in parameter_samples:
            results.append(model.validate_parameters(params))
        return results
    
    validation_results = validate_samples()
    
    # Measure memory usage for large sampling
    def memory_intensive_sampling():
        # Generate all samples at once
        sample_dict = sample_parameter_space(param_space, n_samples=n_samples * 10, node_count=1)
        # Convert dict of arrays to list of dicts
        large_samples = []
        for i in range(n_samples * 10):
            sample = {k: v[i] for k, v in sample_dict.items()}
            large_samples.append(sample)
        return large_samples
    
    large_samples = bench.measure_memory(memory_intensive_sampling)
    
    valid_count = sum(1 for r in validation_results if r.is_valid)
    
    return {
        'model_name': model_name,
        'n_samples': n_samples,
        'valid_samples': valid_count,
        'total_samples': len(parameter_samples),
        'sampling_time_per_sample_ms': (
            bench._timings.get('sample_parameters', [0])[0] * 1000 / n_samples
            if n_samples > 0 else 0
        )
    }


def benchmark_cross_model_comparison(bench: Benchmark,
                                    n_regions: int = 50,
                                    simulation_steps: int = 500) -> Dict[str, Any]:
    """Benchmark performance across different model types.
    
    Measures:
    - Comparative simulation speeds
    - Memory usage differences
    - Initialization time comparison
    """
    models = ['hopf', 'mpr', 'wilson-cowan', 'wong-wang', 'kuramoto', 'fitzhugh-nagumo']
    
    results = {}
    connectivity = create_test_connectivity(n_regions=n_regions)
    
    for model_name in models:
        if model_name not in list_models():
            continue
            
        # Get model
        model_class = get_model(model_name)
        model = model_class()
        
        # Get default parameters
        param_space = model.get_parameter_space()
        default_params = get_default_parameters(param_space)
        
        # Create simulation config
        config = SimulationConfig(
            dt=0.001,
            simulation_duration=simulation_steps * 0.001,  # Convert steps to seconds
            transient_duration=0.1,
            num_windows=10,
            use_pmap=False,
            seed=42
        )
        
        # Time simulation
        @bench.timeit(repeats=3, warmup=1)
        def run_simulation(m=model, c=config, conn=connectivity, params=default_params):
            return m.simulate(coupling_matrix=conn, parameters=params, config=c)
        
        simulation_result = run_simulation()
        
        # Store results
        results[model_name] = {
            'simulation_time_ms': bench._timings.get('run_simulation', [0])[0] * 1000,
            'success': simulation_result is not None,
            'n_parameters': len(param_space.parameters)
        }
    
    return {
        'n_models_tested': len(results),
        'models_tested': list(results.keys()),
        'results': results,
        'fastest_model': min(
            results.items(), 
            key=lambda x: x[1]['simulation_time_ms']
        )[0] if results else None,
        'slowest_model': max(
            results.items(), 
            key=lambda x: x[1]['simulation_time_ms']
        )[0] if results else None
    }


def run_model_benchmarks(benchmark_name: str = 'dynamics-models',
                        n_regions: int = 50,
                        simulation_steps: int = 1000,
                        n_samples: int = 50) -> Benchmark:
    """Run comprehensive dynamics model benchmarks.
    
    Args:
        benchmark_name: Name for this benchmark run
        n_regions: Number of regions in test connectivity
        simulation_steps: Number of simulation steps
        n_samples: Number of parameter samples
        
    Returns:
        Benchmark object with results
    """
    bench = Benchmark(benchmark_name)
    
    # Get available models
    available_models = list_models()
    
    # Run registry benchmarks
    registry_results = benchmark_model_registry(bench)
    
    # Run cross-model comparison
    comparison_results = benchmark_cross_model_comparison(
        bench, n_regions, simulation_steps
    )
    
    # Run benchmarks for each model
    model_results = {}
    for model_name in available_models:
        # Skip if model not implemented
        try:
            # Initialization benchmark
            init_results = benchmark_model_initialization(
                bench, model_name, n_regions
            )
            
            # Simulation benchmark
            sim_results = benchmark_simulation_speed(
                bench, model_name, n_regions, simulation_steps
            )
            
            # Parameter sampling benchmark
            sampling_results = benchmark_parameter_sampling(
                bench, model_name, n_samples
            )
            
            model_results[model_name] = {
                'initialization': init_results,
                'simulation': sim_results,
                'sampling': sampling_results
            }
        except Exception as e:
            model_results[model_name] = {
                'error': str(e),
                'skipped': True
            }
    
    # Add metadata about benchmarks
    bench.results.append(BenchmarkResult(
        name='benchmark_metadata',
        elapsed=0.0,
        metadata={
            'registry_results': registry_results,
            'comparison_results': comparison_results,
            'model_results': model_results,
            'parameters': {
                'n_regions': n_regions,
                'simulation_steps': simulation_steps,
                'n_samples': n_samples
            }
        }
    ))
    
    return bench