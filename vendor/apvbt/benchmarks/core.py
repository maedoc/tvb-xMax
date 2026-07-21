"""Core benchmarking utilities for timing and memory measurement."""

import time
import functools
import numpy as np
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass, field
from collections import defaultdict
import json
import sys
import tracemalloc


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    elapsed: float  # seconds
    memory_usage: Optional[float] = None  # bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'name': self.name,
            'elapsed': self.elapsed,
            'metadata': self.metadata
        }
        if self.memory_usage is not None:
            result['memory_usage'] = self.memory_usage
        return result


class Benchmark:
    """Performance benchmarking harness."""
    
    def __init__(self, name: str = "benchmark"):
        self.name = name
        self.results: List[BenchmarkResult] = []
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._memory: Dict[str, List[float]] = defaultdict(list)
    
    def time(self, func: Callable, name: Optional[str] = None, **kwargs) -> Any:
        """Time execution of a function.
        
        Args:
            func: Function to time
            name: Name for this benchmark (defaults to function name)
            **kwargs: Arguments passed to func
            
        Returns:
            Result of func(**kwargs)
        """
        func_name = name or func.__name__
        start = time.perf_counter()
        result = func(**kwargs)
        elapsed = time.perf_counter() - start
        
        self._timings[func_name].append(elapsed)
        self.results.append(BenchmarkResult(
            name=func_name,
            elapsed=elapsed,
            metadata={'args': list(kwargs.keys())}
        ))
        
        return result
    
    def timeit(self, repeats: int = 3, warmup: int = 1):
        """Decorator to time a function multiple times.
        
        Args:
            repeats: Number of timed repetitions
            warmup: Number of warmup runs (not timed)
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Warmup runs
                for _ in range(warmup):
                    func(*args, **kwargs)
                
                # Timed runs
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                
                # Store results
                func_name = func.__name__
                for t in times:
                    self._timings[func_name].append(t)
                    self.results.append(BenchmarkResult(
                        name=func_name,
                        elapsed=t,
                        metadata={'args': str(args), 'kwargs': kwargs}
                    ))
                
                # Return the last result
                return result
            return wrapper
        return decorator
    
    def measure_memory(self, func: Callable, name: Optional[str] = None, *args, **kwargs) -> Any:
        """Measure memory usage of a function.
        
        Args:
            func: Function to measure
            name: Name for this benchmark (defaults to function name)
            *args: Positional arguments passed to func
            **kwargs: Keyword arguments passed to func
            
        Returns:
            Result of func(*args, **kwargs)
        """
        func_name = name or func.__name__
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]
        result = func(*args, **kwargs)
        end_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        memory_used = end_memory - start_memory
        
        self._memory[func_name].append(memory_used)
        self.results.append(BenchmarkResult(
            name=func_name,
            elapsed=0.0,  # Not timed
            memory_usage=memory_used,
            metadata={'args': args, 'kwargs': kwargs}
        ))
        
        return result
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Generate summary statistics for all benchmarks."""
        summary = {}
        for name, times in self._timings.items():
            if times:
                summary[name] = {
                    'mean': float(np.mean(times)),
                    'std': float(np.std(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'n': len(times)
                }
        
        for name, mem in self._memory.items():
            if mem:
                key = f'{name}_memory'
                summary[key] = {
                    'mean': float(np.mean(mem)),
                    'std': float(np.std(mem)),
                    'min': float(np.min(mem)),
                    'max': float(np.max(mem)),
                    'n': len(mem)
                }
        
        return summary
    
    def save(self, path: str):
        """Save benchmark results to JSON file."""
        data = {
            'name': self.name,
            'results': [r.to_dict() for r in self.results],
            'summary': self.summary()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all stored results."""
        self.results.clear()
        self._timings.clear()
        self._memory.clear()


# Global benchmark instance for simple usage
_default_benchmark = Benchmark('default')


def timeit(func=None, repeats=3, warmup=1):
    """Simple decorator for timing functions.
    
    Usage:
        @timeit
        def my_func():
            ...
            
        @timeit(repeats=5, warmup=2)
        def another_func():
            ...
    """
    if func is None:
        return lambda f: _default_benchmark.timeit(repeats=repeats, warmup=warmup)(f)
    else:
        return _default_benchmark.timeit(repeats=repeats, warmup=warmup)(func)


def measure_memory(func=None, name=None):
    """Decorator for measuring memory usage."""
    if func is None:
        # Called as @measure_memory(name='something')
        return lambda f: measure_memory(f, name=name)
    else:
        # Called as @measure_memory
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _default_benchmark.measure_memory(func, name, *args, **kwargs)
        return wrapper


def get_default_benchmark() -> Benchmark:
    """Get the default global benchmark instance."""
    return _default_benchmark