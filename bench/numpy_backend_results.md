# Portable NumPy backend benchmark

Measured on 2026-07-22, Apple Silicon CPU, with warmed 2×128-tanh-trunk +
linear-head forwards (`nlat=16`, `d_param=4`, `d_feat=76`). These are equal
surrogate-forward workloads, not end-to-end SDE speedups.

| Batch | NumPy | JAX | NumPy / JAX |
|---:|---:|---:|---:|
| 1 | 0.017 ms | 0.022 ms | 0.74× |
| 64 | 0.311 ms | 0.086 ms | 3.60× |
| 4,096 | 17.489 ms | 3.472 ms | 5.04× |

NumPy wins for single-request latency in this configuration; JAX wins once
batching dominates. Reproduce with `python bench/bench_backend_parity.py`
from the `dev` environment.
