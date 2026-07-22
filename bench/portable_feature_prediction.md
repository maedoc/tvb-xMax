# Portable feature-prediction benchmark

Measured 2026-07-22 on an Apple Silicon CPU with the NumPy/Autograd backend.

| Workload | Time |
|---|---:|
| Stochastic delayed Hopf simulation, 76 regions, 10,000 steps | 2.398 s |
| One compiled surrogate feature prediction | 0.024 ms |
| Feature-prediction speedup | **98,209×** |

Configuration: a seeded (`42`) dense 76-region synthetic connectome; a
TVBL-style delayed Heun integrator; normalized Hopf parameters
`[k=0.2, D=0.01, eta=0.1, omega=1.0]`; and a 2×128 tanh surrogate trunk plus
a 76-output linear feature head. The surrogate is trained on a synthetic
budget, so this is an execution-throughput result, **not a fidelity or
biological-validity claim**.

Reproduce with:

```bash
python bench/bench_portable_long_sdde.py
```
