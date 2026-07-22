"""Measure a long portable SDDE versus one surrogate feature prediction.

The budget is synthetic; this is a latency/throughput benchmark, not a
surrogate-fidelity claim.
"""

import time

import numpy as np

from tvb_xmax.compiler import numpy_codegen, numpy_sim


N_REGIONS = 76
N_STEPS = 10_000


def median_time(fn, repeats=3):
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return float(np.median(values))


def main():
    rng = np.random.default_rng(42)
    u = rng.normal(size=16).astype("f")
    theta = np.array([.2, .01, .1, .5], dtype="f")
    connectivity = rng.random((N_REGIONS, N_REGIONS), dtype="f") * .01
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)

    U = rng.normal(size=(128, 16)).astype("f")
    Theta = rng.random((128, 4), dtype="f")
    XF = rng.normal(size=(128, N_REGIONS)).astype("f")
    artifact = numpy_codegen.compile_artifact("hopf", "var", (U, Theta, XF), 16,
                                              N_REGIONS, niter=20)
    def simulate():
        return numpy_sim.hopf_features(connectivity, [.2, .01, .1, 1.0], n_steps=N_STEPS)

    forward = median_time(lambda: artifact(u, theta), repeats=20)
    sdde = median_time(simulate)
    print(f"SDDE: {N_REGIONS} regions, {N_STEPS:,} steps: {sdde:.3f} s")
    print(f"Surrogate forward: {forward * 1e3:.3f} ms")
    print(f"SDDE / forward: {sdde / forward:,.0f}x")


if __name__ == "__main__":
    main()
