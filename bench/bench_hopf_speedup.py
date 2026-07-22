"""Record honest speedup measurements: real vbjax Hopf SDE vs surrogate.

Builds a mock cross-coder with working encode + decode_conn, compiles a
Hopf surrogate, and measures t_sim / t_surrogate across multiple
configurations (varying n_steps and batch sizes).
"""
import math
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

import tvb_xmax
from tvb_xmax import ir
from tvb_xmax.compiler import codegen, pipeline, vectorize
from tvb_xmax.compiler.sim_fn import make_real_sim_fn
from tvb_xmax.surrogates import get_surrogate

NLAT = 16
N_REGIONS = 76
N_STEPS_LIST = [1000, 5000, 10000]
BATCH_LIST = [64, 256, 1024, 4096]
N_BUDGET = 4096
NITER = 1000


class MockArch:
    def __init__(self, wbs):
        self.wbs = wbs


class MockCrossCoder:
    """Cross-coder stub with working _get_arch + decode_conn."""

    def __init__(self, nlat, n_regions):
        self.parcs = ["parc_a"]
        self.variational = False
        n_triu = n_regions * (n_regions - 1) // 2
        self._n_triu = n_triu

        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        ew = jax.random.normal(k1, (n_triu, nlat)) * 0.1
        eb = jax.random.normal(k2, (nlat,)) * 0.1
        dw = jax.random.normal(k3, (nlat, n_triu)) * 0.1
        db = jax.random.normal(k4, (n_triu,)) * 0.1

        self.means = [jnp.zeros(n_triu)]
        self.stds = [jnp.ones(n_triu)]
        self.scales = [jnp.ones(n_triu)]
        self.norm_types = ["zscore"]
        self._wbs = [((ew, eb), (dw, db))]
        self._dw = dw
        self._db = db

    def _get_arch(self, nlat):
        return MockArch(self._wbs)

    def decode_conn(self, nlat, parc, z, clip_positive=False):
        triu = z @ self._dw + self._db
        if clip_positive:
            triu = jnp.maximum(triu, 0.0)
        n = int((1 + math.sqrt(1 + 8 * triu.shape[-1])) / 2)
        C = jnp.zeros((n, n))
        idx = jnp.triu_indices(n, k=1)
        C = C.at[idx].set(triu[0])
        return (C + C.T)[None]


def make_toy_budget(nlat, d_param, d_feat, n=N_BUDGET):
    key = jax.random.PRNGKey(42)
    U = jax.random.normal(key, (n, nlat))
    Theta = jax.random.uniform(key, (n, d_param))
    XF = jax.random.normal(key, (n, d_feat))
    return U, Theta, XF


def main():
    surr = get_surrogate("hopf")
    d_param = len(surr.param_names)
    d_feat = N_REGIONS

    cc = MockCrossCoder(NLAT, N_REGIONS)
    U, Theta, XF = make_toy_budget(NLAT, d_param, d_feat)

    spec = ir.IRSpec(
        model="hopf",
        connectivity=jnp.zeros(NLAT),
        connectivity_is_latent=True,
        parameters={"k": 0.5, "D": 0.3},
        feature="var",
        target="features",
    )

    report = pipeline.compile_spec(
        spec, cc, (U, Theta, XF), d_feat, train_posterior=False, niter=NITER
    )
    artifact = report.artifact
    mse = artifact.surrogate_mse
    backend = jax.default_backend()
    print(f"Compiled Hopf surrogate: mse={mse:.6f}, n_regions={N_REGIONS}")
    print()

    results = []
    single_results = []
    for n_steps in N_STEPS_LIST:
        sim_fn = make_real_sim_fn(
            "hopf", cc, NLAT, "parc_a", n_steps=n_steps
        )
        xf_test = sim_fn(U[0], Theta[0])
        assert xf_test.shape == (d_feat,), f"sim_fn bad shape: {xf_test.shape}"

        single = vectorize.benchmark_single_feature_eval(
            artifact, sim_fn, U[0], Theta[0]
        )
        single_results.append({"n_steps": n_steps, **single})
        print(
            f"  steps={n_steps:5d}  single SDE={single['t_sim']:.6f}s  "
            f"single surrogate={single['t_surrogate'] * 1000:.3f}ms  "
            f"speedup={single['speedup']:8.1f}x"
        )

        for batch in BATCH_LIST:
            key = jax.random.PRNGKey(batch + n_steps)
            Ub = jax.random.normal(key, (batch, NLAT))
            Tb = jax.random.uniform(key, (batch, d_param))
            bench = vectorize.benchmark_speedup(artifact, sim_fn, Ub, Tb)
            row = {
                "n_steps": n_steps,
                "batch": batch,
                "t_sim": bench["t_sim"],
                "t_surr": bench["t_surrogate"],
                "speedup": bench["speedup"],
            }
            results.append(row)
            print(
                f"  steps={n_steps:5d}  batch={batch:5d}  "
                f"t_sim={row['t_sim']:.3f}s  "
                f"t_surr={row['t_surr']:.6f}s  "
                f"speedup={row['speedup']:8.1f}x"
            )

    long_single = single_results[-1]
    t_per_sim = long_single["t_sim"]
    t_sbi_budget = t_per_sim * N_BUDGET
    t_forward = long_single["t_surrogate"]
    amortized = t_sbi_budget / t_forward

    print()
    print(f"Single SDE time:          {t_per_sim:.4f}s")
    print(f"SBI budget ({N_BUDGET} sims):  {t_sbi_budget:.1f}s = {t_sbi_budget/60:.1f}min")
    print(f"Single surrogate forward: {t_forward:.6f}s = {t_forward*1000:.3f}ms")
    print(f"Single-call speedup:      {long_single['speedup']:,.1f}x")
    print(f"Amortized speedup:        {amortized:,.0f}x")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 50
    lines = [
        sep,
        f"tvb-xMax speedup benchmark — {timestamp}",
        sep,
        f"Model:          hopf (Hopf oscillator)",
        f"Feature:        var (temporal variance)",
        f"Backend:        {backend}",
        f"Regions:         {N_REGIONS}",
        f"Sim budget:      {N_BUDGET} samples",
        f"Surrogate:       trunk(2x128 tanh) + head(1x{d_feat} linear)",
        f"Surrogate MSE:   {mse:.6f}",
        f"Training iters:   {NITER}",
        "",
        "| n_steps | single SDE (s) | single surrogate (ms) | single-call speedup |",
        "|---------|----------------|-----------------------|---------------------|",
    ]
    for r in single_results:
        lines.append(
            f"| {r['n_steps']:7d} | {r['t_sim']:14.6f} | "
            f"{r['t_surrogate'] * 1000:21.3f} | {r['speedup']:18.1f}x |"
        )
    lines += [
        "",
        "| n_steps | batch | t_sim (s) | t_surr (s) | speedup |",
        "|---------|-------|-----------|------------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['n_steps']:7d} | {r['batch']:5d} | "
            f"{r['t_sim']:9.3f} | {r['t_surr']:10.6f} | "
            f"{r['speedup']:8.1f}x |"
        )
    lines += [
        "",
        f"Single SDE:          {t_per_sim:.4f}s",
        f"SBI budget ({N_BUDGET} sims): {t_sbi_budget:.1f}s ({t_sbi_budget/60:.1f} min)",
        f"Single surrogate:    {t_forward:.6f}s ({t_forward*1000:.3f} ms)",
        f"Single-call speedup: {long_single['speedup']:,.1f}x",
        f"Amortized speedup:   {amortized:,.0f}x",
        sep,
        "",
    ]

    output = "\n".join(lines)
    print()
    print(output)

    results_path = os.path.join(os.path.dirname(__file__), "results.md")
    with open(results_path, "w") as f:
        f.write("# tvb-xMax Speedup Benchmarks\n\n")
        f.write(output)
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
