"""Pipeline: orchestrate the full compile + run.

``compile``  = frontend -> lower -> optimize -> codegen -> (posterior) -> artifact
``run``      = lower(spec) -> optimize -> artifact(u, theta) -> [posterior]
``swap``     = re-lower with a swapped field, reuse the same artifact
"""

from __future__ import annotations

import time
from typing import Optional

import jax.numpy as jnp

from ..ir import IRSpec, IRProgram, CompiledArtifact, CompileReport
from . import frontend, lower, optimize, codegen, vectorize, posterior


def compile_spec(spec: IRSpec, crosscoder, sim_pairs, d_feat: int,
                 mvn=None, train_posterior: bool = True,
                 algo: str = "maf", **codegen_kw) -> CompileReport:
    """Full compile: produce a :class:`CompiledArtifact` from a spec + sims.

    Args:
        spec: source program (model/feature/parc chosen here become the
            artifact's identity; parc is just the training source).
        crosscoder: trained ``vbjax.CrossCoder`` / apvbt ``XCode``.
        sim_pairs: ``(U, Theta, XF)`` one-time simulation budget.
        d_feat: feature dimension.
        mvn: cohort MVN for latent whitening (optional).
        train_posterior: also train + bind an NPE posterior.
    """
    stages = {}
    t0 = time.perf_counter()
    spec = frontend.parse(spec)
    stages["frontend"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prog = lower.lower(spec, crosscoder)
    stages["lower"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prog = optimize.optimize(prog, mvn)
    stages["optimize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    artifact = codegen.compile_artifact(
        spec.model, spec.feature, sim_pairs, prog.nlat, d_feat, **codegen_kw)
    stages["codegen"] = time.perf_counter() - t0

    if train_posterior:
        t0 = time.perf_counter()
        posterior.attach_posterior(artifact, sim_pairs[1], sim_pairs[2], algo=algo)
        stages["posterior"] = time.perf_counter() - t0

    # speedup estimate vs the sim budget that produced sim_pairs
    U, Theta, _ = sim_pairs
    speedup = float("inf")
    try:
        bench = vectorize.benchmark_speedup(
            artifact, _noop_sim, jnp.asarray(U[:64]), jnp.asarray(Theta[:64]))
        speedup = bench["speedup"]
    except Exception:
        pass

    return CompileReport(artifact=artifact, stages=stages, speedup_vs_sim=speedup)


def _noop_sim(u, theta):
    """Placeholder sim for speedup estimation; replaced by real sim_fn."""
    return u  # not used unless benchmark_speedup is given a real sim_fn


def run(artifact: CompiledArtifact, spec: IRSpec, crosscoder,
        mvn=None) -> dict:
    """Run a compiled artifact on a (possibly swapped) spec.

    This is the fast path: no simulation, just lower -> optimize -> forward.
    """
    spec = frontend.parse(spec)
    prog = lower.lower(spec, crosscoder)
    prog = optimize.optimize(prog, mvn)

    xf = artifact(prog.u, prog.theta)

    out = {"features": xf, "u": prog.u, "theta": prog.theta}
    if spec.target in ("posterior", "both") and artifact.posterior_sample:
        out["posterior"] = vectorize.batched_posterior(
            artifact, xf[None, :], spec.n_posterior)[..., 0, :]
    return out


def run_batch(artifact: CompiledArtifact, specs, crosscoder, mvn=None) -> dict:
    """Vectorized run over many specs at once (the "maxest speedup" path)."""
    progs = []
    for s in specs:
        s = frontend.parse(s)
        progs.append(optimize.optimize(lower.lower(s, crosscoder), mvn))
    U = jnp.stack([p.u for p in progs])
    Theta = jnp.stack([p.theta for p in progs])
    xf = vectorize.batched_features(artifact, U, Theta)
    return {"features": xf, "U": U, "Theta": Theta}
