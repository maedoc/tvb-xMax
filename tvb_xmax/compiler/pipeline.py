"""Pipeline: orchestrate the full compile + run.

``compile``  = frontend -> lower -> optimize -> codegen -> (posterior) -> artifact
``run``      = lower(spec) -> optimize -> artifact(u, theta) -> [posterior]
``swap``     = re-lower with a swapped field, reuse the same artifact

Cached variants (:func:`resolve_artifact`, :func:`run_cached`) check an
:class:`ArtifactCache` before compiling, enabling model/feature swaps to
reuse previously compiled artifacts.
"""

from __future__ import annotations

import time

import jax.numpy as jnp

from ..ir import IRSpec, CompiledArtifact, CompileReport, SimBudget
from . import frontend, lower, optimize, codegen, vectorize, posterior
from .sim_fn import make_real_sim_fn


def _unpack_sim_pairs(sim_pairs):
    """Return ``(U, Theta, XF)`` from either a :class:`SimBudget` or a tuple."""
    if isinstance(sim_pairs, SimBudget):
        return sim_pairs.U, sim_pairs.Theta, sim_pairs.XF
    return sim_pairs


def compile_spec(spec: IRSpec, crosscoder, sim_pairs, d_feat: int,
                 mvn=None, train_posterior: bool = True,
                 algo: str = "mdn", **codegen_kw) -> CompileReport:
    """Full compile: produce a :class:`CompiledArtifact` from a spec + sims.

    Args:
        spec: source program (model/feature/parc chosen here become the
            artifact's identity; parc is just the training source).
        crosscoder: trained ``vbjax.CrossCoder`` / apvbt ``XCode``.
        sim_pairs: :class:`SimBudget` or raw ``(U, Theta, XF)`` tuple.
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

    U, Theta, XF = _unpack_sim_pairs(sim_pairs)
    sim_tuple = (U, Theta, XF)

    t0 = time.perf_counter()
    artifact = codegen.compile_artifact(
        spec.model, spec.feature, sim_tuple, prog.nlat, d_feat, **codegen_kw)
    stages["codegen"] = time.perf_counter() - t0

    if train_posterior:
        t0 = time.perf_counter()
        posterior.attach_posterior(artifact, Theta, XF, algo=algo)
        stages["posterior"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            artifact.sbc_score = posterior.compute_sbc(artifact, Theta, XF)
            artifact.c2st_score = posterior.compute_c2st(artifact, Theta, XF)
            stages["diagnostics"] = time.perf_counter() - t0
        except Exception as e:
            stages["diagnostics"] = f"failed: {e}"

    # speedup estimate vs the real SDE simulation
    speedup = float("nan")
    try:
        sim_fn = make_real_sim_fn(spec.model, crosscoder, prog.nlat,
                                  parc=spec.parcellation)
        bench = vectorize.benchmark_speedup(
            artifact, sim_fn, jnp.asarray(U[:64]), jnp.asarray(Theta[:64]))
        speedup = bench["speedup"]
        stages["benchmark"] = bench
    except NotImplementedError as e:
        stages["benchmark"] = f"not measured: {e}"
    except Exception as e:
        stages["benchmark"] = f"benchmark failed: {e}"

    return CompileReport(artifact=artifact, stages=stages, speedup_vs_sim=speedup)


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


def resolve_artifact(spec: IRSpec, crosscoder, cache, sim_pairs, d_feat,
                     mvn=None, **compile_kw) -> CompiledArtifact:
    """Return a compiled artifact for *spec*, using cache if available.

    If the cache holds an artifact whose ``(model, feature, nlat)`` matches
    *spec* it is returned immediately; otherwise the full compiler pipeline
    runs and the result is cached.

    Args:
        spec: source program (model, feature, parcellation, parameters, ...).
        crosscoder: trained ``vbjax.CrossCoder`` / apvbt ``XCode``.
        cache: :class:`ArtifactCache` to check before compiling.
        sim_pairs: ``(U, Theta, XF)`` one-time simulation budget.
        d_feat: feature dimension.
        mvn: cohort MVN for latent whitening (optional).
        **compile_kw: forwarded to :func:`compile_spec`.

    Returns:
        CompiledArtifact from cache or freshly compiled.
    """
    report = cache.load_or_compile(spec, crosscoder, sim_pairs, d_feat,
                                   mvn=mvn, **compile_kw)
    return report.artifact


def run_cached(spec: IRSpec, crosscoder, cache, sim_pairs, d_feat,
               mvn=None, **compile_kw) -> dict:
    """Resolve artifact (cache-or-compile) then run inference.

    One-shot convenience: calls :func:`resolve_artifact` and then
    :func:`run`, returning the usual output dict with an added ``"cache"``
    key whose value is ``"hit"`` or ``"miss"``.

    For model/feature swaps the spec's ``(model, feature, nlat)`` determines
    which artifact is fetched; parcellation/parameter swaps reuse the same
    artifact.

    Args:
        spec: source program.
        crosscoder: trained cross-coder.
        cache: :class:`ArtifactCache` to check before compiling.
        sim_pairs: ``(U, Theta, XF)`` one-time simulation budget.
        d_feat: feature dimension.
        mvn: cohort MVN for latent whitening (optional).
        **compile_kw: forwarded to :func:`compile_spec`.

    Returns:
        dict with keys ``"features"``, ``"u"``, ``"theta"``,
        optionally ``"posterior"``, and ``"cache"`` (``"hit"`` / ``"miss"``).
    """
    report = cache.load_or_compile(spec, crosscoder, sim_pairs, d_feat,
                                   mvn=mvn, **compile_kw)
    out = run(report.artifact, spec, crosscoder, mvn=mvn)
    out["cache"] = report.stages.get("cache", "unknown")
    return out
