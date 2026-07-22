"""End-to-end portable compiler path with no JAX imports."""

import numpy as np

from ..ir import CompileReport, IRSpec, SimBudget
from ..surrogates import get_surrogate
from . import frontend, posterior
from . import numpy_codegen, numpy_lower, numpy_vectorize


def _pairs(sim_pairs):
    return (sim_pairs.U, sim_pairs.Theta, sim_pairs.XF) if isinstance(sim_pairs, SimBudget) else sim_pairs


def compile_spec(spec: IRSpec, crosscoder, sim_pairs, d_feat: int, mvn=None,
                 train_posterior: bool = True, algo: str = "mdn", **kw):
    spec = frontend.parse(spec)
    prog = numpy_lower.optimize(numpy_lower.lower(spec, crosscoder), mvn)
    pairs = tuple(np.asarray(x) for x in _pairs(sim_pairs))
    artifact = numpy_codegen.compile_artifact(spec.model, spec.feature, pairs, prog.nlat, d_feat, **kw)
    stages = {"backend": "numpy"}
    if train_posterior:
        posterior.attach_posterior(artifact, pairs[1], pairs[2], algo=algo)
    return CompileReport(artifact=artifact, stages=stages, speedup_vs_sim=float("nan"))


def run(artifact, spec: IRSpec, crosscoder, mvn=None):
    spec = frontend.parse(spec)
    prog = numpy_lower.optimize(numpy_lower.lower(spec, crosscoder), mvn)
    xf = artifact(prog.u, prog.theta)
    out = {"features": xf, "u": prog.u, "theta": prog.theta}
    if spec.target in ("posterior", "both") and artifact.posterior_sample:
        out["posterior"] = numpy_vectorize.batched_posterior(artifact, xf[None], spec.n_posterior)[:, 0]
    return out


def run_batch(artifact, specs, crosscoder, mvn=None):
    programs = [numpy_lower.optimize(numpy_lower.lower(frontend.parse(s), crosscoder), mvn) for s in specs]
    U = np.stack([p.u for p in programs]); theta = np.stack([p.theta for p in programs])
    return {"features": numpy_vectorize.batched_features(artifact, U, theta), "U": U, "Theta": theta}


def resolve_artifact(spec, crosscoder, cache, sim_pairs, d_feat, mvn=None, **kw):
    """Portable cache-or-compile equivalent of ``pipeline.resolve_artifact``."""
    nlat = get_surrogate(spec.model).nlat
    artifact = cache.get(spec.model, spec.feature, nlat)
    if artifact is None:
        artifact = compile_spec(spec, crosscoder, sim_pairs, d_feat, mvn=mvn, **kw).artifact
        cache.put(artifact)
    return artifact


def run_cached(spec, crosscoder, cache, sim_pairs, d_feat, mvn=None, **kw):
    nlat = get_surrogate(spec.model).nlat
    hit = cache.get(spec.model, spec.feature, nlat) is not None
    artifact = resolve_artifact(spec, crosscoder, cache, sim_pairs, d_feat, mvn=mvn, **kw)
    out = run(artifact, spec, crosscoder, mvn=mvn)
    out["cache"] = "hit" if hit else "miss"
    return out
