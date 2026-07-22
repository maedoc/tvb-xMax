"""POST /swap - apply a free swap and re-run."""
from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from ..models import SwapRequest

router = APIRouter()


@router.post("/swap")
async def swap(req: SwapRequest, request: Request):
    """Apply a parcellation/parameter/model/feature swap and re-run.

    Parcellation + parameter swaps reuse the same artifact (free).
    Model + feature swaps select a different artifact (must be compiled).
    """
    store = request.app.state.artifacts
    entry = store.get(req.artifact_id)
    if entry is None or entry.get("status") != "ready":
        raise HTTPException(409, "artifact not ready")
    artifact = entry["artifact"]
    from ...compiler import ir, swap as swap_mod, pipeline
    base = ir.IRSpec(
        model=artifact.model, feature=artifact.feature,
        connectivity=req.connectivity or [0.0]*artifact.nlat,
        connectivity_is_latent=req.connectivity is not None,
        parcellation=req.parcellation, parameters=req.parameters or {},
        target="posterior", n_posterior=req.n_posterior)
    kw = {}
    if req.kind == "parcellation":
        kw = {"connectivity": req.connectivity, "parcellation": req.parcellation}
    elif req.kind == "parameters":
        kw = req.parameters or {}
    elif req.kind == "model":
        kw = {"model": req.model}
        # need a different artifact
        for e in store.values():
            if e.get("status") == "ready" and e["artifact"].model == req.model:
                artifact = e["artifact"]; break
    elif req.kind == "features":
        kw = {"feature": req.feature}
    new_spec = swap_mod.apply_swap(base, req.kind, **kw)
    out = pipeline.run(artifact, new_spec, request.app.state.crosscoder)
    return {"features": out["features"].tolist(),
            "posterior": out.get("posterior", []).tolist()}
