"""POST /infer - run a compiled artifact (the fast path)."""
from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from ..models import InferRequest

router = APIRouter()


@router.post("/infer")
async def infer(req: InferRequest, request: Request):
    """Run the surrogate + posterior. This is the ~ms fast path."""
    store = request.app.state.artifacts
    entry = store.get(req.artifact_id)
    if entry is None or entry.get("status") != "ready":
        raise HTTPException(409, f"artifact {req.artifact_id} not ready")
    artifact = entry.get("artifact")
    if artifact is None:
        raise HTTPException(500, "artifact has no compiled object")
    from ...compiler import ir, pipeline
    spec = ir.IRSpec(
        model=artifact.model, feature=artifact.feature,
        connectivity=req.connectivity,
        connectivity_is_latent=req.connectivity_is_latent,
        parcellation=req.parcellation, parameters=req.parameters,
        target=req.target, n_posterior=req.n_posterior)
    cc = request.app.state.crosscoder
    out = pipeline.run(artifact, spec, cc)
    return {"features": out["features"].tolist(),
            "posterior": out.get("posterior", []).tolist() if req.target != "features" else None}
