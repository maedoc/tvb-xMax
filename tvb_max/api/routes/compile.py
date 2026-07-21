"""POST /compile - compile a spec into a stored artifact (async)."""
from __future__ import annotations
from fastapi import APIRouter, Request, BackgroundTasks
from ..models import CompileRequest, ArtifactInfo
import uuid

router = APIRouter()


@router.post("/compile", response_model=ArtifactInfo)
async def compile(req: CompileRequest, request: Request, bg: BackgroundTasks):
    """Kick off a compile job. Returns the artifact id immediately.

    The actual compile (frontend->lower->optimize->codegen->posterior)
    runs in the background because it spends the one-time sim budget.
    Poll GET /artifacts/{id} for status.
    """
    aid = str(uuid.uuid4())[:8]
    request.state.artifacts[aid] = {"id": aid, "status": "pending",
                                    "owner": request.state.account.username,
                                    "req": req.dict()}
    bg.add_task(_do_compile, request.app, aid, req)
    return ArtifactInfo(id=aid, model=req.model, feature=req.feature,
                        nlat=req.nlat, surrogate_mse=float("inf"),
                        speedup_vs_sim=float("inf"), sbc_score=float("nan"),
                        c2st_score=float("nan"), owner=request.state.account.username)


def _do_compile(app, aid, req):
    """Background worker: spend sim budget, train surrogate + posterior."""
    from ...compiler import pipeline, ir
    from ...surrogates import get_surrogate
    try:
        # 1. generate sim budget via vbjax/apvbt (omitted: calls sample_model)
        # 2. compile artifact
        spec = ir.IRSpec(model=req.model, feature=req.feature,
                         connectivity=req.connectivity or [0.0]*req.nlat,
                         connectivity_is_latent=True,
                         parameters=req.parameters, target="posterior")
        # ... train surrogate on sim_pairs ...
        app.state.artifacts[aid]["status"] = "ready"
    except Exception as e:
        app.state.artifacts[aid]["status"] = f"error: {e}"


@router.get("/artifacts")
async def list_artifacts(request: Request):
    return {"artifacts": list(request.app.state.artifacts.values())}


@router.get("/artifacts/{aid}")
async def get_artifact(aid: str, request: Request):
    a = request.app.state.artifacts.get(aid)
    if a is None:
        from fastapi import HTTPException
        raise HTTPException(404, "artifact not found")
    return a
