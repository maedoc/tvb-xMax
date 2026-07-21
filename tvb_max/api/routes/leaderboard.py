"""GET /leaderboard - agent rankings by posterior calibration + speedup."""
from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/leaderboard")
async def leaderboard(request: Request):
    """Aggregate artifact metrics into a ranked table.

    Scoring (lower is better):  score = c2st + (1 - sbc) + log10(mse+1) - log10(speedup)
    """
    store = request.app.state.artifacts
    rows = []
    for e in store.values():
        if e.get("status") != "ready":
            continue
        a = e["artifact"]
        c2st = getattr(a, "c2st_score", 0.5) or 0.5
        sbc = getattr(a, "sbc_score", 0.5) or 0.5
        mse = getattr(a, "surrogate_mse", 1.0) or 1.0
        sp = getattr(a, "speedup_vs_sim", 1.0) or 1.0
        import math
        score = c2st + (1 - sbc) + math.log10(mse + 1) - math.log10(sp + 1)
        rows.append({"agent": e.get("owner", "?"), "model": a.model,
                     "feature": a.feature, "speedup": sp, "sbc": sbc,
                     "c2st": c2st, "mse": mse, "score": score})
    rows.sort(key=lambda r: r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {"leaderboard": rows}
