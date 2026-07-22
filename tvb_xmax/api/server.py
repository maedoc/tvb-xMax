"""FastAPI app factory wiring auth, rate limiting, and routes."""

from __future__ import annotations

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .auth import (create_account, verify_password, issue_jwt, verify_jwt,
                   account_by_api_key, record_usage)
from .ratelimit import check as rate_check
from .routes import compile as compile_routes
from .routes import infer as infer_routes
from .routes import swap as swap_routes
from .routes import leaderboard as lb_routes


def create_app(config: dict | None = None) -> FastAPI:
    config = config or {}
    app = FastAPI(title="tvb-xMax", version="0.1.0",
                  description="Advanced AI math compiler for virtual brain simulation")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # in-memory artifact store (swap for Redis/DB in prod)
    app.state.artifacts = {}
    app.state.crosscoder = None  # loaded at startup via TVBXMAX_CC env

    @app.middleware("http")
    async def auth_and_rate(request: Request, call_next):
        if request.url.path in ("/health", "/api/v1/account", "/api/v1/token"):
            return await call_next(request)
        # bearer JWT or X-API-Key
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key")
        account = None
        if auth.startswith("Bearer "):
            payload = verify_jwt(auth[7:])
            if payload:
                account = type("A", (), {"username": payload["sub"],
                                         "tier": payload["tier"], "api_key": ""})()
        elif api_key:
            account = account_by_api_key(api_key)
        if account is None:
            return JSONResponse(status_code=401, content={
                "error": "auth required: send X-API-Key or Authorization: Bearer"})
        ok, retry, remaining, msg = rate_check(
            account.username, account.tier, request.url.path)
        if not ok:
            return JSONResponse(status_code=429, content={
                "error": "rate limited", "retry_after": retry,
                "remaining": remaining})
        record_usage(account.username, request.url.path)
        request.state.account = account
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "tvb-xMax", "version": "0.1.0"}

    @app.post("/api/v1/account")
    async def acct(body: dict):
        from .models import AccountCreate
        a = AccountCreate(**body)
        acc = create_account(a.username, a.password, a.tier)
        return {"username": acc.username, "api_key": acc.api_key, "tier": acc.tier}

    @app.post("/api/v1/token")
    async def token(body: dict):
        from .models import Login
        l = Login(**body)
        if not verify_password(l.username, l.password):
            raise HTTPException(401, "bad credentials")
        from .auth import _conn
        with _conn() as c:
            row = c.execute("SELECT tier FROM accounts WHERE username=?",
                            (l.username,)).fetchone()
        tier = row[0] if row else "free"
        return {"access_token": issue_jwt(l.username, tier), "token_type": "bearer"}

    app.include_router(compile_routes.router, prefix="/api/v1")
    app.include_router(infer_routes.router, prefix="/api/v1")
    app.include_router(swap_routes.router, prefix="/api/v1")
    app.include_router(lb_routes.router, prefix="/api/v1")
    return app


app = create_app()
