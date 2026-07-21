"""FastAPI server with accounts, JWT auth, and per-user rate limiting.

Endpoints:
  POST /api/v1/compile   - compile a spec into an artifact (async job)
  POST /api/v1/infer     - run a compiled artifact (fast path)
  POST /api/v1/swap      - apply a free swap and re-run
  GET  /api/v1/artifacts - list compiled artifacts
  GET  /api/v1/leaderboard
  POST /api/v1/account   - create account
  POST /api/v1/token     - login -> JWT
"""

from .server import create_app, app

__all__ = ["create_app", "app"]
