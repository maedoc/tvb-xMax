"""JWT authentication + simple account store.

Accounts are stored in SQLite (one file, no external deps).  Each account
gets an API key (for header auth) and can mint JWTs (for bearer auth).
Rate limits and quota are attached to the account.
"""

from __future__ import annotations

import sqlite3
import hashlib
import hmac
import os
import time
import secrets
from dataclasses import dataclass
from typing import Optional

DB_PATH = os.environ.get("TVBXMAX_DB", "tvbxmax.db")
JWT_SECRET = os.environ.get("TVBXMAX_JWT_SECRET", secrets.token_hex(32))
JWT_TTL = 3600 * 24  # 24h


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        pw_hash TEXT,
        api_key TEXT UNIQUE,
        tier TEXT DEFAULT 'free',
        created REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY,
        username TEXT,
        endpoint TEXT,
        ts REAL
    )""")
    return c


def _hash_pw(pw: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), b"tvbxmax-salt", 100000).hex()


@dataclass
class Account:
    username: str
    api_key: str
    tier: str = "free"


def create_account(username: str, password: str, tier: str = "free") -> Account:
    api_key = secrets.token_urlsafe(32)
    with _conn() as c:
        c.execute(
            "INSERT INTO accounts(username, pw_hash, api_key, tier, created) "
            "VALUES(?,?,?,?,?)",
            (username, _hash_pw(password), api_key, tier, time.time()))
    return Account(username, api_key, tier)


def verify_password(username: str, password: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT pw_hash FROM accounts WHERE username=?", (username,)).fetchone()
    return row is not None and hmac.compare_digest(row[0], _hash_pw(password))


def account_by_api_key(api_key: str) -> Optional[Account]:
    with _conn() as c:
        row = c.execute(
            "SELECT username, tier FROM accounts WHERE api_key=?",
            (api_key,)).fetchone()
    return Account(row[0], api_key, row[1]) if row else None


def issue_jwt(username: str, tier: str) -> str:
    import base64, json
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": username, "tier": tier, "exp": time.time() + JWT_TTL}
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    signing_input = f"{b64(header)}.{b64(payload)}"
    sig = hmac.new(JWT_SECRET.encode(), signing_input.encode(), hashlib.sha256).hexdigest()
    return f"{signing_input}.{sig}"


def verify_jwt(token: str) -> Optional[dict]:
    import base64, json
    try:
        signing_input, sig = token.rsplit(".", 1)
        expected = hmac.new(JWT_SECRET.encode(), signing_input.encode(),
                            hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(
            signing_input.split(".")[1] + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def record_usage(username: str, endpoint: str) -> None:
    with _conn() as c:
        c.execute("INSERT INTO usage(username, endpoint, ts) VALUES(?,?,?)",
                  (username, endpoint, time.time()))


# tier -> (requests/min, max batch size)
TIER_LIMITS = {
    "free":   (60, 128),
    "pro":    (600, 4096),
    "agent":  (6000, 65536),
}
