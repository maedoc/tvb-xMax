"""Pydantic request/response schemas for the API."""

from __future__ import annotations
from typing import Any, Optional, List
from pydantic import BaseModel, Field


class AccountCreate(BaseModel):
    username: str
    password: str
    tier: str = "free"


class Login(BaseModel):
    username: str
    password: str


class CompileRequest(BaseModel):
    model: str
    feature: str = "var"
    parcellation: Optional[str] = None
    connectivity: Optional[Any] = None
    connectivity_is_latent: bool = False
    parameters: dict = Field(default_factory=dict)
    sim_budget: int = 4096
    nlat: int = 16
    train_posterior: bool = True
    algo: str = "maf"


class InferRequest(BaseModel):
    artifact_id: str
    connectivity: Any
    connectivity_is_latent: bool = False
    parcellation: Optional[str] = None
    parameters: dict = Field(default_factory=dict)
    n_posterior: int = 1000
    target: str = "posterior"


class SwapRequest(BaseModel):
    artifact_id: str
    kind: str            # 'parcellation' | 'parameters' | 'model' | 'features'
    connectivity: Optional[Any] = None
    parcellation: Optional[str] = None
    parameters: Optional[dict] = None
    model: Optional[str] = None
    feature: Optional[str] = None
    n_posterior: int = 1000


class ArtifactInfo(BaseModel):
    id: str
    model: str
    feature: str
    nlat: int
    surrogate_mse: float
    speedup_vs_sim: float
    sbc_score: float
    c2st_score: float
    owner: str


class LeaderboardEntry(BaseModel):
    agent: str
    model: str
    feature: str
    speedup: float
    sbc: float
    c2st: float
    mse: float
    rank: int
