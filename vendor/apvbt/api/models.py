from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class ModelInfo(BaseModel):
    """Model metadata information."""

    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    version: Optional[str] = Field(None, description="Model version")
    parameters: Optional[List[str]] = Field(
        default_factory=list, description="Parameter names"
    )
    loaded: bool = Field(default=False, description="Whether model is loaded in memory")

    model_config = {"extra": "forbid"}


class InferenceRequest(BaseModel):
    """Inference request schema."""

    model_id: str = Field(..., description="Model identifier to use for inference")
    features: List[float] = Field(..., description="Observed feature vector")
    num_samples: int = Field(
        default=200, ge=1, le=10000, description="Number of posterior samples"
    )
    seed: Optional[int] = Field(
        None, ge=0, description="Random seed for reproducibility"
    )

    @field_validator("features")
    @classmethod
    def validate_features_not_empty(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("Features list cannot be empty")
        return v

    model_config = {"extra": "forbid"}


class InferenceMetadata(BaseModel):
    """Metadata about inference operation."""

    model_id: str
    num_samples: int
    computation_time_ms: float
    timestamp: str
    seed: Optional[int] = None


class InferenceResponse(BaseModel):
    """Inference response schema."""

    samples: List[List[float]] = Field(
        ..., description="Posterior samples (num_samples x num_parameters)"
    )
    mean: List[float] = Field(..., description="Posterior mean for each parameter")
    std: List[float] = Field(..., description="Posterior std for each parameter")
    metadata: InferenceMetadata

    model_config = {"extra": "forbid"}


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    timestamp: str = Field(..., description="Error timestamp (ISO 8601)")

    model_config = {"extra": "forbid"}
