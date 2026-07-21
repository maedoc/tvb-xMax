from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import json


class ApvbtError(Exception):
    """Base exception for APVBT errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(ApvbtError):
    """Request validation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class ModelNotFoundError(ApvbtError):
    """Requested model not found."""

    def __init__(self, model_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"Model '{model_id}' not found"
        if details is None:
            details = {"model_id": model_id}
        super().__init__(message, status_code=404, details=details)


class InferenceError(ApvbtError):
    """Inference computation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)
