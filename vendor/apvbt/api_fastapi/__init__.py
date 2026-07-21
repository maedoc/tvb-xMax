"""
FastAPI implementation of APVBT REST API.

This module provides a FastAPI-based alternative to the Flask API
with async support and improved performance.
"""

from apvbt.api_fastapi.server import create_app
from apvbt.api_fastapi.metrics import (
    record_inference_request,
    record_job_submitted,
    record_job_started,
    record_job_finished,
    record_job_failed,
    update_queue_stats,
    get_metrics,
)

__all__ = [
    "create_app",
    "record_inference_request",
    "record_job_submitted",
    "record_job_started",
    "record_job_finished",
    "record_job_failed",
    "update_queue_stats",
    "get_metrics",
]