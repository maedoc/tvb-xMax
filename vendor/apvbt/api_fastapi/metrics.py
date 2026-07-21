"""
Prometheus metrics for APVBT FastAPI server.

This module provides custom Prometheus metrics for monitoring APVBT API
and job queue performance.
"""

import time
import logging
from typing import Optional, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

logger = logging.getLogger(__name__)

# Use default registry for compatibility with instrumentator
registry = REGISTRY

# Custom metrics
INFERENCE_REQUESTS_TOTAL = Counter(
    'apvbt_inference_requests_total',
    'Total number of inference requests',
    ['model_id', 'type'],  # type: 'single' or 'batch'
    registry=registry
)

INFERENCE_DURATION_SECONDS = Histogram(
    'apvbt_inference_duration_seconds',
    'Duration of inference requests in seconds',
    ['model_id', 'type'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=registry
)

JOBS_TOTAL = Counter(
    'apvbt_jobs_total',
    'Total number of jobs submitted',
    ['job_type', 'status'],  # status: 'submitted', 'started', 'finished', 'failed'
    registry=registry
)

JOBS_QUEUE_SIZE = Gauge(
    'apvbt_jobs_queue_size',
    'Current number of jobs in queue',
    ['queue_name'],
    registry=registry
)

JOBS_ACTIVE = Gauge(
    'apvbt_jobs_active',
    'Current number of active (running) jobs',
    ['queue_name'],
    registry=registry
)

MODEL_REQUESTS_TOTAL = Counter(
    'apvbt_model_requests_total',
    'Total number of requests per model',
    ['model_id', 'endpoint'],  # endpoint: 'infer', 'batch', 'jobs'
    registry=registry
)

# Metrics update functions

def record_inference_request(model_id: str, request_type: str, duration: float):
    """
    Record an inference request metric.
    
    Args:
        model_id: Model identifier
        request_type: 'single' or 'batch'
        duration: Request duration in seconds
    """
    INFERENCE_REQUESTS_TOTAL.labels(model_id=model_id, type=request_type).inc()
    INFERENCE_DURATION_SECONDS.labels(model_id=model_id, type=request_type).observe(duration)
    MODEL_REQUESTS_TOTAL.labels(model_id=model_id, endpoint='infer' if request_type == 'single' else 'batch').inc()

def record_job_submitted(job_type: str, queue_name: str = 'default'):
    """
    Record a job submission.
    
    Args:
        job_type: Type of job ('inference', 'batch_inference', etc.)
        queue_name: Name of the queue
    """
    JOBS_TOTAL.labels(job_type=job_type, status='submitted').inc()
    # Update queue size (approximate - we'll need to get actual queue size)
    # This will be updated separately by a periodic task

def record_job_started(job_type: str, queue_name: str = 'default'):
    """
    Record a job started.
    
    Args:
        job_type: Type of job
        queue_name: Name of the queue
    """
    JOBS_TOTAL.labels(job_type=job_type, status='started').inc()
    JOBS_ACTIVE.labels(queue_name=queue_name).inc()

def record_job_finished(job_type: str, queue_name: str = 'default'):
    """
    Record a job finished.
    
    Args:
        job_type: Type of job
        queue_name: Name of the queue
    """
    JOBS_TOTAL.labels(job_type=job_type, status='finished').inc()
    JOBS_ACTIVE.labels(queue_name=queue_name).dec()

def record_job_failed(job_type: str, queue_name: str = 'default'):
    """
    Record a job failed.
    
    Args:
        job_type: Type of job
        queue_name: Name of the queue
    """
    JOBS_TOTAL.labels(job_type=job_type, status='failed').inc()
    JOBS_ACTIVE.labels(queue_name=queue_name).dec()

def update_queue_metrics(queue_name: str, size: int):
    """
    Update queue size metric.
    
    Args:
        queue_name: Name of the queue
        size: Current queue size
    """
    JOBS_QUEUE_SIZE.labels(queue_name=queue_name).set(size)

def update_queue_stats():
    """
    Update queue statistics from Redis.
    
    This function queries Redis for current queue sizes and active jobs
    and updates the corresponding metrics.
    """
    try:
        from apvbt.api_fastapi.jobs import get_redis_connection, get_queue
        redis_conn = get_redis_connection()
        if redis_conn is None:
            return
        
        # Get default queue
        queue = get_queue('default')
        if queue is None:
            return
        
        # Get queue size (number of pending jobs)
        try:
            # Try to use queue.count property
            queue_size = queue.count
        except AttributeError:
            # Fall back to counting jobs in queue
            queue_size = len(queue.get_jobs())
        update_queue_metrics('default', queue_size)
        
        # Get active jobs (jobs that are started but not finished)
        # This requires checking worker registrations; for simplicity we can
        # use RQ's registry sizes.
        # We'll implement this later if needed.
        
    except ImportError:
        logger.debug("RQ not available, skipping queue stats update")
    except Exception as e:
        logger.debug(f"Failed to update queue stats: {e}")

def get_metrics():
    """
    Get all metrics in Prometheus text format.
    
    Returns:
        Metrics as a string
    """
    # Update queue stats before returning metrics
    update_queue_stats()
    return generate_latest(registry)