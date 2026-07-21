"""
Job queue module for asynchronous inference operations.

This module provides job definitions and utilities for running inference
asynchronously using Redis Queue (RQ).
"""

import os
import sys
import json
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import Redis Queue (RQ) components
try:
    from redis import Redis  # type: ignore
    from rq import Queue, Worker  # type: ignore
    from redis.connection import Connection  # type: ignore
    from rq.job import Job  # type: ignore
    from rq.exceptions import NoSuchJobError  # type: ignore
    RQ_AVAILABLE = True
except ImportError:
    RQ_AVAILABLE = False
    # Create dummy classes for type hints
    class Redis:
        def __init__(self, *args, **kwargs):
            pass
        def ping(self):
            pass
    class Queue:
        def __init__(self, *args, **kwargs):
            pass
        def enqueue(self, *args, **kwargs):
            pass
        def get_jobs(self, *args, **kwargs):
            return []
        @property
        def count(self):
            return 0
    class Job:
        def __init__(self):
            self.is_failed = False
            self.is_finished = False
            self.is_started = False
            self.created_at = None
            self.started_at = None
            self.ended_at = None
            self.description = ""
            self.exc_info = None
            self.result = None
        
        @classmethod
        def fetch(cls, *args, **kwargs):
            return cls()
    class NoSuchJobError(Exception):
        pass

from apvbt.api.models import (
    InferenceRequest, 
    InferenceResponse, 
    InferenceMetadata,
    BatchInferenceRequest,
    BatchInferenceResponse,
    BatchInferenceMetadata
)
from apvbt import __version__
from apvbt.api_fastapi.metrics import (
    record_job_submitted,
    record_job_started,
    record_job_finished,
    record_job_failed,
)

# Configure logging
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the queue."""
    PENDING = "pending"
    STARTED = "started"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass
class JobMetadataDataclass:
    """Metadata for a job in the queue (dataclass version)."""
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    job_type: str = "inference"
    model_id: Optional[str] = None
    error_message: Optional[str] = None
    result_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobMetadataDataclass':
        """Create from dictionary."""
        return cls(**data)


def get_redis_connection() -> Optional[Redis]:
    """Get Redis connection from environment variables."""
    if not RQ_AVAILABLE:
        logger.warning("RQ not available. Job queue functionality disabled.")
        return None
    
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    redis_password = os.environ.get('REDIS_PASSWORD')
    
    try:
        if redis_password:
            redis_conn = Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True
            )
        else:
            redis_conn = Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
        # Test connection
        redis_conn.ping()
        logger.debug(f"Redis connection established to {redis_host}:{redis_port}")
        return redis_conn
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return None


def get_queue(queue_name: str = 'default') -> Optional[Queue]:
    """Get Redis Queue instance."""
    redis_conn = get_redis_connection()
    if redis_conn is None:
        return None
    return Queue(queue_name, connection=redis_conn)


def submit_inference_job(
    inference_request: InferenceRequest,
    job_id: Optional[str] = None,
    queue_name: str = 'default'
) -> Optional[str]:
    """
    Submit an inference job to the queue.
    
    Args:
        inference_request: Inference request parameters
        job_id: Optional custom job ID (generated if not provided)
        queue_name: Queue name (default: 'default')
        
    Returns:
        Job ID if successful, None otherwise
    """
    if not RQ_AVAILABLE:
        logger.error("Cannot submit job: RQ not available")
        return None
    
    queue = get_queue(queue_name)
    if queue is None:
        logger.error("Cannot submit job: Redis connection failed")
        return None
    
    # Create job metadata
    if job_id is None:
        import uuid
        job_id = str(uuid.uuid4())
    
    # Submit job
    try:
        job = queue.enqueue(
            run_inference_job,
            inference_request.model_dump(),
            job_id=job_id,
            job_timeout=300,  # 5 minutes timeout
            result_ttl=86400  # Keep results for 24 hours
        )
        logger.info(f"Inference job submitted: {job_id} for model {inference_request.model_id}")
        # Record metrics
        try:
            record_job_submitted('inference', queue_name)
        except Exception as e:
            logger.debug(f"Failed to record job submission metrics: {e}")
        return job_id
    except Exception as e:
        logger.error(f"Failed to submit inference job: {e}")
        return None


def submit_batch_inference_job(
    batch_request: BatchInferenceRequest,
    job_id: Optional[str] = None,
    queue_name: str = 'default'
) -> Optional[str]:
    """
    Submit a batch inference job to the queue.
    
    Args:
        batch_request: Batch inference request parameters
        job_id: Optional custom job ID (generated if not provided)
        queue_name: Queue name (default: 'default')
        
    Returns:
        Job ID if successful, None otherwise
    """
    if not RQ_AVAILABLE:
        logger.error("Cannot submit batch job: RQ not available")
        return None
    
    queue = get_queue(queue_name)
    if queue is None:
        logger.error("Cannot submit batch job: Redis connection failed")
        return None
    
    # Create job metadata
    if job_id is None:
        import uuid
        job_id = str(uuid.uuid4())
    
    # Submit job
    try:
        job = queue.enqueue(
            run_batch_inference_job,
            batch_request.model_dump(),
            job_id=job_id,
            job_timeout=1800,  # 30 minutes timeout for batch jobs
            result_ttl=86400  # Keep results for 24 hours
        )
        logger.info(f"Batch inference job submitted: {job_id} for model {batch_request.model_id} (batch size: {len(batch_request.features_list)})")
        # Record metrics
        try:
            record_job_submitted('batch_inference', queue_name)
        except Exception as e:
            logger.debug(f"Failed to record job submission metrics: {e}")
        return job_id
    except Exception as e:
        logger.error(f"Failed to submit batch inference job: {e}")
        return None


def run_inference_job(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference job (worker function).
    
    This function is executed by RQ workers.
    
    Args:
        request_data: InferenceRequest as dictionary
        
    Returns:
        InferenceResponse as dictionary
    """
    from apvbt.api_fastapi.server import create_app
    
    logger.info(f"Starting inference job for model {request_data.get('model_id')}")
    start_time = time.time()
    
    # Record job started
    try:
        record_job_started('inference', 'default')
    except Exception as e:
        logger.debug(f"Failed to record job started metrics: {e}")
    
    try:
        # Create a minimal app context to access posterior models
        # In production, we'd want to load models once and reuse
        app = create_app()
        
        # Convert request data to InferenceRequest for validation
        inference_request = InferenceRequest(**request_data)
        
        # Get posterior model (simplified - in real implementation we'd load from file)
        # For now, we'll reuse the same logic as in the server
        if not hasattr(app.state, 'posteriors') or app.state.posteriors is None:
            raise ValueError("No posterior models loaded")
        
        model_id = inference_request.model_id
        if model_id not in app.state.posteriors:
            raise ValueError(f"Model '{model_id}' not found")
        
        posterior = app.state.posteriors[model_id]
        
        features_array = np.array(inference_request.features)
        
        if inference_request.seed is not None:
            np.random.seed(inference_request.seed)
        
        samples = posterior.sample(
            shape=(inference_request.num_samples,),
            x=features_array,
            show_progress_bars=False,
        )
        
        samples_list = samples.tolist()
        mean = np.mean(samples, axis=0).tolist()
        std = np.std(samples, axis=0).tolist()
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        metadata = InferenceMetadata(
            model_id=model_id,
            num_samples=inference_request.num_samples,
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=inference_request.seed,
        )
        
        response = InferenceResponse(
            samples=samples_list,
            mean=mean,
            std=std,
            metadata=metadata,
        )
        
        logger.info(f"Inference job completed for model {model_id}, {inference_request.num_samples} samples, {computation_time_ms:.2f}ms")
        
        # Record job finished
        try:
            record_job_finished('inference', 'default')
        except Exception as e:
            logger.debug(f"Failed to record job finished metrics: {e}")
        
        return response.model_dump()
        
    except Exception as e:
        logger.error(f"Inference job failed: {e}")
        # Record job failed
        try:
            record_job_failed('inference', 'default')
        except Exception as metric_error:
            logger.debug(f"Failed to record job failed metrics: {metric_error}")
        raise  # RQ will mark the job as failed


def run_batch_inference_job(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run batch inference job (worker function).
    
    This function is executed by RQ workers.
    
    Args:
        request_data: BatchInferenceRequest as dictionary
        
    Returns:
        BatchInferenceResponse as dictionary
    """
    from apvbt.api_fastapi.server import create_app
    
    logger.info(f"Starting batch inference job for model {request_data.get('model_id')}")
    start_time = time.time()
    
    # Record job started
    try:
        record_job_started('batch_inference', 'default')
    except Exception as e:
        logger.debug(f"Failed to record job started metrics: {e}")
    
    try:
        # Create a minimal app context to access posterior models
        app = create_app()
        
        # Convert request data to BatchInferenceRequest for validation
        batch_request = BatchInferenceRequest(**request_data)
        
        # Get posterior model
        if not hasattr(app.state, 'posteriors') or app.state.posteriors is None:
            raise ValueError("No posterior models loaded")
        
        model_id = batch_request.model_id
        if model_id not in app.state.posteriors:
            raise ValueError(f"Model '{model_id}' not found")
        
        posterior = app.state.posteriors[model_id]
        
        # Validate batch size
        batch_size = len(batch_request.features_list)
        if batch_request.max_batch_size and batch_size > batch_request.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum allowed {batch_request.max_batch_size}")
        
        results = []
        total_computation_time_ms = 0
        
        if batch_request.seed is not None:
            np.random.seed(batch_request.seed)
        
        for i, features in enumerate(batch_request.features_list):
            iteration_start = time.time()
            features_array = np.array(features)
            
            samples = posterior.sample(
                shape=(batch_request.num_samples,),
                x=features_array,
                show_progress_bars=False,
            )
            
            samples_list = samples.tolist()
            mean = np.mean(samples, axis=0).tolist()
            std = np.std(samples, axis=0).tolist()
            
            iteration_time_ms = (time.time() - iteration_start) * 1000
            total_computation_time_ms += iteration_time_ms
            
            metadata = InferenceMetadata(
                model_id=model_id,
                num_samples=batch_request.num_samples,
                computation_time_ms=iteration_time_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                seed=batch_request.seed,
            )
            
            result = InferenceResponse(
                samples=samples_list,
                mean=mean,
                std=std,
                metadata=metadata,
            )
            results.append(result.model_dump())
        
        batch_metadata = BatchInferenceMetadata(
            model_id=model_id,
            batch_size=batch_size,
            num_samples=batch_request.num_samples,
            total_computation_time_ms=total_computation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=batch_request.seed,
        )
        
        response = BatchInferenceResponse(
            results=results,
            metadata=batch_metadata,
        )
        
        logger.info(f"Batch inference job completed for model {model_id}, batch size {batch_size}, {batch_request.num_samples} samples per vector, total time {total_computation_time_ms:.2f}ms")
        
        # Record job finished
        try:
            record_job_finished('batch_inference', 'default')
        except Exception as e:
            logger.debug(f"Failed to record job finished metrics: {e}")
        
        return response.model_dump()
        
    except Exception as e:
        logger.error(f"Batch inference job failed: {e}")
        # Record job failed
        try:
            record_job_failed('batch_inference', 'default')
        except Exception as metric_error:
            logger.debug(f"Failed to record job failed metrics: {metric_error}")
        raise  # RQ will mark the job as failed


def get_job_status(job_id: str) -> Optional[JobMetadataDataclass]:
    """
    Get status of a job.
    
    Args:
        job_id: Job ID
        
    Returns:
        JobMetadata if job exists, None otherwise
    """
    if not RQ_AVAILABLE:
        return None
    
    redis_conn = get_redis_connection()
    if redis_conn is None:
        return None
    
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        # Determine status
        if job.is_failed:
            status = JobStatus.FAILED
            error_message = job.exc_info
        elif job.is_finished:
            status = JobStatus.FINISHED
        elif job.is_started:
            status = JobStatus.STARTED
        else:
            status = JobStatus.PENDING
        
        # Create metadata
        metadata = JobMetadataDataclass(
            job_id=job_id,
            status=status,
            created_at=job.created_at.isoformat() if job.created_at else datetime.now(timezone.utc).isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            finished_at=job.ended_at.isoformat() if job.ended_at else None,
            job_type=job.description or "inference",
            error_message=job.exc_info if job.is_failed else None,
            result_url=f"/api/v1/jobs/{job_id}/result" if job.is_finished else None
        )
        
        return metadata
        
    except NoSuchJobError:
        logger.warning(f"Job not found: {job_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching job status: {e}")
        return None


def get_job_result(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get result of a finished job.
    
    Args:
        job_id: Job ID
        
    Returns:
        Job result if job exists and is finished, None otherwise
    """
    if not RQ_AVAILABLE:
        return None
    
    redis_conn = get_redis_connection()
    if redis_conn is None:
        return None
    
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if not job.is_finished:
            logger.warning(f"Job {job_id} is not finished (status: {job.get_status()})")
            return None
        
        if job.is_failed:
            logger.warning(f"Job {job_id} failed: {job.exc_info}")
            return {
                "error": "Job failed",
                "message": job.exc_info
            }
        
        return job.result
        
    except NoSuchJobError:
        logger.warning(f"Job not found: {job_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching job result: {e}")
        return None


def list_jobs(queue_name: str = 'default', limit: int = 100) -> List[JobMetadataDataclass]:
    """
    List jobs in a queue.
    
    Args:
        queue_name: Queue name
        limit: Maximum number of jobs to return
        
    Returns:
        List of JobMetadataDataclass
    """
    if not RQ_AVAILABLE:
        return []
    
    queue = get_queue(queue_name)
    if queue is None:
        return []
    
    try:
        jobs = queue.get_jobs(offset=0, limit=limit)
        metadata_list = []
        
        for job in jobs:
            if job.is_failed:
                status = JobStatus.FAILED
            elif job.is_finished:
                status = JobStatus.FINISHED
            elif job.is_started:
                status = JobStatus.STARTED
            else:
                status = JobStatus.PENDING
            
            metadata = JobMetadataDataclass(
                job_id=job.id,
                status=status,
                created_at=job.created_at.isoformat() if job.created_at else datetime.now(timezone.utc).isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                finished_at=job.ended_at.isoformat() if job.ended_at else None,
                job_type=job.description or "inference",
                error_message=job.exc_info if job.is_failed else None,
                result_url=f"/api/v1/jobs/{job.id}/result" if job.is_finished else None
            )
            metadata_list.append(metadata)
        
        return metadata_list
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return []