"""
FastAPI server for APVBT REST API.

This module provides a FastAPI-based implementation of the APVBT API
with async support and improved performance.
"""

import sys
import argparse
import logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from apvbt import __version__
import time
import numpy as np
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError
from apvbt.api.errors import ValidationError, ModelNotFoundError, InferenceError
from apvbt.api.models import InferenceRequest, InferenceResponse, InferenceMetadata, ModelInfo, BatchInferenceRequest, BatchInferenceResponse, BatchInferenceMetadata, JobMetadata, JobSubmitResponse, JobsListResponse, JobStatus, LatentSpaceRequest, LatentSpaceResponse, RegimeValidationRequest, RegimeValidationResponse, BenchmarkParcsRequest, BenchmarkParcsResponse, ParcMetrics, CompareAlgosRequest, CompareAlgosResponse, AlgoDiagnostics, FeatureCovarianceRequest, FeatureCovarianceResponse
from prometheus_fastapi_instrumentator import Instrumentator
from apvbt.api_fastapi.metrics import record_inference_request, update_queue_stats


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting - created per app instance


def create_app(config: Optional[dict] = None) -> FastAPI:
    """
    Create and configure a FastAPI application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="APVBT API",
        description="Amortizing Personalization in Virtual Brain Twins - REST API",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Store configuration
    app.state.config = config or {}
    app.state.posteriors = {}
    
    # Default configuration for authentication
    app.state.config.setdefault("API_KEY_ENABLED", False)
    app.state.config.setdefault("API_KEY", None)
    app.state.config.setdefault("API_KEYS", [])
    
    # Default configuration for rate limiting
    app.state.config.setdefault("RATE_LIMIT_ENABLED", False)
    app.state.config.setdefault("RATE_LIMIT_PER_MINUTE", 60)  # requests per minute
    app.state.config.setdefault("RATE_LIMIT_PER_HOUR", 3600)  # requests per hour
    
    # Log authentication status
    if app.state.config["API_KEY_ENABLED"]:
        if app.state.config.get("API_KEY") or app.state.config.get("API_KEYS"):
            logger.info("API key authentication enabled (FastAPI)")
        else:
            logger.warning("API key authentication enabled but no API keys configured (FastAPI)")
    else:
        logger.info("API key authentication disabled (FastAPI)")
    
    # Dependency for API key authentication
    async def authenticate_api_key(
        request: Request,
        api_key: Optional[str] = Depends(api_key_header)
    ):
        """Dependency to authenticate API key."""
        # Skip authentication if disabled
        if not request.app.state.config.get("API_KEY_ENABLED", False):
            return
        
        # Get valid keys
        valid_keys = request.app.state.config.get("API_KEYS", [])
        single_key = request.app.state.config.get("API_KEY")
        if single_key:
            valid_keys = [single_key]
        
        # If no keys configured, allow all requests
        if not valid_keys:
            return
        
        # Require API key
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key is required. Provide X-API-Key header.",
            )
        
        # Validate API key
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key for {request.method} {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )
        
        # Authentication successful
        logger.debug(f"API key authentication successful for {request.method} {request.url.path}")
    
    # Store authentication dependency
    app.state.authenticate = authenticate_api_key
    
    # Configure rate limiting
    limiter = Limiter(key_func=get_remote_address)
    limiter.app = app
    app.state.limiter = limiter
    
    # Add rate limit exceeded error handler
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Configure Prometheus metrics
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        logger.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        logger.info("Prometheus metrics disabled (prometheus-fastapi-instrumentator not installed)")
    
    # Register routes
    register_routes(app)
    
    logger.info(f"APVBT FastAPI v{__version__} created")
    return app


def register_routes(app: FastAPI):
    """Register all routes with the FastAPI application."""
    logger.info("Registering routes...")
    
    # Helper to apply rate limiting conditionally
    def rate_limit_decorator(limit_value: str, scope: str | None = None):
        """Return a decorator that applies rate limiting if enabled."""
        logger.debug(f"rate_limit_decorator called with limit={limit_value}, scope={scope}")
        if app.state.config.get("RATE_LIMIT_ENABLED", False):
            # Support config-based limits: "per_minute" uses RATE_LIMIT_PER_MINUTE
            # "per_hour" uses RATE_LIMIT_PER_HOUR
            if limit_value == "per_minute":
                limit = f"{app.state.config.get('RATE_LIMIT_PER_MINUTE', 60)}/minute"
            elif limit_value == "per_hour":
                limit = f"{app.state.config.get('RATE_LIMIT_PER_HOUR', 3600)}/hour"
            else:
                limit = limit_value
            logger.debug(f"Applying rate limit {limit} to endpoint with scope={scope}")
            if scope:
                return app.state.limiter.shared_limit(limit, scope=scope)
            else:
                return app.state.limiter.limit(limit)
        # Return a no-op decorator
        def no_op_decorator(func):
            logger.debug(f"No-op decorator applied to {func.__name__}")
            return func
        return no_op_decorator
    
    # Helper to check job queue availability
    def check_job_queue_available():
        """Check if job queue functionality is available."""
        try:
            from apvbt.api_fastapi import jobs
            if not jobs.RQ_AVAILABLE:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Job queue functionality not available (RQ not installed)"
                )
            redis_conn = jobs.get_redis_connection()
            if redis_conn is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Redis connection failed. Job queue unavailable."
                )
            return jobs
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Job queue module not available: {e}"
            )
    
    # Helper to convert dataclass to Pydantic model
    def convert_job_metadata(dc):
        '''Convert JobMetadataDataclass to Pydantic JobMetadata.'''
        return JobMetadata(
            job_id=dc.job_id,
            status=dc.status,
            created_at=dc.created_at,
            started_at=dc.started_at,
            finished_at=dc.finished_at,
            job_type=dc.job_type,
            model_id=dc.model_id,
            error_message=dc.error_message,
            result_url=dc.result_url
        )
    
    @app.get("/health", tags=["health"])
    @rate_limit_decorator("per_minute", scope="health")
    async def health_check(request: Request):
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "apvbt-api",
            "version": __version__,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    @app.get("/api/v1/models", tags=["models"], dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="models")
    async def list_models(request: Request):
        """List all available dynamics models."""
        from apvbt.dynamics.models import ModelRegistry
        try:
            registered_models = ModelRegistry.list_available()
            models_info = []
            for model_name in registered_models:
                metadata = ModelRegistry.get_metadata(model_name)
                model_info = ModelInfo(
                    model_id=metadata.name,
                    name=metadata.name,
                    description=metadata.description,
                    version=metadata.version,
                    parameters=metadata.parameters,
                    loaded=False,
                )
                models_info.append(model_info)
            return {
                "models": [m.model_dump() for m in models_info],
                "count": len(models_info)
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list models"
            )
    
    @app.get("/api/v1/models/{model_id}", tags=["models"], dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="models")
    async def get_model(model_id: str, request: Request):
        """Get details for a specific model."""
        from apvbt.dynamics.models import ModelRegistry
        try:
            if not ModelRegistry.is_registered(model_id):
                raise ModelNotFoundError(model_id)

            metadata = ModelRegistry.get_metadata(model_id)
            model_info = ModelInfo(
                model_id=metadata.name,
                name=metadata.name,
                description=metadata.description,
                version=metadata.version,
                parameters=metadata.parameters,
                loaded=False,
            )
            return model_info
        except ModelNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=e.message
            )
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get model details for {model_id}"
            )
    
    @app.post("/api/v1/infer", tags=["inference"], response_model=InferenceResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="infer")
    async def infer(inference_request: InferenceRequest, request: Request):
        """Run inference on observed features."""
        start_time = time.time()

        if not hasattr(request.app.state, "posteriors") or request.app.state.posteriors is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No posterior models loaded. Use /models endpoint to check available models."
            )

        model_id = inference_request.model_id

        if model_id not in request.app.state.posteriors:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found"
            )

        posterior = request.app.state.posteriors[model_id]

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

        logger.info(
            f"Inference completed for model {model_id}, {inference_request.num_samples} samples, {computation_time_ms:.2f}ms"
        )
        
        # Record metrics
        try:
            duration_seconds = computation_time_ms / 1000.0
            record_inference_request(model_id, 'single', duration_seconds)
        except Exception as e:
            logger.debug(f"Failed to record inference metrics: {e}")

        return response
    
    @app.post("/api/v1/infer/batch", tags=["inference"], response_model=BatchInferenceResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="batch")
    async def batch_infer(batch_request: BatchInferenceRequest, request: Request):
        """Run batch inference on multiple observed feature vectors."""
        start_time = time.time()

        if not hasattr(request.app.state, "posteriors") or request.app.state.posteriors is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No posterior models loaded. Use /models endpoint to check available models."
            )

        model_id = batch_request.model_id

        if model_id not in request.app.state.posteriors:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found"
            )

        posterior = request.app.state.posteriors[model_id]

        # Validate batch size
        batch_size = len(batch_request.features_list)
        if batch_request.max_batch_size and batch_size > batch_request.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {batch_size} exceeds maximum allowed {batch_request.max_batch_size}"
            )

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
            results.append(result)

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

        logger.info(
            f"Batch inference completed for model {model_id}, batch size {batch_size}, {batch_request.num_samples} samples per vector, total time {total_computation_time_ms:.2f}ms"
        )
        
        # Record metrics
        try:
            duration_seconds = total_computation_time_ms / 1000.0
            record_inference_request(model_id, 'batch', duration_seconds)
        except Exception as e:
            logger.debug(f"Failed to record inference metrics: {e}")

        return response
    
    # Job queue endpoints
    @app.post("/api/v1/jobs/infer", tags=["jobs"], response_model=JobSubmitResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="jobs")
    async def submit_inference_job(inference_request: InferenceRequest, request: Request):
        """Submit an inference job to the queue."""
        jobs = check_job_queue_available()
        job_id = jobs.submit_inference_job(inference_request)
        if job_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit job to queue"
            )
        
        # Create response
        return JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            job_url=f"/api/v1/jobs/{job_id}",
            result_url=f"/api/v1/jobs/{job_id}/result"
        )
    
    @app.post("/api/v1/jobs/infer/batch", tags=["jobs"], response_model=JobSubmitResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="jobs")
    async def submit_batch_inference_job(batch_request: BatchInferenceRequest, request: Request):
        """Submit a batch inference job to the queue."""
        jobs = check_job_queue_available()
        job_id = jobs.submit_batch_inference_job(batch_request)
        if job_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to submit batch job to queue"
            )
        
        # Create response
        return JobSubmitResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            job_url=f"/api/v1/jobs/{job_id}",
            result_url=f"/api/v1/jobs/{job_id}/result"
        )
    
    @app.get("/api/v1/jobs", tags=["jobs"], response_model=JobsListResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="jobs")
    async def list_jobs(request: Request, queue_name: str = 'default', limit: int = 100):
        """List jobs in a queue."""
        jobs = check_job_queue_available()
        job_list = jobs.list_jobs(queue_name=queue_name, limit=limit)
        converted_jobs = [convert_job_metadata(job) for job in job_list]
        return JobsListResponse(
            jobs=converted_jobs,
            count=len(converted_jobs),
            queue_name=queue_name
        )
    
    @app.get("/api/v1/jobs/{job_id}", tags=["jobs"], response_model=JobMetadata, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="jobs")
    async def get_job_status(job_id: str, request: Request):
        """Get status of a job."""
        jobs = check_job_queue_available()
        job_metadata = jobs.get_job_status(job_id)
        if job_metadata is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found"
            )
        return convert_job_metadata(job_metadata)
    
    @app.get("/api/v1/jobs/{job_id}/result", tags=["jobs"], dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="jobs")
    async def get_job_result(job_id: str, request: Request):
        """Get result of a finished job."""
        jobs = check_job_queue_available()
        result = jobs.get_job_result(job_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job '{job_id}' not found or not finished"
            )
        return result
    
    # Diagnostics endpoints
    @app.post("/api/v1/diagnostics/latent", tags=["diagnostics"], response_model=LatentSpaceResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="diagnostics")
    async def evaluate_latent_space(latent_request: LatentSpaceRequest, request: Request):
        """Evaluate crosscoder latent space quality."""
        from apvbt.data import XCode
        from apvbt.diagnostics import evaluate_latent_space_quality
        import os
        
        if not os.path.exists(latent_request.data_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data file '{latent_request.data_file}' not found"
            )
        
        try:
            xc = XCode.from_pkl(latent_request.data_file)
            result = evaluate_latent_space_quality(xc, arch=latent_request.arch, tts=latent_request.tts)
            
            return LatentSpaceResponse(
                latent_variance=result['latent_variance'].tolist(),
                latent_explained_ratio=result['latent_explained_ratio'],
                reconstruction_error_train=result['reconstruction_error_train'],
                reconstruction_error_test=result['reconstruction_error_test'],
                generalization_gap=result['generalization_gap'],
                optimal_dimensionality=result['optimal_dimensionality'],
                subject_separability=result['subject_separability'],
                quality_score=result['quality_score'],
                architecture=result['architecture'],
                n_parcellations=result['n_parcellations']
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error evaluating latent space: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to evaluate latent space: {str(e)}"
            )
    
    @app.post("/api/v1/diagnostics/regime", tags=["diagnostics"], response_model=RegimeValidationResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="diagnostics")
    async def validate_regime_endpoint(regime_request: RegimeValidationRequest, request: Request):
        """Validate regime selection for SBI."""
        from apvbt.diagnostics import validate_regime_selection
        
        params = np.array(regime_request.params)
        features = np.array(regime_request.features)
        
        if params.shape[0] != features.shape[0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"params and features must have same number of samples: {params.shape[0]} vs {features.shape[0]}"
            )
        
        try:
            result = validate_regime_selection(params, features, method=regime_request.method)
            
            return RegimeValidationResponse(
                is_valid=result['is_valid'],
                metric_value=result['metric_value'],
                recommendation=result['recommendation'],
                confidence=result['confidence'],
                method=result['method'],
                n_samples=result['n_samples'],
                n_params=result['n_params'],
                n_features=result['n_features']
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error validating regime: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to validate regime: {str(e)}"
            )
    
    # Benchmark endpoints
    @app.post("/api/v1/benchmark/parcellations", tags=["benchmark"], response_model=BenchmarkParcsResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_hour", scope="benchmark")
    async def benchmark_parcellations(bench_request: BenchmarkParcsRequest, request: Request):
        """Run multi-parcellation benchmarking."""
        import os
        
        if not os.path.exists(bench_request.data_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data file '{bench_request.data_file}' not found"
            )
        
        try:
            from apvbt.data import XCode
            from apvbt import crosscoder
            from apvbt.benchmarking import bench_model_multi_parc, aggregate_parc_results
            from apvbt.dynamics import DynaModel
            from apvbt.dynamics.hopf import hopf_dfun
            import jax.numpy as jp
            
            xc = XCode.from_pkl(bench_request.data_file)
            
            parcs = bench_request.parcellations
            if parcs is None:
                parcs = [p for p in xc.parcs if p != '031-MIST']
            
            if not parcs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No parcellations found in data file"
                )
            
            nreg = int(parcs[0].split('-')[0])
            ti, tj = jp.triu_indices(nreg, k=1)
            def features(x):
                return jp.corrcoef(x[500:, 0].T)[ti, tj]
            model = DynaModel('hopf', hopf_dfun, features, dt=0.02)
            
            results = bench_model_multi_parc(
                xc, model, parcs, arch=bench_request.architecture,
                num_batch=bench_request.num_batch, batch_size=bench_request.batch_size
            )
            
            aggregated = aggregate_parc_results(
                results, 
                metrics=['ok_percent', 'median_shrinkage', 'median_z']
            )
            
            results_dict = {}
            for parc, r in results.items():
                results_dict[parc] = ParcMetrics(
                    ok_percent=r['metrics']['ok_percent'],
                    median_shrinkage=r['metrics']['median_shrinkage'],
                    median_z=r['metrics']['median_z'],
                    mean_shrinkage=r['metrics']['mean_shrinkage'],
                    mean_z=r['metrics']['mean_z'],
                    ci90_coverage=r['metrics']['ci90_coverage']
                )
            
            return BenchmarkParcsResponse(
                results=results_dict,
                means={k: float(v) for k, v in aggregated['means'].items()},
                best_parc=aggregated['best_parc'],
                worst_parc=aggregated['worst_parc'],
                n_parcellations=len(parcs),
                architecture=bench_request.architecture
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to run benchmark: {str(e)}"
            )
    
    @app.post("/api/v1/benchmark/algorithms", tags=["benchmark"], response_model=CompareAlgosResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_hour", scope="benchmark")
    async def compare_algorithms(compare_request: CompareAlgosRequest, request: Request):
        """Compare SBI algorithms (MAF vs MDN)."""
        import os
        
        if not os.path.exists(compare_request.samples_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Samples file '{compare_request.samples_file}' not found"
            )
        
        try:
            from apvbt.benchmarking import bench_model_multi_algo
            
            npz = np.load(compare_request.samples_file, allow_pickle=True)
            theta = npz['theta']
            feats = npz['feats']
            theta = theta.reshape(-1, theta.shape[-1])
            feats = feats.reshape(-1, feats.shape[-1])
            
            results = bench_model_multi_algo(
                theta, feats,
                algos=compare_request.algorithms,
                num_post_samples=compare_request.num_post_samples
            )
            
            diagnostics_dict = {}
            for algo, diags in results['diagnostics'].items():
                diagnostics_dict[algo] = AlgoDiagnostics(
                    ok_percent=diags['ok_percent'],
                    median_shrinkage=diags['median_shrinkage'],
                    median_z=diags['median_z'],
                    mean_shrinkage=diags['mean_shrinkage'],
                    mean_z=diags['mean_z']
                )
            
            return CompareAlgosResponse(
                algorithms=results['algorithms'],
                diagnostics=diagnostics_dict,
                n_samples=theta.shape[0],
                n_params=theta.shape[1],
                n_features=feats.shape[1]
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error comparing algorithms: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to compare algorithms: {str(e)}"
            )
    
    @app.post("/api/v1/diagnostics/features", tags=["diagnostics"], response_model=FeatureCovarianceResponse, dependencies=[Depends(app.state.authenticate)])
    @rate_limit_decorator("per_minute", scope="diagnostics")
    async def analyze_features(feature_request: FeatureCovarianceRequest, request: Request):
        """Analyze feature covariance structure."""
        from apvbt.diagnostics import compute_feature_covariance
        
        features = np.array(feature_request.features)
        params = np.array(feature_request.params) if feature_request.params else None
        
        try:
            result = compute_feature_covariance(
                features, 
                params=params,
                by_param=feature_request.by_param
            )
            
            eigenvalues = result['eigenvalues']
            eigenvalues_summary = {
                'min': float(np.min(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'mean': float(np.mean(eigenvalues))
            }
            
            return FeatureCovarianceResponse(
                mean_cov=result['mean_cov'],
                std_cov=result['std_cov'],
                condition_number=result['condition_number'],
                n_features=features.shape[1] if features.ndim > 1 else 1,
                n_samples=features.shape[0],
                eigenvalues_summary=eigenvalues_summary
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logger.error(f"Error analyzing features: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to analyze features: {str(e)}"
            )
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle generic exceptions."""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


def main():
    """Main entry point for FastAPI server."""
    parser = argparse.ArgumentParser(description="APVBT FastAPI REST Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (sets API_KEY_ENABLED=True)",
    )
    parser.add_argument(
        "--enable-api-key",
        action="store_true",
        default=False,
        help="Enable API key authentication (requires --api-key or API_KEY environment variable)",
    )
    parser.add_argument(
        "--enable-rate-limit",
        action="store_true",
        default=False,
        help="Enable rate limiting (default limits: 60/minute, 3600/hour)",
    )
    parser.add_argument(
        "--rate-limit-per-minute",
        type=int,
        default=60,
        help="Maximum requests per minute (applies when rate limiting is enabled)",
    )
    parser.add_argument(
        "--rate-limit-per-hour",
        type=int,
        default=3600,
        help="Maximum requests per hour (applies when rate limiting is enabled)",
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {"LOG_LEVEL": args.log_level}
    
    # API key authentication
    if args.api_key:
        config["API_KEY"] = args.api_key
        config["API_KEY_ENABLED"] = True
    elif args.enable_api_key:
        config["API_KEY_ENABLED"] = True
    
    # Rate limiting configuration
    if args.enable_rate_limit:
        config["RATE_LIMIT_ENABLED"] = True
        config["RATE_LIMIT_PER_MINUTE"] = args.rate_limit_per_minute
        config["RATE_LIMIT_PER_HOUR"] = args.rate_limit_per_hour
    
    app = create_app(config=config)
    
    print(f"Starting APVBT FastAPI v{__version__} on {args.host}:{args.port}")
    print(f"Log level: {args.log_level}")
    print(f"Auto-reload: {args.reload}")
    if config.get("API_KEY_ENABLED"):
        if args.api_key:
            print(f"API key authentication enabled (key: {args.api_key[:8]}...)")
        else:
            print("API key authentication enabled (no keys configured - allowing all requests)")
    else:
        print("API key authentication disabled")
    
    if config.get("RATE_LIMIT_ENABLED"):
        print(f"Rate limiting enabled: {config['RATE_LIMIT_PER_MINUTE']}/minute, {config['RATE_LIMIT_PER_HOUR']}/hour")
    else:
        print("Rate limiting disabled")
    
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()