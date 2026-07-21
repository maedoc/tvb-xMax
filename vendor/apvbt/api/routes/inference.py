from flask import Blueprint, jsonify, request, current_app
from typing import Dict
import logging
import time
import numpy as np
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError

from apvbt.api.models import InferenceRequest, InferenceResponse, InferenceMetadata
from apvbt.api.errors import ValidationError, ModelNotFoundError, InferenceError

bp = Blueprint("inference", __name__)
logger = logging.getLogger(__name__)


@bp.route("/infer", methods=["POST"])
def infer():
    """Run inference on observed features."""
    start_time = time.time()

    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(
            {
                "error": "ValidationError",
                "message": "Invalid JSON in request body",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 400

    try:
        if data is None:
            raise ValidationError("Request body must be JSON")

        req = InferenceRequest(**data)

        if not hasattr(current_app, "posteriors") or current_app.posteriors is None:
            raise InferenceError(
                "No posterior models loaded. Use /models endpoint to check available models."
            )

        model_id = req.model_id

        if model_id not in current_app.posteriors:
            raise ModelNotFoundError(model_id)

        posterior = current_app.posteriors[model_id]

        features_array = np.array(req.features)

        np.random.seed(req.seed) if req.seed is not None else None

        samples = posterior.sample(
            shape=(req.num_samples,),
            x=features_array,
            show_progress_bars=False,
        )

        samples_list = samples.tolist()

        mean = np.mean(samples, axis=0).tolist()
        std = np.std(samples, axis=0).tolist()

        computation_time_ms = (time.time() - start_time) * 1000

        metadata = InferenceMetadata(
            model_id=model_id,
            num_samples=req.num_samples,
            computation_time_ms=computation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=req.seed,
        )

        response = InferenceResponse(
            samples=samples_list,
            mean=mean,
            std=std,
            metadata=metadata,
        )

        logger.info(
            f"Inference completed for model {model_id}, {req.num_samples} samples, {computation_time_ms:.2f}ms"
        )

        return jsonify(response.model_dump()), 200

    except PydanticValidationError as e:
        current_app.logger.warning(f"Validation error: {e}")
        return jsonify(
            {
                "error": "ValidationError",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 400

    except ValidationError as e:
        current_app.logger.warning(f"Validation error: {e.message}")
        return jsonify(
            {
                "error": "ValidationError",
                "message": e.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 400

    except ModelNotFoundError as e:
        current_app.logger.warning(f"Model not found: {e.message}")
        return jsonify(
            {
                "error": "ModelNotFoundError",
                "message": e.message,
                "details": e.details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 404

    except Exception as e:
        current_app.logger.error(f"Inference error: {str(e)}")
        return jsonify(
            {
                "error": "InferenceError",
                "message": f"Inference failed: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ), 500
