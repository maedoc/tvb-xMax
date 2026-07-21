from typing import Optional
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
import logging
import traceback
from datetime import datetime

from apvbt.api.models import InferenceRequest, InferenceResponse, ModelInfo
from apvbt.api.errors import (
    ValidationError,
    ModelNotFoundError,
    InferenceError,
    ApvbtError,
)

__version__ = "2.0.0"


def create_app(config: Optional[dict] = None) -> Flask:
    app = Flask(__name__)

    if config:
        app.config.update(config)

    setup_logging(app)
    register_error_handlers(app)
    register_routes(app)

    return app


def setup_logging(app: Flask):
    log_level = app.config.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app.logger.setLevel(log_level)
    app.logger.info(f"APVBT API v{__version__} starting...")


def register_error_handlers(app: Flask):
    @app.errorhandler(ApvbtError)
    def handle_apvbt_error(error: ApvbtError):
        app.logger.error(f"{error.__class__.__name__}: {error.message}")
        response = {
            "error": error.__class__.__name__,
            "message": error.message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error.details:
            response["details"] = error.details
        return jsonify(response), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        app.logger.error(f"HTTP {error.code}: {error.description}")
        response = {
            "error": "HTTPError",
            "message": error.description,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return jsonify(response), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        app.logger.error(f"Unexpected error: {str(error)}\n{traceback.format_exc()}")
        response = {
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
        }
        return jsonify(response), 500


def register_routes(app: Flask):
    from apvbt.api.routes.health import bp as health_bp
    from apvbt.api.routes.models import bp as models_bp
    from apvbt.api.routes.inference import bp as inference_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(models_bp, url_prefix="/api/v1")
    app.register_blueprint(inference_bp, url_prefix="/api/v1")
