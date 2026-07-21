from flask import Blueprint, jsonify, current_app
from apvbt.dynamics import ModelRegistry
from apvbt.api.models import ModelInfo

bp = Blueprint("models", __name__)


@bp.route("/models", methods=["GET"])
def list_models():
    """List all available dynamics models."""
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
        return jsonify(
            {"models": [m.model_dump() for m in models_info], "count": len(models_info)}
        ), 200
    except Exception as e:
        current_app.logger.error(f"Error listing models: {e}")
        return jsonify(
            {"error": "InternalServerError", "message": "Failed to list models"}
        ), 500


@bp.route("/models/<model_id>", methods=["GET"])
def get_model(model_id: str):
    """Get details for a specific model."""
    try:
        if not ModelRegistry.is_registered(model_id):
            from apvbt.api.errors import ModelNotFoundError

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
        return jsonify(model_info.model_dump()), 200
    except Exception as e:
        current_app.logger.error(f"Error getting model {model_id}: {e}")
        from apvbt.api.errors import ModelNotFoundError

        if isinstance(e, ModelNotFoundError):
            raise
        return jsonify(
            {
                "error": "InternalServerError",
                "message": f"Failed to get model details for {model_id}",
            }
        ), 500
