from flask import Blueprint, jsonify
from apvbt.api import __version__

bp = Blueprint("health", __name__)


@bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {"status": "ok", "version": __version__, "service": "apvbt-api"}
    ), 200
