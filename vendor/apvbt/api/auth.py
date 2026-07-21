"""
Authentication utilities for APVBT API.

This module provides API key authentication for Flask endpoints.
"""

from typing import Optional, List
from functools import wraps
from flask import request, current_app, jsonify
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def require_api_key(f):
    """
    Decorator to require API key authentication for a route.
    
    Expects API key in X-API-Key header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if authentication is enabled
        if not current_app.config.get("API_KEY_ENABLED", False):
            return f(*args, **kwargs)
        
        # Get valid keys from configuration
        valid_keys = current_app.config.get("API_KEYS", [])
        single_key = current_app.config.get("API_KEY")
        
        if single_key:
            valid_keys = [single_key]
        
        # If no keys configured, allow all requests (development mode)
        if not valid_keys:
            logger.warning("No API keys configured, allowing request")
            return f(*args, **kwargs)
        
        # Now we have valid keys, so require API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            logger.warning("Missing API key in request")
            return jsonify({
                "error": "AuthenticationError",
                "message": "API key is required. Provide X-API-Key header.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 401
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key provided: {api_key[:8]}...")
            return jsonify({
                "error": "AuthenticationError",
                "message": "Invalid API key",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 403
        
        # API key is valid
        logger.debug(f"API key authentication successful for key: {api_key[:8]}...")
        return f(*args, **kwargs)
    
    return decorated_function


def setup_authentication(app):
    """
    Setup authentication for the Flask application.
    
    This function configures API key authentication and adds a before_request
    handler for protected routes.
    """
    # Configuration defaults
    app.config.setdefault("API_KEY_ENABLED", False)
    app.config.setdefault("API_KEY", None)
    app.config.setdefault("API_KEYS", [])
    
    # Log authentication status
    if app.config["API_KEY_ENABLED"]:
        if app.config.get("API_KEY") or app.config.get("API_KEYS"):
            logger.info("API key authentication enabled")
        else:
            logger.warning("API key authentication enabled but no API keys configured")
    else:
        logger.info("API key authentication disabled")
    
    # Apply authentication to all routes except health endpoint
    @app.before_request
    def authenticate():
        # Skip authentication for health endpoint
        if request.path == "/health" or request.path.startswith("/health/"):
            return
        
        # Skip if authentication disabled
        if not app.config.get("API_KEY_ENABLED", False):
            return
        
        # Get valid keys from configuration
        valid_keys = app.config.get("API_KEYS", [])
        single_key = app.config.get("API_KEY")
        
        if single_key:
            valid_keys = [single_key]
        
        # If no keys configured, allow all requests (development mode)
        if not valid_keys:
            return
        
        # Now we have valid keys, so require API key
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.path}")
            return jsonify({
                "error": "AuthenticationError",
                "message": "API key is required. Provide X-API-Key header.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 401
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key for {request.method} {request.path}")
            return jsonify({
                "error": "AuthenticationError",
                "message": "Invalid API key",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 403
        
        # API key is valid
        logger.debug(f"API key authentication successful for {request.method} {request.path}")