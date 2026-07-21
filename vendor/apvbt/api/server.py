import sys
import argparse
from flask import Flask
import logging

from apvbt.api import create_app
from apvbt import __version__


def main():
    parser = argparse.ArgumentParser(description="APVBT REST API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    app = create_app(config={"LOG_LEVEL": args.log_level})

    print(f"Starting APVBT API v{__version__} on {args.host}:{args.port}")
    print(f"Log level: {args.log_level}")
    print(f"Debug mode: {args.debug}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
