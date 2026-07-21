"""APVBT API module for running the REST server.

This module allows running the API server directly:
    python -m apvbt.api.server --host 0.0.0.0 --port 8080
"""

from apvbt.api.server import main

if __name__ == "__main__":
    main()
