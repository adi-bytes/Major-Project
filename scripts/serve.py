"""FastAPI server runner script for PG-SBD Designer."""

from __future__ import annotations

import sys

import uvicorn


def main() -> None:
    """Run the FastAPI server."""
    port = 8000
    host = "127.0.0.1"
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    if "--host" in sys.argv:
        idx = sys.argv.index("--host")
        if idx + 1 < len(sys.argv):
            host = sys.argv[idx + 1]

    print(f"Starting PG-SBD Designer API server on http://{host}:{port} ...")
    uvicorn.run("pgsbd.api.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
