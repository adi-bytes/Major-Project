"""FastAPI application entry point for PG-SBD Designer."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pgsbd.api.routes import router
from pgsbd.api.schemas import HONESTY_NOTICE
from pgsbd.device.sbd import BetaGa2O3SBD
from pgsbd.inverse.reverse import SBDReverseEngine
from pgsbd.ml.ensemble import EnsembleSurrogate

logger = logging.getLogger("pgsbd.api")


def _load_json_file(path: Path) -> dict | None:
    """Helper to safely read JSON artifacts synchronously."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model checkpoints and calibration artifacts on application startup."""
    logger.info("Initializing PG-SBD Designer API backend...")

    # 1. Device specification
    app.state.device_spec = BetaGa2O3SBD()

    # 2. Load calibration parameters
    calib_path = Path("artifacts/calib.json")
    app.state.calib = _load_json_file(calib_path)
    if app.state.calib is not None:
        logger.info("Loaded physics calibration from %s", calib_path)
    else:
        logger.warning("Calibration artifact %s not found.", calib_path)

    # 3. Load model ensemble
    ensemble_dir = Path("artifacts/ensemble")
    if (ensemble_dir / "model_0.pt").exists():
        ensemble = EnsembleSurrogate()
        ensemble.load_ensemble(ensemble_dir)
        app.state.ensemble = ensemble
        logger.info("Loaded 5 surrogate ensemble checkpoints from %s", ensemble_dir)
    else:
        app.state.ensemble = None
        logger.warning("Ensemble directory %s does not contain model_0.pt.", ensemble_dir)

    # 4. Initialize reverse mode engine
    if app.state.ensemble is not None:
        app.state.reverse_engine = SBDReverseEngine(
            ensemble=app.state.ensemble,
            calib_path=calib_path,
            dataset_path=Path("data/raw/new_dataset_GaO_WF+T.csv"),
            splits_path=Path("artifacts/splits.json"),
        )
        logger.info("Initialized Reverse Mode engine.")
    else:
        app.state.reverse_engine = None

    # 5. Load validation metrics
    metrics_path = Path("artifacts/metrics.json")
    app.state.metrics = _load_json_file(metrics_path)
    if app.state.metrics is not None:
        logger.info("Loaded validation metrics from %s", metrics_path)
    else:
        logger.warning("Metrics artifact %s not found.", metrics_path)

    yield

    logger.info("Shutting down PG-SBD Designer API backend...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title="PG-SBD Designer API",
        version="1.0.0",
        description=(
            "Physics-Guided Surrogate Modeling and Inverse Design Framework for "
            "Power Semiconductor Devices (Vertical Beta-Ga2O3 SBD).\n\n"
            f"**Disclaimer:** {HONESTY_NOTICE}"
        ),
        lifespan=lifespan,
    )

    # Configure CORS for Vite dev server and local clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "*",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom 422 validation error handler providing transparent messages
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(part) for part in err.get("loc", []))
            msg = err.get("msg", "Validation error")
            errors.append({"field": loc, "message": msg, "type": err.get("type")})

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": errors,
                "honesty_notice": HONESTY_NOTICE,
            },
        )

    # Mount API router
    application.include_router(router, prefix="/api", tags=["API"])

    # Serve the built React UI from the same port when web/dist exists (run: cd web && npm run build)
    dist_dir = Path(__file__).resolve().parents[3] / "web" / "dist"
    if (dist_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        application.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui")
    else:

        @application.get("/", tags=["Root"])
        async def root_info() -> dict[str, str]:
            return {
                "title": "PG-SBD Designer API",
                "version": "1.0.0",
                "docs_url": "/docs",
                "health_url": "/api/health",
                "honesty_notice": HONESTY_NOTICE,
            }

    return application


app = create_app()
