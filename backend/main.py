"""
FastAPI backend for the Eco-Switch Intent Prediction application.

Endpoints:
  GET  /health          → service health + model status
  POST /api/v1/predict  → predict switching intent from survey inputs
  GET  /docs            → Swagger UI (built-in)
  GET  /openapi.json    → OpenAPI schema
"""

import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import HealthResponse, PredictionRequest, PredictionResponse
from backend.services import build_explainer, run_prediction

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parent.parent / "eco_switch_intent_pipeline.joblib"),
    )
    logger.info("Loading pipeline from: %s", model_path)
    try:
        app.state.pipeline = joblib.load(model_path)
        app.state.model_path = model_path
        app.state.shap_explainer = build_explainer(app.state.pipeline)
        logger.info("Pipeline loaded successfully. SHAP explainer: %s",
                    type(app.state.shap_explainer).__name__ if app.state.shap_explainer else "None")
    except FileNotFoundError:
        logger.error("Pipeline file not found at: %s — exiting.", model_path)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load pipeline: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)

    yield  # app is running

    logger.info("Shutting down — releasing resources.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Eco-Switch Intent Prediction API",
    description=(
        "Predicts whether a menstrual hygiene product user is likely to switch "
        "to an eco-friendly alternative, based on a 333-respondent real-world survey. "
        "Powered by a scikit-learn pipeline (Random Forest / XGBoost) with SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow Streamlit frontend (configure via env var for production)
# ---------------------------------------------------------------------------
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",")]
else:
    # Development defaults — restrict in production
    allowed_origins = ["http://localhost:8501", "http://127.0.0.1:8501", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s\n%s",
                 request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
def health_check():
    """Returns service health and model load status."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "model_loaded": False, "model_path": None},
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_path=getattr(app.state, "model_path", None),
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict eco-switch intent",
    description=(
        "Submit a survey profile and receive a binary prediction (0/1) indicating "
        "whether the respondent is likely to switch to eco-friendly menstrual products, "
        "along with a probability score and optional SHAP feature attributions."
    ),
)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    - Validates all 22 feature-vector fields via Pydantic (returns 422 on bad input).
    - Runs the pre-trained sklearn pipeline.
    - Returns switch intent, probability, and SHAP values.
    """
    logger.info(
        "POST /api/v1/predict — Age=%s, Area=%s, Occupation=%s",
        request.Age, request.Area, request.Occupation,
    )

    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again shortly.")

    try:
        # model_dump with by_alias=True gives us the original survey column names
        result = run_prediction(
            pipeline=pipeline,
            explainer=getattr(app.state, "shap_explainer", None),
            request_data=request.model_dump(by_alias=True),
        )
    except Exception as exc:
        logger.error("Prediction failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(**result)
