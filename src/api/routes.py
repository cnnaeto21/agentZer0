"""API routes for prompt injection detection."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import logging

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    ErrorResponse
)
from src.api.dependencies import get_current_predictor
from src.inference.predictor import PromptInjectionPredictor
from src.utils.exceptions import PredictionError, ModelLoadError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post(
    "/v1/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
        503: {"model": ErrorResponse, "description": "Model not available"}
    },
    summary="Predict prompt injection",
    description="Analyze text for potential prompt injection attacks"
)
async def predict(
    request: PredictionRequest,
    predictor: PromptInjectionPredictor = Depends(get_current_predictor)
) -> PredictionResponse:
    """
    Analyze text for prompt injection attacks.
    
    Args:
        request: Prediction request with text and optional parameters
        predictor: Injected predictor instance
        
    Returns:
        Prediction response with classification results
        
    Raises:
        HTTPException: If prediction fails
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        logger.info(f"[{request_id}] Received prediction request: {len(request.text)} chars")
        
        # Extract options
        threshold = 0.5
        return_explanation = False
        
        if request.options:
            threshold = request.options.threshold
            return_explanation = request.options.return_explanation
        
        # Run prediction
        result = predictor.predict(
            text=request.text,
            threshold=threshold,
            return_explanation=return_explanation,
            context=request.context.dict() if request.context else None
        )
        
        # Add request_id to result
        result["request_id"] = request_id
        
        logger.info(
            f"[{request_id}] Prediction complete: "
            f"label={result['label']}, confidence={result['confidence']:.3f}"
        )
        
        return PredictionResponse(**result)
        
    except PredictionError as e:
        logger.error(f"[{request_id}] Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": str(e),
                "request_id": request_id
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": request_id
            }
        )


@router.get(
    "/health",
    summary="Basic health check",
    description="Check if API is running"
)
async def health() -> Dict[str, str]:
    """
    Basic health check endpoint.
    
    Returns:
        Status message
    """
    return {
        "status": "healthy",
        "service": "agentZero-api"
    }


@router.get(
    "/health/ready",
    summary="Readiness check",
    description="Check if API is ready to handle requests (model loaded)"
)
async def health_ready(
    predictor: PromptInjectionPredictor = Depends(get_current_predictor)
) -> Dict[str, Any]:
    """
    Readiness check - verifies model is loaded and ready.
    
    Args:
        predictor: Injected predictor instance
        
    Returns:
        Readiness status
        
    Raises:
        HTTPException: If model is not ready
    """
    try:
        # Check if predictor and model are available
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "ready": False,
                    "model_loaded": False,
                    "message": "Model not loaded"
                }
            )
        
        # Get model info to verify it's working
        model_info = predictor.model_loader.get_model_info()
        
        return {
            "ready": True,
            "model_loaded": True,
            "model_version": model_info.get("model_version"),
            "device": model_info.get("device")
        }
        
    except ModelLoadError as e:
        logger.error(f"Model not ready: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ready": False,
                "model_loaded": False,
                "message": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "ready": False,
                "message": "Readiness check failed"
            }
        )


@router.get(
    "/health/live",
    summary="Liveness check",
    description="Check if API process is alive"
)
async def health_live() -> Dict[str, bool]:
    """
    Liveness check - verifies API process is responsive.
    
    Returns:
        Liveness status
    """
    return {
        "alive": True
    }