"""FastAPI application server initialization."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.routes import router
from src.utils.exceptions import AgentZeroException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AgentZero Prompt Injection Detector API",
    description="AI-powered prompt injection detection for enterprise AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Global exception handler
@app.exception_handler(AgentZeroException)
async def agentzero_exception_handler(request: Request, exc: AgentZeroException):
    """Handle custom AgentZero exceptions."""
    logger.error(f"AgentZero exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred"
        }
    )


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting AgentZero API...")
    logger.info("Loading model...")
    
    try:
        # Pre-load model by importing predictor
        from src.api.dependencies import get_predictor
        predictor = get_predictor()
        
        model_info = predictor.model_loader.get_model_info()
        logger.info(f"Model loaded successfully: {model_info['model_version']}")
        logger.info(f"Device: {model_info['device']}")
        logger.info(f"Parameters: {model_info['num_parameters']:,}")
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.warning("API will start but predictions may fail")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Shutting down AgentZero API...")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "AgentZero Prompt Injection Detector API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)