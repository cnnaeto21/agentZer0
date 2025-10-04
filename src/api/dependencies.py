"""Shared dependencies for FastAPI endpoints."""

import os
from functools import lru_cache
from typing import Optional

from src.inference.model_loader import get_model_loader, ModelLoader
from src.inference.predictor import PromptInjectionPredictor
from src.config.model_config import ModelConfig
from src.utils.exceptions import ModelLoadError

import logging

logger = logging.getLogger(__name__)


@lru_cache()
def get_model_path() -> str:
    """
    Get model path from environment or config.
    
    Returns:
        Model path (local or GCS)
    """
    # Check environment variable first
    model_path = os.getenv('MODEL_PATH')
    print(f"Model path: {model_path}")
    
    if not model_path:
        # Fall back to config default
        config = ModelConfig()
        model_path = config.output_dir
    print(f"Model path: {model_path}")
    logger.info(f"Using model path: {model_path}")
    return model_path


@lru_cache()
def get_predictor() -> PromptInjectionPredictor:
    """
    Get or create cached predictor instance.
    
    This is cached to avoid reloading the model on every request.
    
    Returns:
        PromptInjectionPredictor instance
    """
    try:
        model_path = get_model_path()
        device = os.getenv('MODEL_DEVICE', 'auto')
        print(f"Device: {device}")
        logger.info(f"Initializing predictor with model from {model_path}")
        
        # Get or load model
        model_loader = get_model_loader(model_path, device)
        print(f"Model loader: {model_loader}")
        # Create predictor
        predictor = PromptInjectionPredictor(model_loader)
        print(f"Predictor: {predictor}")    
        logger.info("Predictor initialized successfully")
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise ModelLoadError(f"Failed to initialize predictor: {str(e)}")


def get_current_predictor() -> PromptInjectionPredictor:
    """
    Dependency function for FastAPI routes.
    
    Returns:
        Current predictor instance
    """
    return get_predictor()