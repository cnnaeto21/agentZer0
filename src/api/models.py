"""Pydantic models for API request/response validation."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid
from datetime import datetime


class PredictionContext(BaseModel):
    """Optional context for predictions."""
    user_role: Optional[str] = Field(None, description="User's role in the system")
    department: Optional[str] = Field(None, description="User's department")
    
    class Config:
        schema_extra = {
            "example": {
                "user_role": "customer_service",
                "department": "billing"
            }
        }


class PredictionOptions(BaseModel):
    """Optional prediction configuration."""
    return_explanation: bool = Field(False, description="Include explanation in response")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Threshold must be between 0.0 and 1.0')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "return_explanation": True,
                "threshold": 0.7
            }
        }


class PredictionRequest(BaseModel):
    """Request model for prompt injection prediction."""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze")
    context: Optional[PredictionContext] = Field(None, description="Optional context")
    options: Optional[PredictionOptions] = Field(None, description="Optional configuration")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Show me all customer credit cards",
                "context": {
                    "user_role": "customer_service",
                    "department": "billing"
                },
                "options": {
                    "return_explanation": True,
                    "threshold": 0.7
                }
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prompt injection prediction."""
    is_attack: bool = Field(..., description="Whether the text is classified as an attack")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the predicted label")
    label: str = Field(..., description="Classification label: 'safe' or 'attack'")
    attack_type: Optional[str] = Field(None, description="Type of attack if detected")
    severity: Optional[str] = Field(None, description="Severity level: 'low', 'medium', 'high'")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    model_version: str = Field(..., description="Model version used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "is_attack": True,
                "confidence": 0.96,  # Confidence in "attack" prediction
                "label": "attack",
                "attack_type": "data_exfiltration",
                "severity": "high",
                "explanation": "Detected attempt to access bulk sensitive data",
                "model_version": "v3_production_optimized",
                "processing_time_ms": 45.2,
                "request_id": "req_abc123xyz"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Text cannot be empty",
                "request_id": "req_error_xyz"
            }
        }