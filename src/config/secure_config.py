"""
Secure configuration management for the prompt injection detector.
Handles API keys and sensitive configuration securely.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Configure logging to avoid accidentally logging secrets
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureConfig:
    """Secure configuration manager that prevents accidental exposure of secrets."""
    
    def __init__(self):
        self._secrets = {}
        self._config = {}
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables securely."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not installed. Using system environment only.")
    
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value without exposing it in logs or errors."""
        value = os.getenv(key, default)
        if value and key not in self._secrets:
            # Store masked version for logging
            if len(value) > 8:
                masked = f"{value[:4]}...{value[-4:]}"
            else:
                masked = "***"
            self._secrets[key] = masked
            logger.info(f"Loaded secret {key}: {masked}")
        return value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a non-sensitive configuration value."""
        value = os.getenv(key, default)
        if value is not None:
            # Convert string values to appropriate types
            if isinstance(default, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
        
        self._config[key] = value
        logger.info(f"Loaded config {key}: {value}")
        return value
    
    def validate_required_secrets(self, required_keys: list) -> bool:
        """Validate that all required secrets are present."""
        missing = []
        for key in required_keys:
            if not self.get_secret(key):
                missing.append(key)
        
        if missing:
            logger.error(f"Missing required secrets: {missing}")
            logger.error("Please check your .env file and ensure all required keys are set.")
            return False
        return True
    
    def get_huggingface_config(self) -> Dict[str, Any]:
        """Get Hugging Face configuration securely."""
        token = self.get_secret('HF_TOKEN')
        if not token:
            raise ValueError("HF_TOKEN is required. Get it from https://huggingface.co/settings/tokens")
        
        return {
            'token': token,
            'model_name': self.get_config('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct')
        }
    
    def get_wandb_config(self) -> Dict[str, Any]:
        """Get Weights & Biases configuration securely."""
        return {
            'api_key': self.get_secret('WANDB_API_KEY'),
            'project': self.get_config('WANDB_PROJECT', 'prompt-injection-detector'),
            'enabled': bool(self.get_secret('WANDB_API_KEY'))
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            'max_length': self.get_config('MAX_LENGTH', 512),
            'batch_size': self.get_config('BATCH_SIZE', 4),
            'learning_rate': self.get_config('LEARNING_RATE', 2e-5),
            'epochs': self.get_config('EPOCHS', 2),
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration securely."""
        api_key = self.get_secret('API_KEY')
        if not api_key and self.get_config('ENVIRONMENT') == 'production':
            raise ValueError("API_KEY is required in production. Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"")
        
        return {
            'host': self.get_config('API_HOST', '0.0.0.0'),
            'port': self.get_config('API_PORT', 8000),
            'api_key': api_key,
            'environment': self.get_config('ENVIRONMENT', 'development'),
            'debug': self.get_config('DEBUG', True)
        }
    
    def __str__(self) -> str:
        """String representation that doesn't expose secrets."""
        safe_config = {
            'secrets_loaded': list(self._secrets.keys()),
            'config_loaded': self._config
        }
        return f"SecureConfig({safe_config})"
    
    def __repr__(self) -> str:
        """Representation that doesn't expose secrets."""
        return self.__str__()

# Global instance
config = SecureConfig()

def get_secure_config() -> SecureConfig:
    """Get the global secure configuration instance."""
    return config

# Convenience functions
def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    required_secrets = ['HF_TOKEN']
    return config.validate_required_secrets(required_secrets)

def get_safe_model_config() -> Dict[str, Any]:
    """Get model configuration with secure token handling."""
    hf_config = config.get_huggingface_config()
    training_config = config.get_training_config()
    
    return {
        'model_name': hf_config['model_name'],
        'token': hf_config['token'],  # Will be handled securely
        **training_config
    }