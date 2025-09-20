"""Model configuration settings."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the prompt injection detection model."""
    
    # Base model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    num_labels: int = 2  # Binary classification: safe vs injection
    max_length: int = 2048
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    
    # Training settings
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    
    # Inference settings
    device: str = "auto"  # Will auto-detect GPU/CPU
    torch_dtype: str = "float16"
    
    # Paths
    output_dir: str = "./models"
    cache_dir: str = "./cache"
    data_dir: str = "./data"
    
    def __post_init__(self):
        """Set default LoRA target modules if not provided."""
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", cls.model_name),
            max_length=int(os.getenv("MAX_LENGTH", cls.max_length)),
            batch_size=int(os.getenv("BATCH_SIZE", cls.batch_size)),
            learning_rate=float(os.getenv("LEARNING_RATE", cls.learning_rate)),
        )

@dataclass 
class APIConfig:
    """Configuration for the API server."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    max_requests_per_minute: int = 60
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load API configuration from environment variables."""
        return cls(
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", cls.port)),
            api_key=os.getenv("API_KEY"),
        )