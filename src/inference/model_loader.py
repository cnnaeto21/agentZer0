"""Model loading and caching for inference."""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
from functools import lru_cache

from src.utils.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model loading from local or GCP storage."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to model (local or GCS path like gs://bucket/model)
            device: Device to load model on (cuda/cpu/auto)
        """
        self.model_path = model_path
        self.device = device or self._detect_device()
        self.model = None
        self.tokenizer = None
        self.model_version = self._extract_version(model_path)
        
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _extract_version(self, path: str) -> str:
        """Extract model version from path."""
        # Extract version from path like ./models/v3_production or gs://bucket/v3_production
        path_parts = path.rstrip('/').split('/')
        version = path_parts[-1] if path_parts else "unknown"
        return version
    
    def _download_from_gcs(self, gcs_path: str, local_path: str) -> str:
        """
        Download model from GCS to local storage.
        
        Args:
            gcs_path: GCS path (gs://bucket/path/to/model)
            local_path: Local path to download to
            
        Returns:
            Local path where model was downloaded
        """
        try:
            from google.cloud import storage
            
            # Parse GCS path
            if not gcs_path.startswith('gs://'):
                raise ValueError(f"Invalid GCS path: {gcs_path}")
            
            path_parts = gcs_path[5:].split('/', 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1] if len(path_parts) > 1 else ""
            
            logger.info(f"Downloading model from GCS: {gcs_path}")
            
            # Initialize GCS client
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create local directory
            local_path_obj = Path(local_path)
            local_path_obj.mkdir(parents=True, exist_ok=True)
            
            # Download all files in the model directory
            blobs = bucket.list_blobs(prefix=blob_path)
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                
                # Calculate relative path
                rel_path = blob.name[len(blob_path):].lstrip('/')
                local_file = local_path_obj / rel_path
                
                # Create parent directories
                local_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                logger.debug(f"Downloading {blob.name} to {local_file}")
                blob.download_to_filename(str(local_file))
            
            logger.info(f"Model downloaded to {local_path}")
            return local_path
            
        except ImportError:
            raise ModelLoadError(
                "google-cloud-storage not installed. Install with: pip install google-cloud-storage"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to download model from GCS: {str(e)}")
    
    def load_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Handle GCS paths
            if self.model_path.startswith('gs://'):
                # Download to cache directory
                cache_dir = Path('./cache/models')
                local_path = cache_dir / self.model_version
                
                # Check if already cached
                if not local_path.exists():
                    self.model_path = self._download_from_gcs(self.model_path, str(local_path))
                else:
                    logger.info(f"Using cached model from {local_path}")
                    self.model_path = str(local_path)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            logger.info(f"Loading model from {self.model_path} to {self.device}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(f"Failed to load model from {self.model_path}: {str(e)}")
    
    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        if self.model is None:
            raise ModelLoadError("Model not loaded")
        
        return {
            "model_version": self.model_version,
            "device": self.device,
            "model_path": self.model_path,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_labels": self.model.config.num_labels
        }


# Global model cache
_model_cache = {}


@lru_cache(maxsize=5)
def get_model_loader(model_path: str, device: Optional[str] = None) -> ModelLoader:
    """
    Get or create a cached model loader.
    
    Args:
        model_path: Path to model
        device: Device to load on
        
    Returns:
        ModelLoader instance
    """
    cache_key = f"{model_path}:{device}"
    
    if cache_key not in _model_cache:
        logger.info(f"Creating new model loader for {model_path}")
        loader = ModelLoader(model_path, device)
        loader.load_model()
        _model_cache[cache_key] = loader
    
    return _model_cache[cache_key]