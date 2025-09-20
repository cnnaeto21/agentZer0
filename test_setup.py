#!/usr/bin/env python3
"""
Test script to verify setup is working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        from peft import LoraConfig
        print("✓ PEFT (LoRA) imported successfully")
    except ImportError as e:
        print(f"✗ PEFT import failed: {e}")
        return False
    
    try:
        import wandb
        print("✓ Weights & Biases imported successfully")
    except ImportError as e:
        print(f"✗ W&B import failed: {e}")
        return False
    
    return True

def test_config():
    """Test model configuration."""
    print("\nTesting configuration...")
    
    try:
        from config.model_config import ModelConfig
        config = ModelConfig.from_env()
        print(f"✓ Model config loaded: {config.model_name}")
        print(f"✓ Max length: {config.max_length}")
        print(f"✓ Batch size: {config.batch_size}")
        return True
    except ImportError as e:
        print(f"✗ Config import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False

def test_env_file():
    """Test environment file."""
    print("\nTesting environment...")
    
    if os.path.exists('.env'):
        print("✓ .env file exists")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                print(f"✓ HF_TOKEN loaded: {hf_token[:10]}...")
            else:
                print("⚠ HF_TOKEN not found in .env")
            
            wandb_key = os.getenv('WANDB_API_KEY')
            if wandb_key:
                print(f"✓ WANDB_API_KEY loaded: {wandb_key[:10]}...")
            else:
                print("⚠ WANDB_API_KEY not found in .env")
                
        except Exception as e:
            print(f"✗ Error loading .env: {e}")
    else:
        print("⚠ .env file not found - copy .env.example to .env and add your keys")

def main():
    print("=== Prompt Injection Detector Setup Test ===\n")
    
    success = True
    success &= test_imports()
    success &= test_config()
    test_env_file()  # This doesn't affect success
    
    print(f"\n=== Setup {'PASSED' if success else 'FAILED'} ===")
    
    if success:
        print("\n🎉 Ready to run training! Try:")
        print("python scripts/train_model.py")
    else:
        print("\n❌ Fix the issues above before proceeding")

if __name__ == "__main__":
    main()
