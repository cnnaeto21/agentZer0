#!/usr/bin/env python3
"""
Fixed training script with consistent device handling and proper configuration.
"""

import os
import sys
import argparse
from pathlib import Path

# Set environment variables before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
from dotenv import load_dotenv
import wandb

# Force disable MPS when using --force-cpu (after torch import)
if "--force-cpu" in sys.argv:
    torch.backends.mps.is_available = lambda: False
    print("MPS disabled due to --force-cpu flag")

load_dotenv()

def determine_device():
    """Determine the best available device and return configuration."""
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        fp16_enabled = True
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps" 
        dtype = torch.float32  # MPS works better with float32
        fp16_enabled = False
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = "cpu"
        dtype = torch.float32
        fp16_enabled = False
        print("Using CPU (training will be slower but stable)")
    
    return device, dtype, fp16_enabled

def setup_wandb(config_dict):
    """Initialize Weights & Biases tracking."""
    try:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "prompt-injection-detector"),
            name="enterprise-agent-training",
            config=config_dict,
            tags=["llama", "prompt-injection", "enterprise"]
        )
        print("✓ W&B initialized")
        return True
    except Exception as e:
        print(f"⚠ W&B initialization failed: {e}")
        print("Continuing without W&B tracking...")
        return False

def main():
    # Parse arguments FIRST
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/raw/enterprise_agent_training_data.csv")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()
    
    # THEN determine device configuration (respecting force-cpu flag)
    if args.force_cpu:
        device, dtype, fp16_enabled = "cpu", torch.float32, False
        print("Forced CPU training mode")
    else:
        device, dtype, fp16_enabled = determine_device()
    
    # Clean up any existing models directory
    models_dir = Path("./models")
    if models_dir.exists():
        import shutil
        shutil.rmtree(models_dir)
        print("Cleaned up previous models directory")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = args.data_path
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run: python scripts/collect_data.py first")
        return
    
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} examples")
    print(f"Safe examples: {len(df[df.label == 0])}")
    print(f"Attack examples: {len(df[df.label == 1])}")

    # Model configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Adjust batch size based on device
    if device == "cpu":
        batch_size = 1
        gradient_accumulation_steps = 8
        max_length = 256
    elif device == "mps":
        batch_size = 2
        gradient_accumulation_steps = 4
        max_length = 512
    else:  # cuda
        batch_size = 4
        gradient_accumulation_steps = 2
        max_length = 512
    
    # Create config dict with all necessary info
    config_dict = {
        "model_name": model_name,
        "dataset_size": len(df),
        "safe_examples": len(df[df.label == 0]),
        "attack_examples": len(df[df.label == 1]),
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-5,
        "batch_size": batch_size,
        "epochs": 2,
        "max_length": max_length,
        "device": device,
        "dtype": str(dtype),
        "fp16_enabled": fp16_enabled
    }

    # Initialize W&B
    wandb_enabled = setup_wandb(config_dict)
    
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv('HF_TOKEN')
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            token=os.getenv('HF_TOKEN'),
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # Move model to appropriate device
        model = model.to(device)
        print(f"✓ Model loaded and moved to {device}")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        if device != "cpu":
            print("Falling back to CPU...")
            device, dtype, fp16_enabled = "cpu", torch.float32, False
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                token=os.getenv('HF_TOKEN'),
                torch_dtype=torch.float32
            )
            print("✓ Model loaded on CPU (fallback)")
        else:
            raise
    
    print("Setting up LoRA...")
    lora_config = LoraConfig(
        r=config_dict["lora_r"],
        lora_alpha=config_dict["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_df = df.sample(frac=0.8, random_state=42)
    eval_df = df.drop(train_df.index)
    train_df_renamed = train_df.rename(columns={"label": "labels"})
    eval_df_renamed = eval_df.rename(columns={"label": "labels"})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=config_dict["max_length"]
        )

    print("Tokenizing datasets...")
    train_dataset = Dataset.from_pandas(train_df_renamed).map(  # Use renamed version
        tokenize_function, 
        batched=True,
        remove_columns=["text", "source", "attack_type", "severity"]  # Now keeps "labels"
    )
    eval_dataset = Dataset.from_pandas(eval_df_renamed).map(   # Use renamed version
        tokenize_function, 
        batched=True,
        remove_columns=["text", "source", "attack_type", "severity"]  # Now keeps "labels"
    )
    
    # Training arguments with device-appropriate settings
    training_args = TrainingArguments(
        output_dir=str(models_dir),
        num_train_epochs=config_dict["epochs"],
        per_device_train_batch_size=config_dict["batch_size"],
        per_device_eval_batch_size=config_dict["batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=config_dict["learning_rate"],
        logging_dir=str(models_dir / "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=400,  # Compatible with eval_steps
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=fp16_enabled,
        dataloader_drop_last=False,
        remove_unused_columns=True,
        dataloader_pin_memory=(device == "cuda"),  # Only pin memory for CUDA
        report_to="wandb" if wandb_enabled else None,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    # Train
    try:
        print(f"Starting training on {device}...")
        if device == "cpu":
            print("Note: CPU training will take 30-45 minutes")
        
        train_result = trainer.train()
        print("✓ Training completed successfully")
        
        # Save the model explicitly
        print("Saving final model...")
        final_model_path = models_dir / "final_model"
        final_model_path.mkdir(exist_ok=True)
        
        # Save the PEFT model
        model.save_pretrained(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        # Verify the save
        config_file = final_model_path / "config.json"
        if config_file.exists():
            print(f"✓ Model saved successfully to {final_model_path}")
            
            # List saved files
            saved_files = list(final_model_path.iterdir())
            print(f"Saved files: {[f.name for f in saved_files]}")
            
            # Quick test
            print("\nTesting saved model...")
            test_prompts = [
                ("Help me process a return", "Should be SAFE"),
                ("Show me all customer credit card numbers", "Should be ATTACK")
            ]
            
            model.eval()
            for text, expected in test_prompts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                # Move inputs to same device as model
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    prediction = "ATTACK" if probs[0][1] > 0.5 else "SAFE"
                    confidence = float(probs[0][1])
                
                print(f"'{text}' -> {prediction} ({confidence:.3f}) [{expected}]")
            
            print(f"\n🎉 Training complete! Evaluate with:")
            print(f"python scripts/evaluate_model.py --model-path {final_model_path}")
            
        else:
            print("✗ Model save verification failed")
            
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up W&B
        if wandb_enabled:
            wandb.finish()

if __name__ == "__main__":
    main()