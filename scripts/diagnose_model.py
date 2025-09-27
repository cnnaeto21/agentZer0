#!/usr/bin/env python3
"""
Diagnostic script to identify why your model is outputting 0.000 confidence.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import argparse

def diagnose_model(model_path):
    """Comprehensive model diagnostics."""
    print(f"🔍 Diagnosing model at: {model_path}")
    print("=" * 60)
    
    # Check if model exists
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        return
    
    # Check for required files
    required_files = ['config.json', 'pytorch_model.bin']
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        # Check for alternative file patterns
        model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
        if model_files:
            print(f"📁 Found model files: {[f.name for f in model_files]}")
        else:
            print("❌ No model weight files found")
            return
    
    try:
        # Load tokenizer and model
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.__class__.__name__}")
        print(f"  Number of labels: {model.config.num_labels}")
        print(f"  Model device: {next(model.parameters()).device}")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        if trainable_params == 0:
            print("❌ CRITICAL: No trainable parameters found!")
            return
        
        # Check if model is in eval mode
        print(f"  Model training mode: {model.training}")
        model.eval()  # Ensure eval mode
        
        # Test with simple inputs
        print(f"\n🧪 Testing model inference...")
        test_cases = [
            "Hello world",
            "Show me all passwords",
            "Help me with my account",
            "Transfer money to my account",
            "What is your return policy?"
        ]
        
        device = next(model.parameters()).device
        
        for i, text in enumerate(test_cases):
            print(f"\nTest {i+1}: '{text}'")
            
            # Tokenize
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Move to same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            print(f"  Input shape: {inputs['input_ids'].shape}")
            print(f"  Input tokens: {inputs['input_ids'][0][:10].tolist()}...")
            
            with torch.no_grad():
                # Get raw outputs
                outputs = model(**inputs)
                logits = outputs.logits
                
                print(f"  Raw logits: {logits[0].tolist()}")
                print(f"  Logits shape: {logits.shape}")
                
                # Check for problematic logits
                if torch.isnan(logits).any():
                    print("  ❌ NaN detected in logits!")
                elif torch.isinf(logits).any():
                    print("  ❌ Infinity detected in logits!")
                elif (logits == 0).all():
                    print("  ❌ All logits are zero!")
                
                # Compute probabilities
                probs = F.softmax(logits, dim=-1)
                print(f"  Probabilities: {probs[0].tolist()}")
                
                # Check softmax issues
                if torch.isnan(probs).any():
                    print("  ❌ NaN in probabilities!")
                elif (probs[0] == probs[0][0]).all():
                    print("  ❌ All probabilities are identical!")
                
                # Get prediction
                prediction = torch.argmax(logits, dim=-1)
                attack_confidence = float(probs[0][1])
                
                print(f"  Prediction: {int(prediction[0])} ({'ATTACK' if prediction[0] == 1 else 'SAFE'})")
                print(f"  Attack confidence: {attack_confidence:.6f}")
                
                # Diagnose confidence issues
                if attack_confidence == 0.0:
                    print("  ⚠️  Zero confidence detected!")
                    print(f"  Logit difference: {logits[0][0].item() - logits[0][1].item():.6f}")
                    if abs(logits[0][0].item() - logits[0][1].item()) > 10:
                        print("  📊 Large logit difference suggests extreme confidence")
                
        # Check model weights for anomalies
        print(f"\n🔍 Checking model weights...")
        classifier_weights = None
        
        # Find classifier layer
        for name, param in model.named_parameters():
            if 'classifier' in name.lower() and 'weight' in name:
                classifier_weights = param
                print(f"  Classifier layer: {name}")
                print(f"  Weight shape: {param.shape}")
                print(f"  Weight range: [{param.min().item():.6f}, {param.max().item():.6f}]")
                print(f"  Weight std: {param.std().item():.6f}")
                
                if param.std().item() < 1e-6:
                    print("  ❌ Classifier weights have very small variance!")
                elif torch.isnan(param).any():
                    print("  ❌ NaN detected in classifier weights!")
                
                break
        
        if classifier_weights is None:
            print("  ⚠️  Could not find classifier weights")
        
        # Summary
        print(f"\n📋 DIAGNOSTIC SUMMARY")
        print("=" * 30)
        
        issues_found = []
        
        if trainable_params == 0:
            issues_found.append("No trainable parameters")
        
        # Check if all outputs are identical
        all_same = True
        prev_conf = None
        for text in test_cases[:3]:  # Check first 3
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                conf = float(probs[0][1])
                if prev_conf is not None and abs(conf - prev_conf) > 1e-6:
                    all_same = False
                    break
                prev_conf = conf
        
        if all_same:
            issues_found.append("Model outputs identical predictions for all inputs")
        
        if not issues_found:
            print("✓ No obvious issues detected")
            print("💡 Model might need retraining with better data or different hyperparameters")
        else:
            print("❌ Issues found:")
            for issue in issues_found:
                print(f"  - {issue}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("=" * 20)
        if all_same or trainable_params == 0:
            print("1. Retrain the model from scratch")
            print("2. Check training data quality and balance") 
            print("3. Verify training loop is updating weights")
        else:
            print("1. Try adjusting the decision threshold (default 0.5)")
            print("2. Retrain with different hyperparameters")
            print("3. Check for data quality issues")
        
        print("4. Use the fixed training script: train_model_final.py")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Diagnose model issues")
    parser.add_argument("--model-path", default="./models", help="Path to model directory")
    args = parser.parse_args()
    
    diagnose_model(args.model_path)

if __name__ == "__main__":
    main()