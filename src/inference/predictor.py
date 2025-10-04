"""Core prediction logic for prompt injection detection."""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging

from src.inference.model_loader import ModelLoader
from src.utils.exceptions import PredictionError

logger = logging.getLogger(__name__)


class PromptInjectionPredictor:
    """Handles prompt injection predictions."""
    
    def __init__(self, model_loader: ModelLoader):
        """
        Initialize predictor with a model loader.
        
        Args:
            model_loader: Loaded ModelLoader instance
        """
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.model_version = model_loader.model_version
        
    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        return_explanation: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict if text is a prompt injection attack.
        
        Args:
            text: Input text to analyze
            threshold: Classification threshold (default 0.5)
            return_explanation: Whether to include explanation
            context: Optional context (user_role, department, etc.)
            
        Returns:
            Prediction dictionary with classification results
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                
                # Get probabilities
                probs = F.softmax(logits, dim=-1)
                safe_confidence = float(probs[0])
                attack_confidence = float(probs[1])
            
            # Classify based on threshold
            is_attack = attack_confidence > threshold
            label = "attack" if is_attack else "safe"
            
            # Confidence = probability of the predicted class
            confidence = attack_confidence if is_attack else safe_confidence
            
            # Determine attack type and severity if it's an attack
            attack_type = None
            severity = None
            explanation = None
            
            if is_attack:
                attack_type = self._classify_attack_type(text)
                severity = self._assess_severity(attack_confidence, attack_type)
                
                if return_explanation:
                    explanation = self._generate_explanation(
                        text, attack_type, attack_confidence, context
                    )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            result = {
                "is_attack": is_attack,
                "confidence": confidence,  # Confidence in the predicted class
                "label": label,
                "attack_type": attack_type,
                "severity": severity,
                "explanation": explanation,
                "model_version": self.model_version,
                "processing_time_ms": round(processing_time, 2)
            }
            
            logger.info(
                f"Prediction complete: label={label}, confidence={confidence:.3f}, "
                f"time={processing_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Failed to generate prediction: {str(e)}")
    
    def _classify_attack_type(self, text: str) -> str:
        """
        Classify the type of attack based on text patterns.
        
        Args:
            text: Input text
            
        Returns:
            Attack type string
        """
        text_lower = text.lower()
        
        # Data exfiltration patterns
        if any(word in text_lower for word in [
            'show', 'display', 'export', 'list', 'reveal', 'dump'
        ]) and any(word in text_lower for word in [
            'all', 'customer', 'user', 'password', 'credit card', 'data', 'database'
        ]):
            return "data_exfiltration"
        
        # Instruction bypass patterns
        if any(phrase in text_lower for phrase in [
            'ignore previous', 'ignore all', 'forget', 'disregard', 'override'
        ]):
            return "instruction_bypass"
        
        # Social engineering patterns
        if any(phrase in text_lower for phrase in [
            "i'm the ceo", "i'm from", "this is urgent", "emergency", "authorized"
        ]):
            return "social_engineering"
        
        # Function manipulation patterns
        if any(word in text_lower for word in [
            'bypass', 'disable', 'grant', 'admin', 'permission', 'access'
        ]):
            return "function_manipulation"
        
        # Context pollution patterns
        if any(phrase in text_lower for phrase in [
            'as we discussed', 'following up', 'per our conversation', 'as agreed'
        ]):
            return "context_pollution"
        
        # Default to generic injection
        return "prompt_injection"
    
    def _assess_severity(self, confidence: float, attack_type: str) -> str:
        """
        Assess severity of the attack.
        
        Args:
            confidence: Model confidence score
            attack_type: Type of attack
            
        Returns:
            Severity level: 'low', 'medium', or 'high'
        """
        # High-risk attack types
        high_risk_types = ['data_exfiltration', 'function_manipulation']
        
        if attack_type in high_risk_types and confidence > 0.9:
            return "high"
        elif attack_type in high_risk_types or confidence > 0.8:
            return "medium"
        else:
            return "low"
    
    def _generate_explanation(
        self,
        text: str,
        attack_type: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable explanation for the classification.
        
        Args:
            text: Input text
            attack_type: Detected attack type
            confidence: Model confidence
            context: Optional context
            
        Returns:
            Explanation string
        """
        explanations = {
            "data_exfiltration": "Detected attempt to access bulk sensitive data",
            "instruction_bypass": "Detected attempt to override system instructions",
            "social_engineering": "Detected social engineering or authority impersonation",
            "function_manipulation": "Detected attempt to manipulate system functions",
            "context_pollution": "Detected attempt to manipulate conversation context",
            "prompt_injection": "Detected potential prompt injection attack"
        }
        
        base_explanation = explanations.get(attack_type, "Detected suspicious pattern")
        
        # Add confidence qualifier
        if confidence > 0.95:
            qualifier = "High confidence"
        elif confidence > 0.8:
            qualifier = "Moderate confidence"
        else:
            qualifier = "Low confidence"
        
        explanation = f"{qualifier}: {base_explanation}"
        
        # Add context if available
        if context:
            if context.get('user_role'):
                explanation += f" (User role: {context['user_role']})"
        
        return explanation