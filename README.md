# Prompt Injection Detector

A fine-tuned AI model for detecting prompt injection attacks against enterprise AI systems. Built on Llama 3.1-8B with LoRA fine-tuning for parameter efficiency.

## Problem Statement

Enterprise AI agents are vulnerable to prompt injection attacks that can:
- Bypass safety guardrails (jailbreaking)
- Extract sensitive data through crafted prompts
- Manipulate function calling and automation
- Override system instructions

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (for training)
- Hugging Face account with Llama access

### Installation
```bash
git clone https://github.com/yourusername/prompt-injection-detector.git
cd prompt-injection-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys

