# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory for model downloads
RUN mkdir -p /app/cache/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=gs://agentzero-models/models/agentZer0_v2
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV MODEL_DEVICE=cpu

# Expose port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the API
CMD ["python", "scripts/start_api.py", "--host", "0.0.0.0", "--port", "8080"]