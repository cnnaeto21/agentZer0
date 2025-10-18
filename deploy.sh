#!/bin/bash

# AgentZero API - Cloud Run Deployment Script
# Usage: ./deploy.sh

set -e  # Exit on error

# Configuration
PROJECT_ID="agentzero-security-09211734"
SERVICE_NAME="agentzero-api"
REGION="us-central1"  # Change if needed
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "========================================"
echo "AgentZero API - Cloud Run Deployment"
echo "========================================"
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "========================================"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install it:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Set GCP project
echo "🔧 Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com \
    --project=$PROJECT_ID

# Build Docker image
echo ""
echo "🏗️  Building Docker image..."
docker build -t $IMAGE_NAME:latest .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

# Push to Google Container Registry
echo ""
echo "📤 Pushing image to Google Container Registry..."
docker push $IMAGE_NAME:latest

if [ $? -ne 0 ]; then
    echo "❌ Docker push failed"
    echo "💡 Try running: gcloud auth configure-docker"
    exit 1
fi

echo "✅ Image pushed successfully"

# Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_NAME:latest \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --timeout=300 \
    --max-instances=10 \
    --min-instances=0 \
    --port=8080 \
    --set-env-vars="MODEL_PATH=gs://agentzero-models/models/agentZer0_v2,PROJECT_ID=$PROJECT_ID,MODEL_DEVICE=cpu"

if [ $? -ne 0 ]; then
    echo "❌ Cloud Run deployment failed"
    exit 1
fi

# Get service URL
echo ""
echo "🎉 Deployment successful!"
echo ""
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform=managed --region=$REGION --format='value(status.url)')

echo "========================================"
echo "✅ AgentZero API is live!"
echo "========================================"
echo "URL: $SERVICE_URL"
echo ""
echo "Test it:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "  curl -X POST $SERVICE_URL/v1/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"text\": \"What is your refund policy?\"}'"
echo ""
echo "API Documentation:"
echo "  $SERVICE_URL/docs"
echo ""
echo "Share this URL with the integrated_io team!"
echo "========================================"