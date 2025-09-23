#!/bin/bash
# GCP Training Environment Setup Script for agentZer0
# Fixed version addressing GPU configuration and image issues

set -e

# Configuration for agentZer0 project
PROJECT_ID="agentzero-security-09211734"
ZONE="us-east1-c"
INSTANCE_NAME="agentzero-trainer"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
BOOT_DISK_SIZE="200GB"  # Increased for academic datasets

echo "🛡️  Setting up GCP training environment for agentZer0..."
echo "📍 Project: $PROJECT_ID"
echo "🌍 Zone: $ZONE"
echo "💻 GPU: $GPU_TYPE"

# Set project and zone
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --quiet >/dev/null 2>&1; then
    echo "⚠️  Instance $INSTANCE_NAME already exists. Deleting it..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    echo "🗑️  Existing instance deleted"
fi

# Create instance with GPU - FIXED VERSION
echo "🚀 Creating agentZer0 training instance with GPU..."
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --tags="agentzero,ml-training,prompt-security" \
    --labels="project=agentzero,component=training,environment=development,purpose=prompt-injection-detection"

echo "⏳ Instance created. Waiting for it to be ready..."
sleep 60

# Verify GPU is available
echo "🔍 Verifying GPU setup..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="nvidia-smi" || {
    echo "⚠️  GPU not detected. Checking driver installation..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        sudo apt update
        sudo apt install -y nvidia-driver-470
        sudo reboot
    "
    echo "🔄 Rebooting for driver installation..."
    sleep 120
}

# Copy your code to the instance
echo "📦 Copying agentZer0 code to instance..."
gcloud compute scp --recurse . $INSTANCE_NAME:~/agentzero --zone=$ZONE

# Install dependencies on the instance
echo "🔧 Setting up agentZer0 environment on instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    cd ~/agentzero
    
    # Create Python virtual environment
    python3 -m venv agentzero_env
    source agentzero_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn
    pip install wandb tensorboard jupyter notebook
    
    # Create necessary directories
    mkdir -p data/academic_datasets
    mkdir -p models/checkpoints
    mkdir -p results/experiments
    
    # Verify GPU access in Python
    python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); print(f\"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}\")' 
    
    echo '✅ agentZer0 setup complete!'
    echo '🎯 Ready to train model to reduce 50% false positive rate'
"

# Display connection information
echo ""
echo "🎉 agentZer0 GCP training environment ready!"
echo "=============================================="
echo "📍 Instance: $INSTANCE_NAME"
echo "🌍 Zone: $ZONE"
echo "💻 Machine: $MACHINE_TYPE with $GPU_TYPE"
echo "🛡️  Project: agentZer0 Prompt Injection Security"
echo ""
echo "🔗 Connect with:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Estimated costs:"
echo "   T4 GPU: ~$0.35/hour"
echo "   Storage: ~$0.10/day"
echo ""
echo "🎯 Next steps:"
echo "   1. Run academic data collection"
echo "   2. Train hybrid model (synthetic + academic data)"
echo "   3. Target: <10% false positive rate"
echo ""
echo "⚠️  Remember to stop instance when not training:"
echo "   ./scripts/manage_gcp_costs.sh stop"

# Save configuration for cost management script
echo "PROJECT_ID=$PROJECT_ID" > .gcp_config
echo "INSTANCE_NAME=$INSTANCE_NAME" >> .gcp_config
echo "ZONE=$ZONE" >> .gcp_config
echo "MACHINE_TYPE=$MACHINE_TYPE" >> .gcp_config
echo "GPU_TYPE=$GPU_TYPE" >> .gcp_config