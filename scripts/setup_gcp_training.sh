#!/bin/bash
# GCP Training Environment Setup Script for agentZer0
# Auto-discovery version that finds available US zones with T4 GPUs

set -e

# Configuration for agentZer0 project
PROJECT_ID="agentzero-security-09211734"
INSTANCE_NAME="agentzero-trainer"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
BOOT_DISK_SIZE="200GB"

# US zones to try (ordered by typical availability)
US_ZONES=(
    "us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f"
    "us-east1-b" "us-east1-c" "us-east1-d" "us-east1-a"
    "us-east4-a" "us-east4-b" "us-east4-c"
    "us-west1-a" "us-west1-c"
    "us-west2-a" "us-west2-b" "us-west2-c"
    "us-west3-a" "us-west3-b" "us-west3-c"
    "us-west4-a" "us-west4-b" "us-west4-c"
)

echo "🛡️  Setting up GCP training environment for agentZer0..."
echo "📍 Project: $PROJECT_ID"
echo "🔍 Auto-discovering available US zone with T4 GPU..."

# Set project
gcloud config set project $PROJECT_ID

# Function to test zone availability
test_zone_availability() {
    local zone=$1
    echo "Testing $zone..."
    
    gcloud compute instances create temp-test-$zone \
        --zone=$zone \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
        --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-standard \
        --maintenance-policy=TERMINATE \
        --dry-run >/dev/null 2>&1
}

# Find available zone
AVAILABLE_ZONE=""
echo "🔍 Checking T4 GPU availability across US zones..."

for zone in "${US_ZONES[@]}"; do
    echo "  Checking $zone..."
    
    if test_zone_availability $zone; then
        echo "✅ $zone has T4 GPU availability!"
        AVAILABLE_ZONE=$zone
        break
    else
        echo "❌ $zone unavailable"
    fi
done

# Check if we found an available zone
if [ -z "$AVAILABLE_ZONE" ]; then
    echo ""
    echo "😞 No US zones currently have T4 GPU availability"
    echo ""
    echo "💡 Options:"
    echo "1. Try again in 15-30 minutes (availability changes frequently)"
    echo "2. Use a preemptible instance (cheaper, can be stopped by Google)"
    echo "3. Consider a different GPU type (K80, P4, P100)"
    echo "4. Train on CPU only (slower but will work)"
    echo ""
    echo "🔄 Would you like to try preemptible instances? (cheaper but less reliable)"
    read -p "Try preemptible? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔍 Checking preemptible availability..."
        for zone in "${US_ZONES[@]}"; do
            echo "  Checking preemptible in $zone..."
            
            if gcloud compute instances create temp-test-preemptible-$zone \
                --zone=$zone \
                --machine-type=$MACHINE_TYPE \
                --accelerator=type=$GPU_TYPE,count=$GPU_COUNT \
                --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
                --image-project=deeplearning-platform-release \
                --boot-disk-size=$BOOT_DISK_SIZE \
                --boot-disk-type=pd-standard \
                --maintenance-policy=TERMINATE \
                --preemptible \
                --dry-run >/dev/null 2>&1; then
                
                echo "✅ Found preemptible availability in $zone!"
                AVAILABLE_ZONE=$zone
                PREEMPTIBLE_FLAG="--preemptible"
                break
            else
                echo "❌ $zone preemptible unavailable"
            fi
        done
    fi
    
    if [ -z "$AVAILABLE_ZONE" ]; then
        echo "❌ No availability found. Exiting."
        exit 1
    fi
fi

# Set the discovered zone
ZONE=$AVAILABLE_ZONE
gcloud config set compute/zone $ZONE

echo ""
echo "🎯 Selected Zone: $ZONE"
echo "💻 GPU: $GPU_TYPE"
if [ ! -z "$PREEMPTIBLE_FLAG" ]; then
    echo "⚡ Instance Type: Preemptible (cheaper but can be stopped)"
fi

# Check if instance already exists
if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --quiet >/dev/null 2>&1; then
    echo "⚠️  Instance $INSTANCE_NAME already exists in $ZONE. Deleting it..."
    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
    echo "🗑️  Existing instance deleted"
fi

# Create instance with GPU
echo "🚀 Creating agentZer0 training instance with GPU in $ZONE..."
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
    --labels="project=agentzero,component=training,environment=development,purpose=prompt-injection-detection" \
    $PREEMPTIBLE_FLAG

if [ $? -eq 0 ]; then
    echo "✅ Instance created successfully in $ZONE!"
else
    echo "❌ Instance creation failed. The zone may have become unavailable."
    echo "💡 Try running the script again - availability changes frequently."
    exit 1
fi

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
    
    # Install ML requirements
    pip install transformers datasets scikit-learn pandas numpy matplotlib seaborn
    pip install wandb tensorboard jupyter notebook peft accelerate
    
    # Install data collection requirements
    pip install requests beautifulsoup4
    
    # Create necessary directories
    mkdir -p data/raw/comprehensive
    mkdir -p data/academic_datasets
    mkdir -p models/checkpoints
    mkdir -p results/experiments
    
    # Verify GPU access in Python
    python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); print(f\"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}\")' 
    
    echo '✅ agentZer0 setup complete!'
    echo '🎯 Ready to collect data and train model!'
"

# Display connection information
echo ""
echo "🎉 agentZer0 GCP training environment ready!"
echo "=============================================="
echo "📍 Instance: $INSTANCE_NAME"
echo "🌍 Zone: $ZONE"
echo "💻 Machine: $MACHINE_TYPE with $GPU_TYPE"
if [ ! -z "$PREEMPTIBLE_FLAG" ]; then
    echo "⚡ Type: Preemptible (cheaper, can be stopped by Google)"
fi
echo "🛡️  Project: agentZer0 Prompt Injection Security"
echo ""
echo "🔗 Connect with:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Estimated costs:"
if [ ! -z "$PREEMPTIBLE_FLAG" ]; then
    echo "   T4 GPU (preemptible): ~$0.10/hour"
else
    echo "   T4 GPU: ~$0.35/hour"
fi
echo "   Storage: ~$0.10/day"
echo ""
echo "🎯 Next steps:"
echo "   1. SSH into instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "   2. Activate environment: source ~/agentzero/agentzero_env/bin/activate"
echo "   3. Collect data: python scripts/complete_data_collector.py"
echo "   4. Train model: python scripts/train_model_fixed.py --data-path data/raw/comprehensive/balanced_training_data_latest.csv"
echo "   5. Target: <10% false positive rate"
echo ""
echo "⚠️  Remember to stop instance when not training:"
echo "   ./scripts/manage_gcp_costs.sh stop"

# Save configuration for cost management script
echo "PROJECT_ID=$PROJECT_ID" > .gcp_config
echo "INSTANCE_NAME=$INSTANCE_NAME" >> .gcp_config
echo "ZONE=$ZONE" >> .gcp_config
echo "MACHINE_TYPE=$MACHINE_TYPE" >> .gcp_config
echo "GPU_TYPE=$GPU_TYPE" >> .gcp_config

echo ""
echo "✅ Configuration saved to .gcp_config"
echo "🚀 Ready to start training!"