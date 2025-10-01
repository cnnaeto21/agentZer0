#!/bin/bash
# Global T4 Hunter - checks worldwide, prioritizes preemptible

set -e

PROJECT_ID="agentzero-security-09211734"
INSTANCE_NAME="agentzero-trainer"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
BOOT_DISK_SIZE="200GB"

# Global zones ordered by: 1) US zones, 2) Low latency zones, 3) All others
GLOBAL_ZONES=(
    # US zones first
    "us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f"
    "us-east1-b" "us-east1-c" "us-east1-d"
    "us-west1-a" "us-west1-b" "us-west2-b" "us-west2-c"
    
    # Europe (often best availability)
    "europe-west4-a" "europe-west4-b" "europe-west4-c"
    "europe-west1-b" "europe-west1-c" "europe-west1-d"
    "europe-west3-b" "europe-west3-c"
    
    # Asia (good availability)
    "asia-southeast1-a" "asia-southeast1-b" "asia-southeast1-c"
    "asia-east1-a" "asia-east1-c"
    "asia-northeast1-a" "asia-northeast1-b"
    
    # Canada
    "northamerica-northeast1-a" "northamerica-northeast1-b"
)

echo "🛡️  agentZer0 - Global T4 Hunter"
echo "================================"
echo "💰 Targeting: T4 Preemptible (~\$0.10/hr)"
echo "🌍 Checking: ${#GLOBAL_ZONES[@]} zones worldwide"
echo ""

gcloud config set project $PROJECT_ID

# Function to test availability
test_availability() {
    local zone=$1
    local preempt=$2
    
    local cmd="gcloud compute instances create test-$RANDOM-$zone \
        --zone=$zone \
        --machine-type=$MACHINE_TYPE \
        --accelerator=type=$GPU_TYPE,count=1 \
        --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=50GB \
        --maintenance-policy=TERMINATE"
    
    if [ "$preempt" = "true" ]; then
        cmd="$cmd --preemptible"
    fi
    
    cmd="$cmd --dry-run"
    
    eval $cmd >/dev/null 2>&1
}

echo "🔍 Phase 1: Scanning for PREEMPTIBLE T4s (70% cheaper)..."
PREEMPT_ZONES=()

for zone in "${GLOBAL_ZONES[@]}"; do
    printf "%-35s" "  $zone"
    
    if test_availability "$zone" "true"; then
        echo "✅ PREEMPTIBLE"
        PREEMPT_ZONES+=($zone)
    else
        echo "❌"
    fi
done

if [ ${#PREEMPT_ZONES[@]} -gt 0 ]; then
    echo ""
    echo "🎉 Found ${#PREEMPT_ZONES[@]} zones with preemptible T4!"
    
    # Pick first US zone if available, else first zone
    SELECTED_ZONE=""
    for zone in "${PREEMPT_ZONES[@]}"; do
        if [[ $zone == us-* ]]; then
            SELECTED_ZONE=$zone
            break
        fi
    done
    
    if [ -z "$SELECTED_ZONE" ]; then
        SELECTED_ZONE=${PREEMPT_ZONES[0]}
        echo "⚠️  No US zones available, using $SELECTED_ZONE"
    fi
    
    PREEMPTIBLE_FLAG="--preemptible"
    USE_PREEMPTIBLE=true
else
    echo ""
    echo "😞 No preemptible T4s found anywhere"
    echo ""
    echo "🔍 Phase 2: Scanning for STANDARD T4s..."
    
    STANDARD_ZONES=()
    for zone in "${GLOBAL_ZONES[@]}"; do
        printf "%-35s" "  $zone"
        
        if test_availability "$zone" "false"; then
            echo "✅ STANDARD"
            STANDARD_ZONES+=($zone)
        else
            echo "❌"
        fi
    done
    
    if [ ${#STANDARD_ZONES[@]} -gt 0 ]; then
        echo ""
        echo "🎉 Found ${#STANDARD_ZONES[@]} zones with standard T4!"
        
        # Pick first US zone if available
        SELECTED_ZONE=""
        for zone in "${STANDARD_ZONES[@]}"; do
            if [[ $zone == us-* ]]; then
                SELECTED_ZONE=$zone
                break
            fi
        done
        
        if [ -z "$SELECTED_ZONE" ]; then
            SELECTED_ZONE=${STANDARD_ZONES[0]}
            echo "⚠️  No US zones available, using $SELECTED_ZONE"
        fi
        
        PREEMPTIBLE_FLAG=""
        USE_PREEMPTIBLE=false
    else
        echo ""
        echo "😞 No T4 GPUs available ANYWHERE globally"
        echo ""
        echo "💡 Alternative options:"
        echo ""
        echo "1. Wait 30-60 minutes and try again"
        echo "2. Use V100 (available, faster, \$2.50/hr):"
        echo "   Change GPU_TYPE to 'nvidia-tesla-v100'"
        echo ""
        echo "3. Use Lambda Labs (usually has availability):"
        echo "   https://lambdalabs.com/ (~\$0.60/hr)"
        echo ""
        echo "4. Use Google Colab Pro (\$10/month, guaranteed):"
        echo "   https://colab.research.google.com/"
        echo ""
        echo "5. Train on CPU (slow but free):"
        echo "   python scripts/train_model_fixed.py --force-cpu"
        exit 1
    fi
fi

ZONE=$SELECTED_ZONE
gcloud config set compute/zone $ZONE

echo ""
echo "🎯 SELECTED CONFIGURATION"
echo "========================"
echo "Zone: $ZONE"
echo "GPU: $GPU_TYPE"

if [ "$USE_PREEMPTIBLE" = true ]; then
    echo "Type: Preemptible (~\$0.10/hr)"
    echo "⚠️  Can be stopped by Google anytime"
    echo "✅ 70% cost savings vs standard"
else
    echo "Type: Standard (~\$0.35/hr)"
    echo "✅ Guaranteed availability"
fi

echo ""
read -p "Proceed with this configuration? (y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled by user"
    exit 1
fi

# Delete existing instance if present
echo "🧹 Cleaning up any existing instances..."
gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet 2>/dev/null || true

# Create the instance
echo ""
echo "🚀 Creating instance in $ZONE..."

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=1 \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --tags="agentzero,ml-training" \
    --labels="project=agentzero,cost=$([ "$USE_PREEMPTIBLE" = true ] && echo "budget" || echo "standard")" \
    $PREEMPTIBLE_FLAG

if [ $? -ne 0 ]; then
    echo "❌ Instance creation failed"
    echo "Zone may have become unavailable - try running again"
    exit 1
fi

echo ""
echo "✅ Instance created successfully!"
echo ""

# Save configuration
cat > .gcp_config << EOF
PROJECT_ID=$PROJECT_ID
INSTANCE_NAME=$INSTANCE_NAME
ZONE=$ZONE
MACHINE_TYPE=$MACHINE_TYPE
GPU_TYPE=$GPU_TYPE
PREEMPTIBLE=$USE_PREEMPTIBLE
EOF

echo "⏳ Waiting 60 seconds for instance to initialize..."
sleep 60

# Setup environment
echo ""
echo "🔧 Setting up training environment..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
    set -e
    cd ~
    
    # Create venv
    python3 -m venv agentzero_env
    source agentzero_env/bin/activate
    
    # Install dependencies
    pip install --upgrade pip -q
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    pip install transformers datasets peft accelerate -q
    pip install scikit-learn pandas numpy requests beautifulsoup4 -q
    pip install wandb python-dotenv -q
    
    # Verify GPU
    python3 << 'PYEOF'
import torch
print('\\n🔍 GPU Check:')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('\\n✅ Environment ready!')
PYEOF
" 2>/dev/null || echo "⚠️  Setup incomplete - you may need to run setup manually"

# Final instructions
echo ""
echo "🎉 agentZer0 Training Environment Ready!"
echo "========================================"
echo ""
echo "📊 Configuration:"
echo "   Zone: $ZONE"
echo "   GPU: $GPU_TYPE"
if [ "$USE_PREEMPTIBLE" = true ]; then
    echo "   Type: Preemptible"
    echo "   Cost: ~\$0.10/hour (~\$1 for 10 hours)"
else
    echo "   Type: Standard"
    echo "   Cost: ~\$0.35/hour (~\$3.50 for 10 hours)"
fi
echo ""
echo "🔗 Connect:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "🎯 Quick Start:"
echo "   1. SSH in (command above)"
echo "   2. cd ~ && source agentzero_env/bin/activate"
echo "   3. Upload your code: gcloud compute scp --recurse /local/path $INSTANCE_NAME:~/ --zone=$ZONE"
echo "   4. Run training: python scripts/train_model_fixed.py"
echo ""
if [ "$USE_PREEMPTIBLE" = true ]; then
    echo "⚠️  PREEMPTIBLE INSTANCE NOTES:"
    echo "   • Can be stopped by Google anytime (rare but possible)"
    echo "   • Save checkpoints every 100 steps"
    echo "   • Monitor with: watch -n 60 'gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format=\"get(status)\"'"
    echo ""
fi
echo "⚠️  REMEMBER TO STOP WHEN DONE:"
echo "   gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Current cost: \$$([ "$USE_PREEMPTIBLE" = true ] && echo "0.10" || echo "0.35")/hour while running"