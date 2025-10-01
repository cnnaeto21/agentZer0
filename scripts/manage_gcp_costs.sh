#!/bin/bash

# agentZer0 GCP Cost Management
# Helps manage costs while training to improve from 50% false positive baseline

set -e

# Load configuration
if [ -f .gcp_config ]; then
    source .gcp_config
else
    # Default values
    PROJECT_ID="agentzero-prompt-security"
    INSTANCE_NAME="agentzero-trainer"
    ACTIVE_ZONE="us-central1-a"
fi

ACTION=${1:-"help"}

show_help() {
    echo "🛡️  agentZer0 GCP Cost Management"
    echo "================================="
    echo ""
    echo "Usage: $0 <action>"
    echo ""
    echo "Actions:"
    echo "  status    - Check instance status and costs"
    echo "  start     - Start the training instance"
    echo "  stop      - Stop the training instance (saves money)"
    echo "  connect   - SSH into the training instance"
    echo "  costs     - Show estimated daily/monthly costs"
    echo "  delete    - Delete instance (saves maximum money)"
    echo "  monitor   - Set up cost monitoring alerts"
    echo ""
    echo "🎯 Project Goal: Reduce false positives from 50% to <10%"
    echo "💰 T4 GPU costs ~$1.50/hour, only pay when running"
}

check_status() {
    echo "🔍 Checking agentZer0 instance status..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --quiet >/dev/null 2>&1; then
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(status)")
        MACHINE_TYPE=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(machineType)" | cut -d'/' -f10)
        
        echo "📍 Instance: $INSTANCE_NAME"
        echo "🌍 Zone: $ACTIVE_ZONE"
        echo "📊 Status: $STATUS"
        echo "💻 Machine: $MACHINE_TYPE"
        
        if [ "$STATUS" = "RUNNING" ]; then
            echo "💰 Currently incurring costs (~$1.50-3.00/hour)"
            echo "⏰ Running time: $(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(lastStartTimestamp)")"
        else
            echo "💚 Not incurring compute costs (stopped)"
        fi
    else
        echo "❌ Instance $INSTANCE_NAME not found in $ACTIVE_ZONE"
        echo "💡 Run ./scripts/setup_gcp_training.sh to create it"
    fi
}

start_instance() {
    echo "🚀 Starting agentZer0 training instance..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --quiet >/dev/null 2>&1; then
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(status)")
        
        if [ "$STATUS" = "RUNNING" ]; then
            echo "✅ Instance already running"
        else
            echo "⏳ Starting instance..."
            gcloud compute instances start $INSTANCE_NAME --zone=$ACTIVE_ZONE
            echo "✅ Instance started"
            echo "💰 Now incurring costs (~$1.50-3.00/hour)"
        fi
        
        echo ""
        echo "🔗 Connect with:"
        echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ACTIVE_ZONE"
    else
        echo "❌ Instance not found. Create it first:"
        echo "   ./scripts/setup_gcp_training.sh"
    fi
}

stop_instance() {
    echo "⏹️  Stopping agentZer0 training instance..."
    
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --quiet >/dev/null 2>&1; then
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(status)")
        
        if [ "$STATUS" = "TERMINATED" ]; then
            echo "✅ Instance already stopped"
        else
            echo "⏳ Stopping instance..."
            gcloud compute instances stop $INSTANCE_NAME --zone=$ACTIVE_ZONE
            echo "✅ Instance stopped"
            echo "💚 No longer incurring compute costs"
            echo "📁 Disk storage still costs ~$0.05/day"
        fi
    else
        echo "❌ Instance not found"
    fi
}

connect_instance() {
    echo "🔗 Connecting to agentZer0 training instance..."
    
    # Check if instance is running
    STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ACTIVE_ZONE --format="get(status)" 2>/dev/null || echo "NOT_FOUND")
    
    if [ "$STATUS" = "NOT_FOUND" ]; then
        echo "❌ Instance not found. Create it first:"
        echo "   ./scripts/setup_gcp_training.sh"
        exit 1
    elif [ "$STATUS" != "RUNNING" ]; then
        echo "⚠️  Instance is $STATUS. Starting it..."
        start_instance
        echo "⏳ Waiting for startup..."
        sleep 30
    fi
    
    echo "🚪 Opening SSH connection..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ACTIVE_ZONE
}

show_costs() {
    echo "💰 agentZer0 Training Cost Estimates"
    echo "===================================="
    echo ""
    echo "🖥️  Compute Costs (when running):"
    echo "   T4 GPU instance:  ~$1.50/hour"
    echo "   V100 GPU instance: ~$3.00/hour"
    echo ""
    echo "💾 Storage Costs (always incurred):"
    echo "   200GB disk: ~$0.05/day (~$1.50/month)"
    echo ""
    echo "📊 Training Estimates:"
    echo "   1 hour training session: $1.50-3.00"
    echo "   Full day development: $36-72"
    echo "   Monthly storage: $1.50"
    echo ""
    echo "💡 Cost Optimization Tips:"
    echo "   • Stop instance when not training"
    echo "   • Use preemptible instances for long training"
    echo "   • Delete instance if not needed for weeks"
    echo "   • Monitor with billing alerts"
    echo ""
    echo "🎯 Goal: Fix 50% false positive issue efficiently"
}

delete_instance() {
    echo "⚠️  DELETE agentZer0 training instance?"
    echo "======================================"
    echo ""
    echo "This will:"
    echo "• Delete the instance completely"
    echo "• Delete all data on the instance"
    echo "• Stop all costs immediately"
    echo "• Require re-running setup script to recreate"
    echo ""
    read -p "Are you sure? Type 'DELETE' to confirm: " confirmation
    
    if [ "$confirmation" = "DELETE" ]; then
        echo "🗑️  Deleting instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ACTIVE_ZONE --quiet
        echo "✅ Instance deleted"
        echo "💚 All costs stopped"
        echo ""
        echo "💡 To recreate: ./scripts/setup_gcp_training.sh"
    else
        echo "❌ Deletion cancelled"
    fi
}

setup_monitoring() {
    echo "📊 Setting up cost monitoring for agentZer0..."
    
    # Create a simple billing alert
    echo "💡 Setting up billing alerts..."
    echo ""
    echo "Manual steps to set up cost monitoring:"
    echo "1. Go to: https://console.cloud.google.com/billing/budgets"
    echo "2. Create budget for project: $PROJECT_ID"
    echo "3. Set monthly budget limit (e.g., $50-100)"
    echo "4. Set alerts at 50%, 80%, 100%"
    echo "5. Add your email for notifications"
    echo ""
    echo "🎯 Recommended budget: $100/month for agentZer0 development"
}

# Main command dispatcher
case $ACTION in
    "status")
        check_status
        ;;
    "start")
        start_instance
        ;;
    "stop")
        stop_instance
        ;;
    "connect")
        connect_instance
        ;;
    "costs")
        show_costs
        ;;
    "delete")
        delete_instance
        ;;
    "monitor")
        setup_monitoring
        ;;
    "help"|*)
        show_help
        ;;
esac