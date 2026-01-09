#!/bin/bash
# Upload the latest checkpoint and config to Hugging Face Hub, excluding optimizer state.
# usage: bash upload_checkpoint.sh <user_name/repo_name> <folder_path>

# Check if correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <user_name/repo_name> <folder_path>"
    exit 1
fi

REPO_ID=$1
BASE_DIR=$2

# Ensure the base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' does not exist."
    exit 1
fi

echo "🔍 Scanning $BASE_DIR for the latest checkpoint..."

# Find the checkpoint directory with the highest number
LATEST_CHECKPOINT=$(ls -1d "$BASE_DIR"/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No 'checkpoint_*' folders found in $BASE_DIR"
    exit 1
fi

CHECKPOINT_NAME=$(basename "$LATEST_CHECKPOINT")
echo "✅ Found latest checkpoint: $CHECKPOINT_NAME"

echo "🚀 Uploading YAML configs..."
hf upload "$REPO_ID" "$BASE_DIR" . --include "*.yaml"

echo "🚀 Uploading $CHECKPOINT_NAME (excluding optimizer state)..."
# Added --exclude flag to skip the optimizer file
hf upload "$REPO_ID" "$LATEST_CHECKPOINT" "$CHECKPOINT_NAME" --exclude "optimizer.pt"

echo "✨ Done! Uploaded to https://huggingface.co/$REPO_ID"