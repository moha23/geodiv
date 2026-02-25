#!/bin/bash

# Usage: bash run.sh <ENTITY> <DATASET> <PROJECT_ID> [VQA_MODEL]
# Example: bash run.sh "house" "sd21" "my-project-id" "flash"

if [ $# -lt 3 ]; then
    echo "Usage: bash run.sh <ENTITY> <DATASET> <PROJECT_ID> [VQA_MODEL] [BATCH]"
    echo "Example: bash run.sh \"house\" \"sd21\" \"my-project-id\" \"flash\" 1"
    exit 1
fi

ENTITY="$1"
DATASET="$2"
PROJECT_ID="$3"
VQA="${4:-flash}"  # Default to flash if not provided
BATCH="${5:-1}"  # Default to 1 if not provided

echo "Running GeoDiv pipeline with:"
echo "  Entity: $ENTITY"
echo "  Dataset: $DATASET"
echo "  Project ID: $PROJECT_ID"
echo "  VQA Model: $VQA"
echo "  Batch Processing: $BATCH"
echo ""

bash scripts/vdi_entity.sh "$ENTITY" "$DATASET" "$PROJECT_ID" "$VQA" "$BATCH"
bash scripts/vdi_background.sh "$ENTITY" "$DATASET" "$PROJECT_ID" "$VQA" "$BATCH"
bash scripts/sevi.sh "$ENTITY" "$DATASET" "$PROJECT_ID" "$VQA" "$BATCH"