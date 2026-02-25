#!/bin/bash

# Accept command-line arguments or use defaults
ENTITY="${1:-backyard}"
DATASET="${2:-sd21}"
PROJECT_ID="${3:-proj_id}"  # set your own project id here
VQA="${4:-flash}"
BATCH="${5:-1}"

echo "Running VDI Background Appearance analysis for:"
echo "  Entity: $ENTITY"
echo "  Dataset: $DATASET"
echo "  VQA Model: $VQA"
echo "  Project ID: $PROJECT_ID"
echo "  Batch Processing: $BATCH"
echo ""

# Background visibility check
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step1' --proj_id "$PROJECT_ID" --batch "$BATCH" --axis background
python utils/background_prep.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step2'
# Indoor vs Outdoor check
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step2' --proj_id "$PROJECT_ID" --batch "$BATCH" --axis background
python utils/background_prep.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step3'
# Visibility Questions based on indoor vs outdoor classification
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step3' --proj_id "$PROJECT_ID" --batch "$BATCH" --axis background
python utils/background_prep.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step4'
# Final VQA questions
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'step4' --proj_id "$PROJECT_ID" --batch "$BATCH" --axis background
python utils/background_prep.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'final'
# Diversity score calculation
python analysis_vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --axis 'background' --save_plots