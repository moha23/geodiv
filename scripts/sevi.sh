#!/bin/bash

# Accept command-line arguments or use defaults
ENTITY="${1:-backyard}"
DATASET="${2:-sd21}"
PROJECT_ID="${3:-proj_id}"  # set your own project id here
VQA="${4:-flash}"
BATCH="${5:-1}"

echo "Running SEVI (Affluence + Maintenance) analysis for:"
echo "  Entity: $ENTITY"
echo "  Dataset: $DATASET"
echo "  VQA Model: $VQA"
echo "  Project ID: $PROJECT_ID"
echo "  Batch Processing: $BATCH"
echo ""

# Flash using batch processing
BATCH="${5:-1}"
python sevi.py --vqa_model "$VQA" --axis 'affluence' --entity_name "$ENTITY" --dataset_name "$DATASET" --prompts_file_path 'country_prompts' --proj_id "$PROJECT_ID" --batch "$BATCH" --gen_img_path "images/$DATASET"
python sevi.py --vqa_model "$VQA" --axis 'maintenance' --entity_name "$ENTITY" --dataset_name "$DATASET" --prompts_file_path 'country_prompts' --proj_id "$PROJECT_ID" --batch "$BATCH" --gen_img_path "images/$DATASET"
python analysis_sevi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --axis 'affluence' --save_plots
python analysis_sevi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --axis 'maintenance' --save_plots

# Qwen (batch processing not supported)
# VQA="qwen"
# python sevi.py --vqa_model "$VQA" --axis 'affluence' --entity_name "$ENTITY" --dataset_name "$DATASET" --prompts_file_path 'country_prompts' --proj_id "$PROJECT_ID" --batch "$BATCH" --gen_img_path "images/$DATASET"
# python sevi.py --vqa_model "$VQA" --axis 'maintenance' --entity_name "$ENTITY" --dataset_name "$DATASET" --prompts_file_path 'country_prompts' --proj_id "$PROJECT_ID" --batch "$BATCH" --gen_img_path "images/$DATASET"

