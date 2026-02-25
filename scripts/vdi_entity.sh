#!/bin/bash

# Accept command-line arguments or use defaults
ENTITY="${1:-backyard}"
DATASET="${2:-sd21}"
PROJECT_ID="${3:-proj_id}"  # set your own project id here
VQA="${4:-flash}"
BATCH="${5:-1}"

echo "Running VDI Entity Appearance analysis for:"
echo "  Entity: $ENTITY"
echo "  Dataset: $DATASET"
echo "  VQA Model: $VQA"
echo "  Batch Processing: $BATCH"
echo ""

python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'visibility' --proj_id "$PROJECT_ID" --batch "$BATCH"
# run the following to filter out questions which passed visibility test in previous step, replace them with original questions (with multiple answers choice possible) and prepare the csv for multiple choice VQA (vqa_NF):
python utils/NF_prep.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" 
#now VQA for NF flagged questions:
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'vqa_NF' --proj_id "$PROJECT_ID" --batch "$BATCH"
#now VQA for F flagged questions:
python vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --step_name 'vqa_F' --proj_id "$PROJECT_ID" --batch "$BATCH"
python analysis_vdi.py --vqa_model "$VQA" --entity_name "$ENTITY" --dataset_name "$DATASET" --axis 'entity' --save_plots