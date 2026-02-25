#!/bin/bash

# This script uploads images from a local directory to a Google Cloud Storage bucket, maintaining the directory structure.
# It assumes that the Google Cloud SDK is installed and configured on your machine.
# Usage:
# 1. Set the CLASS_NAME environment variable to specify the subdirectory to upload.
# export CLASS_NAME="backyard"
# 2. Run the script: sh upload_gcloud.sh


# First, create the buckets from terminal using:
# Use suffix _gcs to avoid conflict with existing buckets (this will be expected in utils.py)
# gcloud storage buckets create gs://sd21_gcs --project=proj_id --default-storage-class=COLDLINE --location=us-central1 --uniform-bucket-level-access --enable-hierarchical-namespace


# Directory to read files from
TARGET_DIR="../images/sd21/$CLASS_NAME/" 
# Directory to upload and save files to
GCS_BUCKET="gs://sd21_gcs/$CLASS_NAME/"

gcloud storage folders create --recursive $GCS_BUCKET

# Check if the directory exists
if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Directory $TARGET_DIR does not exist."
  exit 1
fi

for SUBDIR in "$TARGET_DIR"*/; do
  # Ensure it's a directory
  if [ -d "$SUBDIR" ]; then
    # Extract the folder name (e.g., c1, c2)
    SUBDIR_NAME=$(basename "$SUBDIR")

    echo "Creating folder in GCS: $GCS_BUCKET$SUBDIR_NAME/"
    gcloud storage folders create "$GCS_BUCKET$SUBDIR_NAME/"

    # Loop through images in the subdirectory and upload them
    for FILE in "$SUBDIR"*; do
      if [ -f "$FILE" ]; then
        echo "Uploading $FILE to $GCS_BUCKET$SUBDIR_NAME/"
        gcloud storage cp "$FILE" "$GCS_BUCKET$SUBDIR_NAME/"
      fi
    done

    echo "Finished processing folder: $SUBDIR_NAME"
    echo "----------------------------------------"
  fi
done


echo "All files processed."
