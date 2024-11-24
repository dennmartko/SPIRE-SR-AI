#!/bin/bash

# Root directory
ROOT_DIR="/mnt/d/SPIRE-SR-AI"

# Create the root directory
mkdir -p $ROOT_DIR

# Define subdirectories
SUBDIRS=(
  "data/raw"
  "data/processed"
  "data/test_samples"
  "configs"             # For JSON parameter files
  "scripts/train"
  "scripts/preprocess"
  "scripts/evaluate"
  "scripts/finetune"    # Fine-tuning scripts
  "scripts/utils"
  "models/architectures"
  "models/checkpoints"
  "results/model_1"
  "results/model_2"
  "results/comparisons"
  "dev/notebooks"
  "dev/prototypes"
  "docs"
  "logs"                # For logs from training/fine-tuning
  "images/training"
  "images/results"
)

# Create subdirectories
for DIR in "${SUBDIRS[@]}"; do
  mkdir -p "$ROOT_DIR/$DIR"
done

# Create placeholder files
touch "$ROOT_DIR/docs/README.md"
touch "$ROOT_DIR/docs/usage_guide.md"

echo "Folder structure created successfully in $ROOT_DIR"
