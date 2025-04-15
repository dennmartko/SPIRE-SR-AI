#!/bin/bash

# Model name
MODEL_NAME="UnetResnet34Tr"

# Evaluation script
# EVALUATION_SCRIPT="/scratch/p317470/SPIRE-SR-AI/scripts/evaluate/evaluate_target_simulations.py"
    
EVALUATION_SCRIPT="/mnt/d/SPIRE-SR-AI/scripts/evaluate/evaluate_target_simulations.py"

# Define the IDs for evaluation
ID_LIST=(1 2 3 4 5 6 7)  # Uncomment to dynamically create run_names
#RUN_NAMES=("new_finetune_run_7")  # Uncomment for predefined run names without ID

# Generate RUN_NAMES dynamically using ID_LIST
RUN_NAMES=()
for ID in "${ID_LIST[@]}"; do
    RUN_NAMES+=("second_finetune_run_${ID}")
done

# Loop over all RUN_NAMES
for RUN_NAME in "${RUN_NAMES[@]}"; do

    # Create YAML content in memory
    YAML_CONFIG=$(cat <<EOL
model:
  # Model specific settings, do not change!
  model: "${MODEL_NAME}"  # Should be the same name as parent config folder
  run_name: "${RUN_NAME}"  # Run name
  input_shape: [256, 256, 4]
  output_shape: [256, 256, 1]

sys_config:
  n_cpu_cores: 12 # Number of CPU cores to use in parallel processing

data:
  # Data specific settings, change to needs.
  test_dataset_path: "/mnt/d/SPIRE-SR-AI/data/processed/50deg_shark_sides_spritz/Test"
  input: ["24", "250", "350", "500"]  # class names
  target: ["500SR"]  # class names
EOL
)

    # Execute the Python script by piping YAML content
    echo "${YAML_CONFIG}" | python3 ${EVALUATION_SCRIPT} --config /dev/stdin

    echo "Executed evaluation script for run name: $RUN_NAME"
done