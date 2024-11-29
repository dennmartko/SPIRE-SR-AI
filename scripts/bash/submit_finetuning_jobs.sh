#!/bin/bash

# Note that the module wrap is for Habrok!

# Model name
MODEL_NAME="UnetResnet34Tr"

# Directory to store configuration files
CONFIG_DIR="/scratch/p317470/SPIRE-SR-AI/configs/${MODEL_NAME}"
TRAIN_SCRIPT="/scratch/p317470/SPIRE-SR-AI/scripts/train/train.py"

# Create the config directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Hyperparameter options (for tuning)
ALPHAS=(0 0.1 0.25 0.5 0.75 0.999 1.0)
BATCH_SIZE=48
INITIAL_LR=0.01
END_LR=0.0001
DECAY_EPOCHS=200

# Counter for unique run names
COUNTER=1

# Loop over all ALPHA values
for ALPHA in "${ALPHAS[@]}"; do
    # Generate a unique run name
    # Should not depend on time, as training restarts are impossible
    RUN_NAME="finetune_run_${COUNTER}"
    
    # Define the config file name
    CONFIG_FILE="$CONFIG_DIR/${RUN_NAME}.yaml"

    # Write the YAML configuration
    cat > "$CONFIG_FILE" <<EOL
model:
  # Model specific settings, do not change!
  model: "${MODEL_NAME}"  # Should be the same name as parent config folder
  run_name: "${RUN_NAME}"  # Such that the run can be recognized, i.e., start date
  input_shape: [256, 256, 4]
  output_shape: [256, 256, 1]

training:
  polynomial_lr_schedule: [${INITIAL_LR}, ${END_LR}, ${DECAY_EPOCHS}, 2]  # initial lr, end lr, epochs, power
  batch_size: ${BATCH_SIZE}
  number_of_epochs: 1000  # Maximum number of epochs
  patience: 100  # After how many epochs training stops with no improvement in val loss
  alpha: ${ALPHA} # hyperparameter for the losses

data:
  # Data specific settings, change to needs.
  data_path: "/mnt/d/SPIRE-SR-AI/data/processed/50deg_shark_sides_spritz"
  input: ["24", "250", "350", "500"]  # class names
  target: ["500SR"]  # class names
EOL

    echo "Created config file: $CONFIG_FILE"

    # Submit the job to Slurm directly
    sbatch --job-name="${RUN_NAME}" \
           --mail-type=BEGIN,END \
           --output="${CONFIG_DIR}/${RUN_NAME}-%j.log" \
           --nodes=1 \
           --ntasks=1 \
           --partition=gpu \
           --gpus-per-node=a100:1 \
           --cpus-per-task=16 \
           --mem=80G \
           --time=24:00:00 \
           --wrap="module load Python/3.9.6-GCCcore-11.2.0 && \
                   module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1 && \
                   python ${TRAIN_SCRIPT} --config ${CONFIG_FILE}"
    
    echo "Submitted job for config: $CONFIG_FILE"

    # Increment counter
    COUNTER=$((COUNTER + 1))
done
