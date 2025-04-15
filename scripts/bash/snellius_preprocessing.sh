#!/bin/bash
#SBATCH --job-name=PROCESSING
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=dmkoopmans@astro.rug.nl
#SBATCH --output=job-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --time=6:00:00

set -euo pipefail

# Paths
SR_LIB_PATH="/home/dkoopmans/SPIRE-SR-AI"
DATAMAPS_DIR="${SR_LIB_PATH}/data/raw/sim_datamaps"
DATA_DIR="${SR_LIB_PATH}/data/processed"
DEST_DIR="/scratch-shared/$USER"

# Create scratch directory
mkdir -p "$DEST_DIR"

check_dir() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        echo "Error: Directory does not exist: $dir"
        exit 1
    fi
}

copy_directory() {
    local src="$1"
    local dst="$2"
    cp -r "$src" "$dst"
}

# Check directories exist
check_dir "$DATAMAPS_DIR"
check_dir "$DEST_DIR"

# Copy datamaps directory to scratch shared filesystem
if copy_directory "$DATAMAPS_DIR" "$DEST_DIR"; then
    echo "Directory copied successfully to scratch-shared filesystem."
else
    echo "Failed to copy directory to scratch-shared filesystem." >&2
    exit 1
fi

module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# Run preprocessing scripts
python "$SR_LIB_PATH/scripts/preprocess/datasets/gen_sim_data.py"
python "$SR_LIB_PATH/scripts/preprocess/datasets/DataMerge.py"

# Copy dataset to HOME processed directory
cp -r "$DEST_DIR"/Dataset* "$DATA_DIR"
