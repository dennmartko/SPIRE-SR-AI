#!/bin/bash

#SBATCH --job-name=Training
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=d.koopmans@sron.nl
#SBATCH --output=job-%j.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=120:00:00
#SBATCH --begin=now

# Model name
MODEL_NAME="UnetResnet34Tr"

# Config file
CONFIG_FILE="TrainConfig.yaml"

# Directory to configuration files
CONFIG_DIR="/home/dkoopmans/SPIRE-SR-AI/configs/${MODEL_NAME}/${CONFIG_FILE}"

# Snellius
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

python "/home/dkoopmans/SPIRE-SR-AI/scripts/train/train.py" -config  "${CONFIG_DIR}"