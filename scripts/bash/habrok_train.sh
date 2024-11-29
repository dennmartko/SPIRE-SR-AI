#!/bin/bash

#SBATCH --job-name=Training
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=D.Koopmans@sron.nl
#SBATCH --output=job-%j.log

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --begin=now

# Model name
MODEL_NAME="UnetResnet34Tr"

# Config file
CONFIG_FILE="TrainConfig.yaml"

# Directory to store configuration files
CONFIG_DIR="/scratch/p317470/SPIRE-SR-AI/configs/${MODEL_NAME}/${CONFIG_FILE}"

# Snellius
# module load 2022
# module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0


# HB
module load Python/3.9.6-GCCcore-11.2.0
module load TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1

python "/scratch/p317470/SPIRE-SR-AI/scripts/train/train.py" -config  "${CONFIG_DIR}"