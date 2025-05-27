#!/bin/bash
#SBATCH --job-name=Training_new_alphamodel
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=d.koopmans@sron.nl
#SBATCH --output=/home/dkoopmans/SPIRE-SR-AI/logs/UnetResnet34Tr/Trainingn-new-alphamodel-job-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=120:00:00
#SBATCH --begin=now

# Model name and configuration
MODEL_NAME="UnetResnet34Tr"
CONFIG_FILE="TrainConfig_new.yaml"
CONFIG_DIR="/home/dkoopmans/SPIRE-SR-AI/configs/${MODEL_NAME}/${CONFIG_FILE}"

# Load modules and environment
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Activate Python virtual environment containing required packages
PYENV_DIR="$HOME/.venvs/tf_cuda_xla"
source "${PYENV_DIR}/bin/activate"

# Load legacy settings from .bashrc if needed (for legacy Keras)
source ~/.bashrc

# Locate CUDA installation directory and set XLA flags
CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_ROOT}"
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
export TF_ENABLE_XLA=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Run training script
python "/home/dkoopmans/SPIRE-SR-AI/scripts/train/train_new.py" --config "${CONFIG_DIR}"