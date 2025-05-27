#!/bin/bash
#
#SBATCH --job-name=tf_install         # job name
#SBATCH --output=tf_install_%j.out    # STDOUT (%j = jobID)
#SBATCH --error=tf_install_%j.err     # STDERR
#SBATCH --time=01:00:00               # hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G                     # RAM per node
#SBATCH --begin=now

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo "Loading modules..."

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0             # adjust GCC as needed
module load CUDA/12.6.0               # adjust CUDA version
module load cuDNN/9.5.0.50-CUDA-12.6.0               # adjust cuDNN version


PYENV_DIR=$HOME/.venvs/tf_cuda_xla
source ${PYENV_DIR}/bin/activate

echo "Upgrading pip and installing TensorFlow..."
pip install --upgrade pip
# You can pin a specific version, e.g. tensorflow==2.14.0
pip install tensorflow==2.16.2
pip install tf-keras==2.16.0

echo "Configuring XLA..."
# Enable XLA devices and JIT compilation
export TF_XLA_FLAGS="--tf_xla_enable_xla_devices"
export TF_ENABLE_XLA="1"
echo "Verifying installation..."
python - <<'EOF'
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

# Simple XLA test: this function will be JIT‑compiled
@tf.function(jit_compile=True)
def add(a, b):
    return a + b

result = add(tf.constant(1), tf.constant(2))
print("XLA add(1,2) =", result.numpy())
EOF

echo "Deactivating environment and exiting."
deactivate
echo "Job finished at $(date)