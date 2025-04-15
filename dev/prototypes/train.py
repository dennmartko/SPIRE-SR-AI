import os
import sys
import numpy as np
import tensorflow as tf
from astropy.io import fits
from matplotlib import pyplot as plt

from scripts.utils.data_loader import create_dataset, generate_interpolated_noise, extract_patches, reconstruct_from_patches
from scripts.utils.file_utils import get_main_dir, create_model_ckpt_folder, create_log_file, printlog, log_epoch_details, load_training_history, save_training_history, create_model_results_subfolder
from scripts.utils.plots import data_debug_plot, display_predictions, plot_history

from models.architectures.UnetResnet34Tr import UnetResnet34Tr
from models.architectures.SwinUnet import swin_unet_2d_base
from models.architectures.Unet import build_unet 
from loss_functions import non_adversarial_loss

import yaml
import argparse
import datetime

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to config file")
args = parser.parse_args()

# Load the YAML file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Main directory
DS_DIR = config["data"]["data_path"]

# Parameters
BATCH_SIZE = config["training"]["batch_size"]

# Define the mapping function for inputs to targets
target_class_names = config["data"]["target"]
input_class_names = config["data"]["input"]

n_targets = len(target_class_names)
# Datasets
TRAIN_DIR = os.path.join(DS_DIR, "Train")
VAL_DIR = os.path.join(DS_DIR, "Validation")

train_ds, train_num_batches = create_dataset(TRAIN_DIR, input_class_names, target_class_names, BATCH_SIZE, is_training=True)
val_ds, val_num_batches = create_dataset(VAL_DIR, input_class_names, target_class_names, BATCH_SIZE*2, is_training=False) # Inference batch_size can always be much larger

# Define parameters
interp_params = {
    '24': {'pixscale': 1.2, 'noise_lvl': 14e-6},
    '250': {'pixscale': 6.0, 'noise_lvl': 2e-3},
    '350': {'pixscale': 8.33, 'noise_lvl': 2e-3},
    '500': {'pixscale': 12.0, 'noise_lvl': 2e-3},
}

# Generate interpolated noise for the input images. Uses (safe) multiprocessing
interpolated_noise = generate_interpolated_noise(input_class_names, interp_params, samples=50, img_size=256)

def add_noise(input_class_names, interpolated_noise, batch):
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)  # Convert batch to a TensorFlow tensor

    for i, name in enumerate(input_class_names):
        # Generate random indices for noise sampling
        rnd_indices = np.random.randint(0, interpolated_noise[name].shape[0], batch.shape[0])
        
        # Fetch noise and convert to TensorFlow tensor
        noise = tf.convert_to_tensor(interpolated_noise[name][rnd_indices], dtype=tf.float32)
        
        # Add noise to the respective channel without in-place modification
        batch = tf.concat([batch[:, :, :, :i], 
                           batch[:, :, :, i:i+1] + noise[..., None], 
                           batch[:, :, :, i+1:]], axis=-1)
    
    return batch

# # Normalizer
# train_ds_norm = train_ds.map(lambda x, y: x)
# normalizer = tf.keras.layers.Normalization()
# normalizer.adapt(train_ds_norm)

# train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalizer(x), y))

# Plot an image sample for debugging purposes
inputs, labels = next(iter(val_ds.take(1)))

inputs = add_noise(input_class_names, interpolated_noise, inputs)
inputs2 = add_noise(input_class_names, interpolated_noise, np.zeros_like(inputs))

save = os.path.join(get_main_dir(), "images/data_samples.jpg")
save2 = os.path.join(get_main_dir(), "images/noise_samples.jpg")
data_debug_plot(inputs, labels, input_class_names, target_class_names, save)
data_debug_plot(inputs2, labels, input_class_names, target_class_names, save2)

# Get model
model_name = config["model"]["model"]
run_name = config["model"]["run_name"]
model_weights_path, first_run = create_model_ckpt_folder(model_name, run_name)
input_shape = config["model"]["input_shape"]
output_shape = config["model"]["output_shape"]
lr_params = config["training"]["polynomial_lr_schedule"]
if model_name == "UnetResnet34Tr":
    model = UnetResnet34Tr(tuple(input_shape), "channels_last")
elif model_name == "SwinUnet":
    filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2         # number of Swin Transformers per downsampling level
    stack_num_up = 2           # number of Swin Transformers per upsampling level
    patch_size = (8, 8)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    num_heads = [2, 4, 4, 4]   # number of attention heads per down/upsampling level
    window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
    num_mlp = 512              # number of MLP nodes within the Transformer
    shift_window=True          # Apply window shifting, i.e., Swin-MSA
    model = swin_unet_2d_base(tuple(input_shape), filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, 
                      shift_window=shift_window, name='swin_unet')
elif model_name == "Unet":
    model = build_unet(tuple(input_shape), "channels_last")
else:
    raise("Other models not yet implemented")

# Logging
log_file = create_log_file(model_name, run_name, first_run)

# Create results directory for the model type as well as the run if they do not exist
training_results_dir = create_model_results_subfolder(model_name=model_name, run_name=run_name, purpose="training")

## Declare completion of initialisation
s = f"{'-' * 24}\nModel Training Log\nDate: {datetime.datetime.now()}\n{'-' * 24}"
printlog(s, log_file)

printlog(f"{datetime.datetime.now()} - Datasets prepared sucessfully!", log_file)
printlog(f"{datetime.datetime.now()} - Number of Training batches: %d" % train_num_batches, log_file)
printlog(f"{datetime.datetime.now()} - Number of Validation batches: %d" % val_num_batches, log_file)

# Get optimizer
lr_fn = tf.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=lr_params[0],
    decay_steps=int(lr_params[2] * train_num_batches),
    end_learning_rate=lr_params[1],
    power=lr_params[3]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn, beta_1=0.9)

# Checkpointing
ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager_bestmodel = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "BestModel"), max_to_keep=1)
manager_ckpt = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "Checkpoint"), max_to_keep=1)

training_history_file = os.path.join(model_weights_path, "history.json")

if not first_run:
    manager_ckpt.restore_or_initialize()
    printlog(f"{datetime.datetime.now()} - Found checkpoint folder: {model_weights_path}!", log_file)
    printlog(f"{datetime.datetime.now()} - Restored previous model checkpoint!", log_file)

    # Get history
    history = load_training_history(training_history_file)

else:
    model.summary(print_fn=lambda x: printlog(x, log_file))
    history = {'epochs': [], 'train_loss': [], 'val_loss': [], 'patience': []}

    # Write history file to disk
    # This ensures that training can always be restarted.
    save_training_history(history, training_history_file)

# Training loop
best_val_loss = np.min(history["val_loss"]) if history["val_loss"] else float('inf')
epochs_without_improvement = 0 if not history["patience"] else history["patience"][-1]
patience = config["training"]["patience"]
num_epochs = config["training"]["number_of_epochs"]
start_epoch = len(history["epochs"]) + 1

# Early stopping (based on patience)
if epochs_without_improvement >= patience:
    printlog(f"{datetime.datetime.now()} - Early stopping after epoch {start_epoch}. No improvement for {patience} epochs.", log_file)
    epochs = [] # We do not go into the training loop!
else:
    epochs = np.arange(start_epoch, num_epochs + 1, 1)

# hyperparameter for the losses
alpha = config["training"]["alpha"]

# @tf.function(jit_compile=True)
# def train_step(x, y, masks, alpha, model, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(x, training=True)
#         train_loss = non_adversarial_loss(predictions, y, masks, alpha)
#     grads = tape.gradient(train_loss, model.trainable_variables)
#     # grads, _ = tf.clip_by_global_norm(grads, 10.0)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return train_loss, tf.linalg.global_norm(grads)

@tf.function(jit_compile=True)
def train_step(x, y, masks, alpha, model, optimizer, lambda_reg=1e-4):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        train_loss = non_adversarial_loss(predictions, y, masks, alpha)
        
        # Compute L2 regularization loss
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
        
        # Add L2 loss to the training loss
        total_loss = train_loss + lambda_reg * l2_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    # grads, _ = tf.clip_by_global_norm(grads, 10.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return train_loss, tf.linalg.global_norm(grads)


for epoch in tqdm(epochs, desc="Training model..."):
    total_train_loss = 0
    total_batch_norm = 0
    for idx, (batch_x, batch_y) in enumerate(train_ds):
        train_batch_loss, batch_norm = train_step(batch_x, batch_y[:, :, :, :n_targets], batch_y[:, :, :, n_targets:], alpha, model, optimizer)
        total_train_loss += train_batch_loss.numpy()
        total_batch_norm += batch_norm.numpy()
    # Compute average training loss for the epoch
    avg_train_loss = total_train_loss / train_num_batches.numpy()
    avg_grad_norm = total_batch_norm / train_num_batches.numpy()

    # Validation loop
    total_val_loss = 0
    for batch_x, batch_y in val_ds:
        predictions = model(batch_x, training=False)
        val_loss = non_adversarial_loss(predictions, batch_y[:, :, :, :n_targets], batch_y[:, :, :, n_targets:], alpha)
        total_val_loss += val_loss.numpy()

    avg_val_loss = total_val_loss / val_num_batches.numpy()

    # Check if the model improved
    improved = avg_val_loss < best_val_loss
    if improved:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        manager_bestmodel.save()
    else:
        epochs_without_improvement+=1;

    log_epoch_details(epoch, avg_train_loss, avg_val_loss, avg_grad_norm, improved, log_file)

    # Update the history
    history["epochs"].append(epoch)
    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["patience"].append(epochs_without_improvement)

    # Checkpointing, history saving and display of training progress
    if epoch % 10 == 0:
        manager_ckpt.save()
        save_training_history(history, training_history_file)
        batch_x, batch_y = next(iter(val_ds.take(1)))
        predictions = model(batch_x, training=False)
        save = os.path.join(training_results_dir, f"epoch_{epoch}_valloss_{avg_val_loss}.jpg")
        display_predictions(batch_x, batch_y[:, :, :, :n_targets], predictions, input_class_names, target_class_names, save)
        plot_history(history, os.path.join(training_results_dir, f"Training_History.jpg"))
        printlog(f"{datetime.datetime.now()} - Created model checkpoint & saved training history!", log_file)

    # Early stopping (based on patience)
    if epochs_without_improvement >= patience:
        printlog(f"{datetime.datetime.now()} - Early stopping after epoch {epoch}. No improvement for {patience} epochs.", log_file)
        manager_ckpt.save()
        save_training_history(history, training_history_file)
        plot_history(history, os.path.join(training_results_dir, f"Training_History.jpg"))
        break;
