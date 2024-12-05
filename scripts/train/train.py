import os
import sys
import numpy as np
import tensorflow as tf
from astropy.io import fits
from matplotlib import pyplot as plt

from scripts.utils.data_loader import create_dataset
from scripts.utils.file_utils import get_main_dir, create_model_ckpt_folder, create_log_file, printlog, log_epoch_details, load_training_history, save_training_history, create_model_results_subfolder
from scripts.utils.plots import data_debug_plot, display_predictions, plot_history

from models.architectures.UnetResnet34Tr import UnetResnet34Tr
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

# # Normalizer
# train_ds_norm = train_ds.map(lambda x, y: x)
# normalizer = tf.keras.layers.Normalization()
# normalizer.adapt(train_ds_norm)

# train_ds = train_ds.map(lambda x, y: (normalizer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalizer(x), y))

# Plot an image sample for debugging purposes
inputs, labels = next(iter(val_ds.take(1)))

save = os.path.join(get_main_dir(), "images/data_samples.jpg")
data_debug_plot(inputs, labels, input_class_names, target_class_names, save)


# Get model
model_name = config["model"]["model"]
run_name = config["model"]["run_name"]
model_weights_path, first_run = create_model_ckpt_folder(model_name, run_name)
input_shape = config["model"]["input_shape"]
lr_params = config["training"]["polynomial_lr_schedule"]
if model_name == "UnetResnet34Tr":
    model = UnetResnet34Tr(tuple(input_shape), "channels_last")
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
best_val_loss = float('inf')
epochs_without_improvement = 0 if len(history["patience"]) == 0 else history["patience"][-1]
patience = config["training"]["patience"]
num_epochs = config["training"]["number_of_epochs"]
start_epoch = len(history["epochs"]) + 1
epochs = np.arange(start_epoch, num_epochs + 1, 1)

# hyperparameter for the losses
alpha = config["training"]["alpha"]

@tf.function(jit_compile=True)
def train_step(x, y, masks, alpha, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        train_loss = non_adversarial_loss(predictions, y, masks, alpha)
    grads = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return train_loss

for epoch in tqdm(epochs, desc="Training model..."):
    total_train_loss = 0
    for idx, (batch_x, batch_y) in enumerate(train_ds):
        total_train_loss += train_step(batch_x, batch_y[:, :, :, :n_targets], batch_y[:, :, :, n_targets:], alpha, model, optimizer).numpy()

    # Compute average training loss for the epoch
    avg_train_loss = total_train_loss / train_num_batches.numpy()

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

    log_epoch_details(epoch, avg_train_loss, avg_val_loss, improved, log_file)

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
        break;
