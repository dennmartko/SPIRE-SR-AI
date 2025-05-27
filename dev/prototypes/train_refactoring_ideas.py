import os
import sys
import numpy as np
import tensorflow as tf
import yaml
import argparse
import datetime
from tqdm import tqdm

from astropy.io import fits
from matplotlib import pyplot as plt

from scripts.utils.data_loader import create_dataset
from scripts.utils.file_utils import get_main_dir, create_model_ckpt_folder, create_log_file, printlog, log_epoch_details, load_training_history, save_training_history
from scripts.utils.plots import plot_class_images
from models.architectures.UnetResnet34Tr import UnetResnet34Tr
from loss_functions import non_adversarial_loss

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    return parser.parse_args()

# Load config from YAML
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Create dataset
def prepare_datasets(data_path, input_class_names, target_class_names, batch_size):
    train_dir = os.path.join(data_path, "Train")
    val_dir = os.path.join(data_path, "Validation")

    train_ds, train_num_batches = create_dataset(train_dir, input_class_names, target_class_names, batch_size, is_training=True)
    val_ds, val_num_batches = create_dataset(val_dir, input_class_names, target_class_names, batch_size * 2, is_training=False)

    return train_ds, train_num_batches, val_ds, val_num_batches

# Initialize model and optimizer
def initialize_model_and_optimizer(model_name, input_shape, lr_params):
    if model_name == "UnetResnet34Tr":
        model = UnetResnet34Tr(tuple(input_shape), "channels_last")
    else:
        raise ValueError(f"Model {model_name} is not yet implemented")
    
    lr_fn = tf.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr_params[0],
        decay_steps=int(lr_params[2]),
        end_learning_rate=lr_params[1],
        power=lr_params[3]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn, beta_1=0.9)
    return model, optimizer

# Training step function
@tf.function(jit_compile=True)
def train_step(x, y, masks, alpha, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = non_adversarial_loss(predictions, y, masks, alpha)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Validation step function
def validate_step(batch_x, batch_y, model, alpha, n_targets):
    predictions = model(batch_x, training=False)
    return non_adversarial_loss(predictions, batch_y[:, :, :, :n_targets], batch_y[:, :, :, n_targets:], alpha)

# Save model checkpoints
def save_checkpoints(epoch, model, optimizer, manager_ckpt, manager_bestmodel, avg_val_loss, best_val_loss):
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        manager_bestmodel.save()
        print(f"Model improved at epoch {epoch}. Saved best model.")

    manager_ckpt.save()
    return best_val_loss

# Main training loop
def main(config):
    # Load configuration
    DS_DIR = config["data"]["data_path"]
    BATCH_SIZE = config["training"]["batch_size"]
    input_class_names = config["data"]["input"]
    target_class_names = config["data"]["target"]
    lr_params = config["training"]["polynomial_lr_schedule"]
    alpha = config["training"]["alpha"]
    patience = config["training"]["patience"]
    num_epochs = config["training"]["number_of_epochs"]

    # Prepare datasets
    train_ds, train_num_batches, val_ds, val_num_batches = prepare_datasets(DS_DIR, input_class_names, target_class_names, BATCH_SIZE)

    # Plot sample images for debugging
    inputs, labels = next(iter(val_ds.take(1)))
    save_path = os.path.join(get_main_dir(), "images/data_samples.jpg")
    plot_class_images(inputs, labels, input_class_names, target_class_names, save_path)

    # Initialize model and optimizer
    model_name = config["model"]["model"]
    run_name = config["model"]["run_name"]
    model_weights_path, first_run = create_model_ckpt_folder(model_name, run_name)
    input_shape = config["model"]["input_shape"]
    model, optimizer = initialize_model_and_optimizer(model_name, input_shape, lr_params)

    # Logging
    log_file = create_log_file(model_name, run_name)
    printlog(f"Model Training Log\nDate: {datetime.datetime.now()}", log_file)
    printlog(f"Datasets prepared successfully!\nTraining batches: {train_num_batches}\nValidation batches: {val_num_batches}", log_file)

    # Checkpoint managers
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager_bestmodel = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "BestModel"), max_to_keep=1)
    manager_ckpt = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "Checkpoint"), max_to_keep=1)

    if not first_run:
        manager_ckpt.restore_or_initialize()
    else:
        model.summary(print_fn=lambda x: printlog(x, log_file))

    # Load training history
    training_history_file = os.path.join(model_weights_path, "history.json")
    history = load_training_history(training_history_file)

    # Training loop
    best_val_loss = float('inf')
    start_epoch = len(history["epochs"]) + 1
    epochs = np.arange(start_epoch, num_epochs + 1)

    for epoch in tqdm(epochs, desc="Training model..."):
        # Training step
        total_train_loss = 0
        for idx, (batch_x, batch_y) in enumerate(train_ds):
            total_train_loss += train_step(batch_x, batch_y[:, :, :, :n_targets], batch_y[:, :, :, n_targets:], alpha, model, optimizer).numpy()
        
        avg_train_loss = total_train_loss / train_num_batches.numpy()

        # Validation step
        total_val_loss = 0
        for batch_x, batch_y in val_ds:
            val_loss = validate_step(batch_x, batch_y, model, alpha, n_targets)
            total_val_loss += val_loss.numpy()

        avg_val_loss = total_val_loss / val_num_batches.numpy()

        # Checkpointing and saving history
        best_val_loss = save_checkpoints(epoch, model, optimizer, manager_ckpt, manager_bestmodel, avg_val_loss, best_val_loss)
        
        history["epochs"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["improved"].append(int(avg_val_loss < best_val_loss))

        save_training_history(history, training_history_file)

        # Early stopping
        if history["improved"][-1] == 0:
            patience -= 1
        if patience <= 0:
            printlog(f"Early stopping at epoch {epoch}. No improvement for {patience} epochs.", log_file)
            break

if __name__ == "__main__":
    # Parse args and load config
    args = parse_args()
    config = load_config(args.config)

    # Start the training process
    main(config)