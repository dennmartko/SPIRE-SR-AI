import os
import datetime
import argparse
import yaml
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

from astropy.io import fits
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from concurrent.futures import ProcessPoolExecutor, as_completed


# # Suppress warnings from photutils
# warnings.filterwarnings("ignore")

# Enable TF32 for more TFLOPS
tf.get_logger().setLevel('ERROR')

# Custom imports
from scripts.utils.data_loader import (
    load_input_data_asarray,
    load_target_data_asarray,
)
from scripts.utils.file_utils import (
    get_main_dir,
    setup_directories,
    create_model_results_subfolder,
)

from scripts.utils.source_extraction_utils import (
    construct_source_catalog
)
from models.architectures.UnetResnet34Tr import UnetResnet34Tr
from astropy.table import Table

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model on a dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the test config file")
    return parser.parse_args()

# Load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
# Set-up progress bar
def create_progress():
    progress = Progress(
        TextColumn("[bold blue]Status:[/bold blue] [medium_purple]{task.description}"),
        BarColumn(
            bar_width=60,
            complete_style="bold green",
            finished_style="green",
            pulse_style="bright_blue"
        ),
        TextColumn("[bold cyan]{task.percentage:>3.0f}% Complete[bold cyan]"),
        TimeElapsedColumn(),
        refresh_per_second=10
    )
    return progress

# Load model
def initialize_model(config):
    model_name = config["model"]["model"]
    input_shape = tuple(config["model"]["input_shape"])

    if model_name == "UnetResnet34Tr":
        model = UnetResnet34Tr(input_shape, "channels_last")
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")
    return model

def SuperResolve(config, data_X, model, progress):
    target_shape = tuple(config["model"]["output_shape"])
    batch_size = int(config["data"]["test_batch_size"])

    batch_its = data_X.shape[0] // batch_size if data_X.shape[0] % batch_size == 0 else data_X.shape[0] // batch_size + 1

    # We store the predictions in memory so we can perform multiprocessing on them later
    pred = np.zeros((data_X.shape[0], ) + tuple(target_shape), dtype=np.float32)

    progress_task = progress.add_task(f"Super-resolving images...", total=batch_its)

    for batch_idx in range(batch_its):
        # Batch bounds
        start = batch_size * batch_idx
        end = batch_size * (batch_idx + 1) if batch_idx != batch_its - 1 else None

        # Prediction
        pred[start:end] = model(data_X[start:end], training=False).numpy()

        progress.update(progress_task, advance=1)

    return pred


# Here, we hardcode FWHM, flux thresholds for each band
# Fluxes are always in Jy, unless otherwise stated/inferred
# FWHM in arcseconds (")
sr_config = {"250SR": {"FWHM": 6.0, "flux_threshold": 2e-3}, 
                "350SR": {"FWHM": 6.0, "flux_threshold": 2e-3},
                "500SR": {"FWHM": 7.9, "flux_threshold": 2e-3}}

if __name__ == "__main__":
    # Unpack arguments
    args = parse_args()
    config = load_config(args.config)

    # Create progress-bar
    progress = create_progress()

    # Data loading
    ## We load all test data in one-go
    test_dir = config["data"]["test_dataset_path"]
    input_folder = config["data"]["input"][0]
    indices = np.arange(len(os.listdir(os.path.join(test_dir, input_folder))))

    with progress as prog:
        X = load_input_data_asarray(indices, config["data"]["input"], test_dir, tensor_shape_X=tuple(config["model"]["input_shape"]), progress=prog)
        Y, wcs_dict = load_target_data_asarray(indices, target_classes=config["data"]["target"], path=test_dir, tensor_shape_Y=tuple(config["model"]["output_shape"]), progress=prog)

    # First, calculate the ground truth high-resolution catalog(s)
    ## We implement multi-processing such that each cutout is processed individually
    with progress as prog:
        for idx, target_cl in enumerate(config["data"]["target"]):
            # progress_task = prog.add_task(f"Catalog extraction for target {target_cl}...", total=int(Y[..., idx].shape[0]))
            target_source_catalog = construct_source_catalog(
                Y[..., idx],
                wcs_dict,
                sr_config[target_cl]["FWHM"],
                sr_config[target_cl]["flux_threshold"],
                N_CPU=config["sys_config"]["n_cpu_cores"],
                progress=prog,
                task_desc=f"Extracting source catalog from target images for {target_cl} class..."
            )

            target_source_catalog.rename(columns={'source_flux': 'source_flux_target'}, inplace=True)

            # Store the catalog
            catalog_filename = f"{target_cl}_target_catalog.fits"
            catalog_path = os.path.join(config['data']['target_catalog_output_dir'], catalog_filename)
            catalog_table = Table.from_pandas(target_source_catalog)
            catalog_table.write(catalog_path, format='fits', overwrite=True)

    # Finally, we perform super-resolution using the chosen model
    # Subsequently, we extract the source catalog from the super-resolved images
    
    ## Initialize model
    model = initialize_model(config)
    model_weights_path, sim_results_dir = setup_directories(config)

    ## Load model weights
    print(f"{datetime.datetime.now()} - Restoring model...")
    ckpt = tf.train.Checkpoint(model=model)
    manager_bestmodel = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "BestModel"), max_to_keep=1)
    manager_bestmodel.restore_or_initialize()
    print(f"{datetime.datetime.now()} - Model Loaded!")

    ## Super-resolve the target bands
    with progress as prog:
        predictions = SuperResolve(config, X, model, prog)
    print(f"{datetime.datetime.now()} - Target bands super-resolved!")

    ## Source extraction on super-resolved images
    with progress as prog:
        for idx, target_cl in enumerate(config["data"]["target"]):
            #progress_task = prog.add_task(f"Catalog extraction for super-resolved {target_cl}...", total=int(predictions[..., idx].shape[0]))
            sr_source_catalog = construct_source_catalog(
                predictions[..., idx],
                wcs_dict,
                sr_config[target_cl]["FWHM"],
                sr_config[target_cl]["flux_threshold"],
                N_CPU=config["sys_config"]["n_cpu_cores"],
                progress=prog,
                task_desc=f"Extracting source catalog from super-resolved {target_cl.replace('SR', '')} micron images"
            )

            sr_source_catalog.rename(columns={'source_flux': f'S{target_cl}'}, inplace=True)

            # Store the catalog
            catalog_filename = f"{target_cl}_SR_catalog.fits"
            catalog_path = os.path.join(sim_results_dir, catalog_filename)
            catalog_table = Table.from_pandas(sr_source_catalog)
            catalog_table.write(catalog_path, format='fits', overwrite=True)