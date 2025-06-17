# Standard libraries
import os
import sys
import datetime
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.ndimage import gaussian_filter

# Project-specific imports
from scripts.utils.metrics import (
    calculate_flux_statistics, 
    cross_match_catalogs, 
    construct_matched_catalog
)
from scripts.utils.data_loader import (
    load_input_data_asarray,
    load_target_data_asarray
)
from scripts.utils.file_utils import (
    get_main_dir,
    setup_directories,
    create_model_results_subfolder
)

from scripts.utils.evaluation_plots import (
    plot_binned_iqr,
    contourplot_completeness_reliability,
    plot_completeness_reliability,
    pos_flux_plot,
    plot_image_grid,
    scuba_recovery_plot,

)

from models.architectures.UnetResnet34Tr import UnetResnet34Tr
from models.architectures.UnetResnet34TrNew import UnetResnet34TrNew
from models.architectures.SwinUnet import swin_unet_2d_base
from models.architectures.Unet import build_unet

# Configure logger and warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", module="photutils")

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

# Load model - Needs to be updated to include all models
def initialize_model(config):
    model_name = config["model"]["model"]
    run_name = config["model"]["run_name"]
    input_shape = tuple(config["model"]["input_shape"])

    if model_name == "UnetResnet34Tr":
        model = UnetResnet34Tr(input_shape, "channels_last")
    elif ("new" in run_name.lower()) and (model_name == "UnetResnet34Tr"):
        model = UnetResnet34TrNew(input_shape, "channels_last")
        model.build((None,) + tuple(input_shape))
        
    elif model_name == "SwinUnet":
        filter_num_begin = 96     # number of channels in the first downsampling block; it is also the number of embedded dimensions
        depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
        stack_num_down = 3         # number of Swin Transformers per downsampling level
        stack_num_up = 3           # number of Swin Transformers per upsampling level
        patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
        num_heads = [6, 12, 12, 12]   # number of attention heads per down/upsampling level
        window_size = [8, 4, 4, 2] # the size of attention window per down/upsampling level
        num_mlp = 512              # number of MLP nodes within the Transformer
        shift_window=True          # Apply window shifting, i.e., Swin-MSA
        model = swin_unet_2d_base(tuple(input_shape), filter_num_begin, depth, stack_num_down, stack_num_up, 
                        patch_size, num_heads, window_size, num_mlp, 
                        shift_window=shift_window, name='swin_unet')
    elif model_name == "Unet":
        model = build_unet(tuple(input_shape), "channels_last")
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

if __name__ == "__main__":
    # Unpack arguments
    args = parse_args()
    config = load_config(args.config)

    # Create progress-bar
    progress = create_progress()

    # Initialize model and results paths
    model = initialize_model(config)
    model_weights_path, results_dir = setup_directories(config, purpose="observations")

    ## Load model weights
    print(f"{datetime.datetime.now()} - Restoring model...")
    ckpt = tf.train.Checkpoint(model=model)
    manager_bestmodel = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "BestModel"), max_to_keep=1)
    manager_bestmodel.restore_or_initialize()
    print(f"{datetime.datetime.now()} - Model Loaded!")


    # Data loading
    ## We load all test data in one-go
    test_data_dir = config["data"]["test_dataset_path"]
    input_folder = config["data"]["input"][0]
    indices = np.array([14, 15, 27, 21, 26]) # Give 5 indices which we will be using to display images

    with progress as prog:
        X = load_input_data_asarray(indices, config["data"]["input"], test_data_dir, tensor_shape_X=tuple(config["model"]["input_shape"]), progress=prog)
        Y, wcs_dict = load_target_data_asarray(indices, target_classes=config["data"]["target"], path=test_data_dir, tensor_shape_Y=tuple(config["model"]["output_shape"]), progress=prog)

    ## Load the input, native, target and SR catalogs
    # Read the FITS files into Table objects
    dataset_name = os.path.basename(os.path.dirname(os.path.dirname(config["data"]["test_dataset_path"])))
    catalog_filename = f"{dataset_name}_SR_catalog.fits"
    scuba_table = Table.read(config['data']['scuba_catalog_path'])
    sr_table = Table.read(os.path.join(results_dir, catalog_filename))

    # Convert the tables to pandas DataFrames
    df_scuba = scuba_table.to_pandas()
    df_sr = sr_table.to_pandas()

    print(df_scuba.head())

    # Define correction factor
    corr_factor = 0.84

    # Set the source flux columns to be in mJy (scuba already in mJy)
    df_sr["S500SR"] *= 1000

    # We mask all catalogs to only include sensible fluxes for which our method is valid
    # Note toself: We need to do this in a loop when doing it for more bands
    flux_min, flux_max = 2, 100
    df_scuba = df_scuba[(df_scuba["S450"] >= flux_min) & (df_scuba["S450"] <= flux_max)]
    df_sr = df_sr[(df_sr["S500SR"] >= flux_min) & (df_sr["S500SR"] <= flux_max)]

    # Rename RA, Dec columns
    df_scuba.rename(columns={"RA": "ra", "Dec": "dec"}, inplace=True)

    df_scuba['scuba_500corr'] = df_scuba['S450'] * corr_factor

    # cross-match the catalogs
    matched_catalog = cross_match_catalogs(df_sr, df_scuba, flux_col_source="S500SR", flux_col_target="scuba_500corr", keep_columns=["S450", "S450_total_err", "file_id"])
    matched_catalog['S500corr'] = matched_catalog['S450'] * corr_factor
    matched_catalog['S500corr_total_err'] = matched_catalog["S450_total_err"] * corr_factor

    print(matched_catalog.head())


    # Plot the fluxes
    scuba_recovery_plot(matched_catalog, save_path=os.path.join(results_dir, "scuba_recovery_plot.pdf"))

    # Store image samples
    
    # First, we super-resolve the chosen images
    predictions = SuperResolve(config, X, model, progress)

    # Now we put the images in the right format
    images = np.squeeze(np.array([X[:, :, :, -1:], predictions, Y*corr_factor]))*1e3

    # We need to select an interesting source for each column/sample
    # We need to center highlighted region on truth/target position
    # We simply mask all sources that are within each WCS and take that catalog
    # and convert to pixel coordinates. As such, we have cat_targets = [], cat_sr = []
    # Then this methodology can also be used for SCUBA2 imaging.
    cat_scuba_images = []
    cat_sr_images = []
    
    for w in wcs_dict.values():
        coords_sr = w.all_world2pix(df_sr[['ra', 'dec']].values, 0)
        coords_scuba = w.all_world2pix(df_scuba[['ra', 'dec']].values, 0)

        x_sr, y_sr = coords_sr[:, 0], coords_sr[:, 1]
        x_scuba, y_scuba = coords_scuba[:, 0], coords_scuba[:, 1]
        mask_sr = (x_sr >= 0) & (x_sr < X.shape[2]) & (y_sr >= 0) & (y_sr < X.shape[1])
        mask_scuba = (x_scuba >= 0) & (x_scuba < X.shape[2]) & (y_scuba >= 0) & (y_scuba < X.shape[1])

        # Append x, y and corresponding flux ('S500SR' for df_sr) for super-resolved data
        cat_sr_images.append(
            np.array([x_sr[mask_sr], y_sr[mask_sr], df_sr['S500SR'].values[mask_sr]]).T
        )
        # Append x, y and corresponding flux ('source_flux_scuba' for df_target) for target data
        cat_scuba_images.append(
            np.array([x_scuba[mask_scuba], y_scuba[mask_scuba], df_scuba['S450'].values[mask_scuba]]).T
        )

    # vmin/vmax per row
    vmins = [0, 0., 0., 0]
    vmaxs = [30, 20, 20, 10]
    plot_image_grid(
        images,
        cat_sr_images=cat_sr_images,
        cat_target_images=cat_scuba_images,
        region_size=24,
        vmins=vmins,
        vmaxs=vmaxs,
        cmap='afmhot',
        pad_w=0.03,
        pad_h=0.04,
        region_color='blue',
        save_path=os.path.join(results_dir, "image_grid_scuba.png"),
    )

    temp_matched_cat = matched_catalog[matched_catalog['S450'] >= 10]

    print("Mean S500SR/Sscuba500:", np.mean(matched_catalog['S500SR']/matched_catalog['scuba_500corr']))
    print("Median S500SR/Sscuba500:", np.median(matched_catalog['S500SR']/matched_catalog['scuba_500corr']))
    print("Mean S500SR/S450:", np.mean(matched_catalog['S500SR']/matched_catalog['S450']))
    print("Median S500SR/S450:", np.median(matched_catalog['S500SR']/matched_catalog['S450']))




    print("Mean S500SR/Sscuba500:", np.mean(temp_matched_cat['S500SR']/temp_matched_cat['scuba_500corr']))
    print("Median S500SR/Sscuba500:", np.median(temp_matched_cat['S500SR']/temp_matched_cat['scuba_500corr']))
    print("Mean S500SR/S450:", np.mean(temp_matched_cat['S500SR']/temp_matched_cat['S450']))
    print("Median S500SR/S450:", np.median(temp_matched_cat['S500SR']/temp_matched_cat['S450']))

    # Save the matched catalog as a CSV file in the results directory
    matched_catalog.to_csv("matched_catalog.csv", index=False)