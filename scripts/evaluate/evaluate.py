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
    pos_flux_cornerplot_deprecated,

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
    model_weights_path, sim_results_dir = setup_directories(config)

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
    indices = np.array([0, 1000, 1500, 2300, 2390]) # Give 5 indices which we will be using to display images

    with progress as prog:
        X = load_input_data_asarray(indices, config["data"]["input"], test_data_dir, tensor_shape_X=tuple(config["model"]["input_shape"]), progress=prog)
        Y, wcs_dict = load_target_data_asarray(indices, target_classes=config["data"]["target"], path=test_data_dir, tensor_shape_Y=tuple(config["model"]["output_shape"]), progress=prog)

    ## Load the input, native, target and SR catalogs
    # Read the FITS files into Table objects
    input_table = Table.read(config['data']['input_catalog_path'])
    native_table = Table.read(config['data']['native_catalog_path'])
    target_table = Table.read(config['data']['target_catalog_path'])
    sr_table = Table.read(os.path.join(sim_results_dir, "500SR_SR_catalog.fits"))

    # Convert the tables to pandas DataFrames
    df_target = target_table.to_pandas()
    df_sr = sr_table.to_pandas()
    df_input = input_table.to_pandas()
    df_native = native_table.to_pandas()

    # Set the source flux columns to be in mJy
    df_target["source_flux_target"] *= 1000
    df_sr["S500SR"] *= 1000
    df_input["SSPIRE500"] *= 1000
    df_native["source_flux_native"] *= 1000

    # We mask all catalogs to only include sensible fluxes for which our method is valid
    # Note toself: We need to do this in a loop when doing it for more bands
    flux_min, flux_max = 2, 100
    df_target = df_target[(df_target["source_flux_target"] >= flux_min) & (df_target["source_flux_target"] <= flux_max)]
    df_sr = df_sr[(df_sr["S500SR"] >= flux_min) & (df_sr["S500SR"] <= flux_max)]
    df_input = df_input[(df_input["SSPIRE500"] >= flux_min) & (df_input["SSPIRE500"] <= flux_max)]
    df_native = df_native[(df_native["source_flux_native"] >= flux_min) & (df_native["source_flux_native"] <= flux_max)]

    # Create a scatterplot of RA and Dec for all four catalogs
    plt.figure(figsize=(8, 6))
    # plt.scatter(df_target["ra"], df_target["dec"], s=1, alpha=0.3, label="Target")
    # plt.scatter(df_sr["ra"], df_sr["dec"], s=1, alpha=0.3, label="SR")
    # plt.scatter(df_input["ra"], df_input["dec"], s=1, alpha=0.3, label="Input")
    plt.scatter(df_native["ra"], df_native["dec"], s=1, alpha=0.6, label="Native")
    plt.xlabel("RA")
    plt.ylabel("Dec")
    plt.title("Catalogs RA vs Dec")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sim_results_dir, "catalogs_ra_dec_scatter.png"))
    plt.close()


    # Create matched catalogs
    matched_sr = cross_match_catalogs(df_sr, df_input, flux_col_source="S500SR", flux_col_target="SSPIRE500")
    matched_target = cross_match_catalogs(df_target, df_input, flux_col_source="source_flux_target", flux_col_target="SSPIRE500")
    matched_native = cross_match_catalogs(df_native, df_input, flux_col_source="source_flux_native", flux_col_target="SSPIRE500", search_radius=8)

    # Check if the matched catalogs have enough entries
    if matched_sr.shape[0] < 0.1 * df_input.shape[0]:
        print("Matched super-resolved catalog has less than 10% of the input catalog entries. "
                     "This may indicate that the model was not trained properly or the catalogs were not generated correctly. Exiting.")
        sys.exit(0)
    # Now its time to plot the results
    # Plots come from our custom library

    # IQR flux statistics Target, Native and SR
    plot_binned_iqr(
        S_in_list=[matched_sr["SSPIRE500"], matched_target["SSPIRE500"], matched_native["SSPIRE500"]],
        S_out_list=[matched_sr["S500SR"], matched_target["source_flux_target"], matched_native["source_flux_native"]],
        bool_scatterplots=[True, False, False],
        legend_labels=['Super-resolved', 'Target (Y)', 'Native (PLW)'],
        xlabel=r'Input 500$\mu m$ Source Flux $S_{in}$ [mJy]',
        ylabel=r'$(S_{out} - S_{in})/S_{in}$',
        colors=["#8b0023", "#0a5c36", "#fbb917"],
        bins=15,
        save_path=os.path.join(sim_results_dir, "IQR_flux_statistics.pdf"),
    )

    print(matched_sr["SSPIRE500"].shape, matched_sr["S500SR"].shape, matched_native["source_flux_native"])

    # plot_binned_iqr(
    #     S_in_list=[matched_sr["SSPIRE500"], matched_native["SSPIRE500"]],
    #     S_out_list=[matched_sr["S500SR"], matched_native["source_flux_native"]],
    #     bool_scatterplots=[True, False],
    #     legend_labels=['Super-resolved', 'Native (PLW)'],
    #     xlabel=r'Input 500$\mu m$ Source Flux $S_{in}$ [mJy]',
    #     ylabel=r'$(S_{out} - S_{in})/S_{in}$',
    #     colors=["#8b0023", "#fbb917"],
    #     bins=15,
    #     save_path=os.path.join(sim_results_dir, "IQR_flux_statistics.pdf"),
    # )

    ## Contour plot of the reliability and completeness. This plot is a bit complicated to interpret, but it's one plot showing both metrics.
    # target vs SR
    contourplot_completeness_reliability(df_target, df_sr,
                                  "source_flux_target", "S500SR",
                                  bins=np.logspace(np.log10(2), np.log10(80), 50, base=10), 
                                  levels=np.arange(10, 80 + 10, 10), search_radius=4, save_path=os.path.join(sim_results_dir, "contourplot_CR_target_vs_SR.pdf"))
    # input vs SR
    contourplot_completeness_reliability(df_input, df_sr,
                                  "SSPIRE500", "S500SR",
                                  bins=np.logspace(np.log10(2), np.log10(80), 50, base=10), 
                                  levels=np.arange(10, 80 + 10, 10), search_radius=4, save_path=os.path.join(sim_results_dir, "contourplot_CR_input_vs_SR.pdf"))
    # input vs native
    contourplot_completeness_reliability(df_input, df_native,
                                  "SSPIRE500", "source_flux_native",
                                  bins=np.logspace(np.log10(2), np.log10(50), 20, base=10), 
                                  levels=np.arange(10, 80 + 10, 10), search_radius=4, save_path=os.path.join(sim_results_dir, "contourplot_CR_input_vs_native.pdf"))
    
    ## Actual reliability and completeness plots. Indicates the cumulative C, R scores at 10mJy that can be used to compare the models. Includes poison error bars.

    # Target vs SR
    plot_completeness_reliability(df_target, df_sr,
                                    "source_flux_target", "S500SR",
                                    bins=np.logspace(np.log10(2), np.log10(80), 20, base=10), search_radius=4, save_path=os.path.join(sim_results_dir, "completeness_reliability_target_vs_SR.pdf"))
    # input vs SR
    plot_completeness_reliability(df_input, df_sr,
                                    "SSPIRE500", "S500SR",
                                    bins=np.logspace(np.log10(2), np.log10(80), 20, base=10), search_radius=4, save_path=os.path.join(sim_results_dir, "completeness_reliability_input_vs_SR.pdf"))

    # input vs native
    plot_completeness_reliability(df_input, df_native,
                                    "SSPIRE500", "source_flux_native",
                                    bins=np.logspace(np.log10(2), np.log10(80), 20, base=10), search_radius=4, save_path=os.path.join(sim_results_dir, "completeness_reliability_input_vs_native.pdf"))


    ## Plot the 2D density plot of the flux reproduction and the positional offset between matches
    n_bins = 75
    xbins1 = np.linspace(-1, 1, n_bins + 1)
    xbins2 = np.linspace(-1, 2, n_bins + 1)
    ybins1 = np.linspace(0, 4, n_bins + 1)
    ybins2 = np.linspace(0, 8, n_bins + 1)
    # Target vs SR
    matched_target_sr = cross_match_catalogs(df_sr, df_target, flux_col_source="S500SR", flux_col_target="source_flux_target")

    pos_flux_plot(
        matched_target_sr['source_flux_target'].values,
        matched_target_sr['S500SR'].values,
        matched_target_sr['angDist'].values,
        xbins=xbins1, ybins=ybins1, interpolation=True, save_path=os.path.join(sim_results_dir, "pos_flux_plot_target_vs_SR.png"),
        xlabel=r'$(S_{SR} - S_{Target})/S_{Target}$',
    )

    # input vs SR
    pos_flux_plot(
        matched_sr['SSPIRE500'].values,
        matched_sr['S500SR'].values,
        matched_sr['angDist'].values,
        xbins=xbins1, ybins=ybins1, interpolation=True, save_path=os.path.join(sim_results_dir, "pos_flux_plot_input_vs_SR.png"),
        xlabel=r'$(S_{SR} - S_{input})/S_{input}$',
    )

    # input vs native
    pos_flux_plot(
        matched_native['SSPIRE500'].values,
        matched_native['source_flux_native'].values,
        matched_native['angDist'].values,
        xbins=xbins2, ybins=ybins2, interpolation=True, save_path=os.path.join(sim_results_dir, "pos_flux_plot_input_vs_native.png"),
        xlabel=r'$(S_{native} - S_{input})/S_{input}$',
    )

    # input vs target
    pos_flux_plot(
        matched_target['SSPIRE500'].values,
        matched_target['source_flux_target'].values,
        matched_target['angDist'].values,
        xbins=xbins1, ybins=ybins1, interpolation=True, save_path=os.path.join(sim_results_dir, "pos_flux_plot_input_vs_target.png"),
        xlabel=r'$(S_{target} - S_{input})/S_{input}$',
    )

    # sanity check
    tmp_df_target = df_target.copy().rename(columns={"source_flux_target": "S500"})
    tmp_df_sr = df_sr.copy().rename(columns={"S500SR": "S500SR"})

    matched_sr_target2 = construct_matched_catalog(tmp_df_sr, tmp_df_target, "500SR", max_distance=4)

    pos_flux_plot(
        matched_sr_target2['S500'].values,
        matched_sr_target2['S500SR'].values,
        matched_sr_target2['distance'].values,
        xbins=xbins1, ybins=ybins1, interpolation=True, save_path=os.path.join(sim_results_dir, "pos_flux_plot_sr_vs_target_sanity_check.png"),
        xlabel=r'$(S_{target} - S_{input})/S_{input}$',
    )
    
    pos_flux_cornerplot_deprecated(tmp_df_target, tmp_df_sr, "500SR", save_path=os.path.join(sim_results_dir, "cornerplot_sr_vs_target_sanity_check.png"))

    ## Store image samples
    
    # First, we super-resolve the chosen images
    predictions = SuperResolve(config, X, model, progress)

    # Now we put the images in the right format
    images = np.squeeze(np.array([Y, predictions, abs(Y-predictions)]))*1e3

    # We need to select an interesting source for each column/sample
    # We need to center highlighted region on truth/target position
    # We simply mask all sources that are within each WCS and take that catalog
    # and convert to pixel coordinates. As such, we have cat_targets = [], cat_sr = []
    # Then this methodology can also be used for SCUBA2 imaging.
    cat_target_images = []
    cat_sr_images = []
    
    for w in wcs_dict.values():
        coords_sr = w.all_world2pix(df_sr[['ra', 'dec']].values, 0)
        coords_target = w.all_world2pix(df_target[['ra', 'dec']].values, 0)

        x_sr, y_sr = coords_sr[:, 0], coords_sr[:, 1]
        x_target, y_target = coords_target[:, 0], coords_target[:, 1]
        mask_sr = (x_sr >= 0) & (x_sr < X.shape[2]) & (y_sr >= 0) & (y_sr < X.shape[1])
        mask_target = (x_target >= 0) & (x_target < X.shape[2]) & (y_target >= 0) & (y_target < X.shape[1])

        # Append x, y and corresponding flux ('S500SR' for df_sr) for super-resolved data
        cat_sr_images.append(
            np.array([x_sr[mask_sr], y_sr[mask_sr], df_sr['S500SR'].values[mask_sr]]).T
        )
        # Append x, y and corresponding flux ('source_flux_target' for df_target) for target data
        cat_target_images.append(
            np.array([x_target[mask_target], y_target[mask_target], df_target['source_flux_target'].values[mask_target]]).T
        )

    # vmin/vmax per row
    vmins = [0., 0., 0]
    vmaxs = [15, 15, 10]
    plot_image_grid(
        images,
        cat_sr_images=cat_sr_images,
        cat_target_images=cat_target_images,
        region_size=24,
        vmins=vmins,
        vmaxs=vmaxs,
        cmap='afmhot',
        pad_w=0.03,
        pad_h=0.04,
        region_color='blue',
        save_path=os.path.join(sim_results_dir, "image_grid_simulations.png"),
    )





