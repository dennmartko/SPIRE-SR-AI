import os
import datetime
import argparse
import yaml
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from astropy.io import fits

# # Suppress warnings from photutils
# warnings.filterwarnings("ignore")

# Enable TF32 for more TFLOPS
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.get_logger().setLevel('ERROR')

# Custom imports
from scripts.utils.data_loader import (
    load_input_data_asarray,
    load_target_data_asarray,
)
from scripts.utils.file_utils import (
    get_main_dir,
    create_model_ckpt_folder,
    create_model_results_subfolder,
)

from scripts.utils.evaluation_plots import (
    flux_reproduction_plot,
    pos_flux_cornerplot,
    CompletenessReliabilityPlot,
    PlotSuperResolvedImage,
    PlotInputImages
)

from scripts.utils.source_extraction_utils import (
    construct_SR_source_catalog_deprecated
)
from models.architectures.UnetResnet34Tr import UnetResnet34Tr

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model on a dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the test config file")
    return parser.parse_args()

# Load configuration
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load data
def load_test_data(config, indices):
    test_dir = config["data"]["test_dataset_path"]
    input_class_names = config["data"]["input"]
    target_class_names = config["data"]["target"]
    input_shape = tuple(config["model"]["input_shape"])
    target_shape = tuple(config["model"]["output_shape"])

    data_X = load_input_data_asarray(indices, input_class_names, test_dir, input_shape)
    data_Y, target_sources_cat, wcs_arr = load_target_data_asarray(
        indices, target_class_names, test_dir, target_shape
    )
    return data_X, data_Y, target_sources_cat, wcs_arr

# Load model
def initialize_model(config):
    model_name = config["model"]["model"]
    input_shape = tuple(config["model"]["input_shape"])

    if model_name == "UnetResnet34Tr":
        model = UnetResnet34Tr(input_shape, "channels_last")
    else:
        raise NotImplementedError(f"Model '{model_name}' is not implemented.")
    return model

# Prepare directories
def setup_directories(config):
    model_name = config["model"]["model"]
    run_name = config["model"]["run_name"]

    # Model weights and results directories
    model_weights_path, _ = create_model_ckpt_folder(model_name, run_name)
    test_results_dir = create_model_results_subfolder(model_name, run_name, "testing")
    sim_results_dir = os.path.join(test_results_dir, "simulation_results")

    os.makedirs(sim_results_dir, exist_ok=True)
    return model_weights_path, sim_results_dir

def SuperResolve(config, data_X, model):
    batch_size = 16 # We assume a batch_size of 16
    target_shape = tuple(config["model"]["output_shape"])

    batch_its = data_X.shape[0]//batch_size if data_X.shape[0] % batch_size == 0 else data_X.shape[0]//batch_size + 1

    # We store the predictions in memory so we can perform multiprocessing on them later
    pred = np.zeros((data_X.shape[0], ) + tuple(target_shape), dtype=np.float32)

    for batch_idx in tqdm(range(batch_its), desc="Super-resolving images...."):
        # Batch bounds
        start = batch_size * batch_idx
        end = batch_size * (batch_idx + 1) if batch_idx != batch_its - 1 else None

        # Prediction
        pred[start:end] = model(data_X[start:end], training=False).numpy()

    return pred

# Main evaluation script
def main():
    args = parse_args()
    config = load_config(args.config)

    # Data loading
    test_dir = config["data"]["test_dataset_path"]
    input_class_names = config["data"]["input"]
    total_samples = len(os.listdir(os.path.join(test_dir, input_class_names[0])))
    indices = np.arange(total_samples)

    print(f"{datetime.datetime.now()} - Loading data...")
    data_X, data_Y, target_sources_cat, wcs_arr = load_test_data(config, indices)
    print(f"{datetime.datetime.now()} - Data loaded! TOTAL SAMPLES: {total_samples}")

    # Initialize model
    model = initialize_model(config)
    model_weights_path, sim_results_dir = setup_directories(config)

    # Load model weights
    print(f"{datetime.datetime.now()} - Restoring model...")
    ckpt = tf.train.Checkpoint(model=model)
    manager_bestmodel = tf.train.CheckpointManager(ckpt, os.path.join(model_weights_path, "BestModel"), max_to_keep=1)
    manager_bestmodel.restore_or_initialize()
    print(f"{datetime.datetime.now()} - Model Loaded!")

    # Super-resolve the target bands
    predictions = SuperResolve(config, data_X, model)
    print(f"{datetime.datetime.now()} - Target bands super-resolved!")

    # We loop over each target band, and analyse the results
    ## Note to self: We should later look at a function that takes in the input catalog, sorts it...
    ## ... to only include the sources within the covered test images (because the input catalog is training + validation + testing)
    ## The target sources we picked up from the header are the sources detected within the target images, they do not tell the entire story
    
    # Here, we hardcode FWHM, flux thresholds for each band
    # Fluxes are always in Jy, unless otherwise stated/inferred
    # FWHM in arcseconds (")
    sr_config = {"250SR": {"FWHM": 6.0, "flux_threshold": 2e-3}, 
                 "350SR": {"FWHM": 6.0, "flux_threshold": 2e-3},
                 "500SR": {"FWHM": 7.9, "flux_threshold": 2e-3}}

    for target_cl in config["data"]["target"]:
                
        # First, we convert the catalog containing the extracted target sources to a pandas df
        test_dtypes = {f"S{target_cl[:-2]}": np.float32, "xpix": np.float32, "ypix":np.float32, "ra":np.float32, "dec":np.float32, "ImageID":np.int32}
        test_cat_cols = [f'S{target_cl[:-2]}', 'xpix', 'ypix', 'ra', 'dec', 'ImageID']

        target_sources_cat = pd.DataFrame(data=target_sources_cat[f"S{target_cl[:-2]}"], columns=test_cat_cols).astype(test_dtypes)

        # Our first objective is to extract the sources from the super-resolved image
        ## Compute the SR test catalog using multi-processing
        SR_cat = construct_SR_source_catalog_deprecated(target_cl, sr_config[target_cl]["FWHM"], sr_config[target_cl]["flux_threshold"], predictions, wcs_arr, N_CPU=config["sys_config"]["n_cpu_cores"])
        print(f"{datetime.datetime.now()} - SR catalog for band: {target_cl[:-2]} constructed!")

        # Flux reproduction plots (against measured extracted target sources!)
        flux_reproduction_plot(target_sources_cat, SR_cat, target_cl, matching_distance=4/3600, save_path=os.path.join(sim_results_dir, f'TargetFluxReproduction_{target_cl[:-2]}.png'))
        print(f"{datetime.datetime.now()} - Flux reproduction plotted!")

        # Positional Accuracy vs Flux Accuracy Plot
        cornerplot_args = {"max_distance": 7.9, "ReproductionRatio_min": -0.5, "ReproductionRatio_max": 0.5}
        pos_flux_cornerplot(target_sources_cat, SR_cat, target_cl, save_path=os.path.join(sim_results_dir, f'PS_plot_{target_cl[:-2]}.png'), **cornerplot_args)
        print(f"{datetime.datetime.now()} - Position-Flux Cornerplot finished!")

        # Completeness and Reliability Plot
        CompletenessReliabilityPlot(target_sources_cat, SR_cat, target_cl, max_distance=4.0/3600, save_path=os.path.join(sim_results_dir, f'Completeness_Reliability_{target_cl[:-2]}.png'))
        print(f"{datetime.datetime.now()} - Completeness and Reliability computed & plotted!")

        # Input vs Target vs SuperResolved/Predicted image comparison plot
        ## You can fill the array with any IDs where image IDs < maximum ID
        ## Comment this loop out if it gives errors, it has very specific use-cases
        ## It will also not work if the predicted images are too bad (no sources detected)
        generated_images_IDs = np.array([0, 1, 50, 51, 293, data_Y.shape[0]//2, data_Y.shape[0]//2 + 1, data_Y.shape[0]-2, data_Y.shape[0]-1], dtype=np.int32)
        
        for ID in tqdm(generated_images_IDs, desc="Creating image comparison plots...."):           
            # Plot the Target image, and retrieve the selected sources
            selected_sources, target_sources_in_region = PlotSuperResolvedImage(target_sources_cat, wcs_arr, target_cl, data_Y[ID], ID, save_path=os.path.join(sim_results_dir, f"TargetImage_{ID}.png"))

            # Project these regions on the SuperResolved image
            selected_sources, target_sources_in_region = PlotSuperResolvedImage(SR_cat, wcs_arr, target_cl, predictions[ID], ID, save_path=os.path.join(sim_results_dir, f"SupeResolvedImage_{ID}.png"), selected_sources=selected_sources, target_sources_in_region=target_sources_in_region)

            # Plot the input images and project these regions on each image
            PlotInputImages(data_X[ID], boxsize=30, selected_sources=selected_sources, save_path=os.path.join(sim_results_dir, f"InputHerschelImages_{ID}.png"))


if __name__ == "__main__":
    main()
