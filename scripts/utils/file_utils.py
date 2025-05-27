import os
import json
import datetime
import numpy as np
import pandas as pd

from astropy.io import fits

from scripts.utils.source_extraction_utils import configure_psf_photometry
from scripts.utils.preprocess_utils import generate_image_masks

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return int(obj)
        else:
            return super(JSONEncoder, self).default(obj)

def get_main_dir():
    # Get the directory where the current file (utils.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up two levels to get to the main directory (since it's in scripts)
    main_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    return main_dir

def save_training_history(history, file_path):
    with open(file_path, 'w') as f:
        json.dump(history, f, indent=4, cls=JSONEncoder)

def load_training_history(file_path):
    with open(file_path, 'r') as f:
        history = json.load(f)
    return history

def create_model_ckpt_folder(model_name, run_name):
    model_ckpt_dir = os.path.join(get_main_dir(), "models/checkpoints", model_name)
    run_name_weights_dir = os.path.join(model_ckpt_dir, run_name)

    # Check if directories need to be created
    if not os.path.exists(model_ckpt_dir):
        os.mkdir(model_ckpt_dir)

    first_run = False

    if not os.path.exists(run_name_weights_dir):
        os.mkdir(run_name_weights_dir)
        first_run = True

    return run_name_weights_dir, first_run

def setup_directories(config):
    model_name = config["model"]["model"]
    run_name = config["model"]["run_name"]

    # Model weights and results directories
    model_weights_path, _ = create_model_ckpt_folder(model_name, run_name)
    test_results_dir = create_model_results_subfolder(model_name, run_name, "testing")
    sim_results_dir = os.path.join(test_results_dir, "simulation_results")

    os.makedirs(sim_results_dir, exist_ok=True)
    return model_weights_path, sim_results_dir

def create_log_file(model_name, run_name, first_run):
    model_log_dir = os.path.join(get_main_dir(), "logs", model_name)
    log_file = os.path.join(model_log_dir, run_name + ".log")

    # Check if directory & log file need to be created
    if first_run:
        if not os.path.exists(model_log_dir):
            os.mkdir(model_log_dir)
        open(log_file, 'w').close()
    return log_file

def log_epoch_details(epoch, train_loss, val_loss, grad_norm, improved, log_file):
    """
    Log the training details of an epoch to a file.

    :param epoch: Current epoch number.
    :param train_loss: Training loss for the epoch.
    :param val_loss: Validation loss for the epoch.
    :param grad_norm: Gradient norm for the epoch (debugging)
    :param improved: Boolean indicating if the model improved (based on validation loss).
    :param log_file: File to append the log.
    """
    
    # Log message format: Time - Epoch - Training Loss - Validation Loss - Improvement
    improvement_status = 'Improved' if improved else 'Not Improved'
    
    # Create the log entry string
    log_entry = f"{datetime.datetime.now()} - Epoch {epoch} - Train Loss: {train_loss:.3e} - Grad Norm: {grad_norm:.3e} - Val Loss: {val_loss:.3e} - {improvement_status}\n"
    
    # Append log to the file
    with open(log_file, 'a') as f:
        f.write(log_entry)

    print(log_entry)  # print it to console as well for monitoring

def create_model_results_subfolder(model_name, run_name, purpose="training"):
    model_results_dir = os.path.join(get_main_dir(), "results", model_name)
    model_runname_results_dir = os.path.join(model_results_dir, run_name)
    model_runname_purpose_results_dir = os.path.join(model_runname_results_dir, purpose)

    # Check if directories need to be created
    if not os.path.exists(model_results_dir):
        os.mkdir(model_results_dir)

    if not os.path.exists(model_runname_results_dir):
        os.mkdir(model_runname_results_dir)

    if not os.path.exists(model_runname_purpose_results_dir):
        os.mkdir(model_runname_purpose_results_dir)

    return model_runname_purpose_results_dir

def printlog(message, log_file):
    with open(log_file, 'a') as log_file:
        log_file.write(message + '\n')

def save_input_image_with_header_to_fits(img_data, img_ID, cl, cutout_wcs, save_dir):
    hdul = fits.HDUList([fits.PrimaryHDU(img_data.astype(np.float32), header=cutout_wcs.to_header())])
    # Store the data to the fits file
    hdul.writeto(f"{os.path.join(save_dir, cl)}" + f"/{cl}_{img_ID}.fits", overwrite=True)
    return 1 # Indicate successful disk save

def save_target_image_catalog_to_fits(img_data, img_ID, cl, pix_scale, fwhm, cutout_wcs, purpose, save_dir):
    sources = configure_psf_photometry(fwhm/pix_scale, 2e-3, img_data)
    sources = sources[sources["flags"] <= 1]

    # Training code will never use the 'ra' and 'dec' source coordinates of the training data
    source_df_cutout = pd.DataFrame(columns=['xpix', 'ypix'])

    # Potential errors may arise --> empty array if no sources are detected in cutout
    if sources['x_fit'] is None:
        source_df_cutout['xpix'] = np.array([], dtype=np.float32)
        source_df_cutout['ypix'] = np.array([], dtype=np.float32)

    else:
        source_df_cutout['xpix'] = np.array(sources['x_fit'], dtype=np.float32)
        source_df_cutout['ypix'] = np.array(sources['y_fit'], dtype=np.float32)

    # Create primary HDU
    primary_hdu = fits.PrimaryHDU(img_data.astype(np.float32), header=cutout_wcs.to_header())

    # Create image mask for sources >= 2mJy
    img_mask = generate_image_masks(source_df_cutout, img_data.shape)
    mask_hdu = fits.PrimaryHDU(img_mask.astype(np.float32), header=cutout_wcs.to_header())

    # Write to FITS file
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(f"{os.path.join(save_dir, cl)}" + f"/{cl}_{img_ID}.fits", overwrite=True)
    if purpose != "Test":
        cl_mask = cl + "_mask"
        mask_hdul = fits.HDUList([mask_hdu])
        mask_hdul.writeto(f"{os.path.join(save_dir, cl_mask)}" + f"/{cl_mask}_{img_ID}.fits", overwrite=True)
    return 1 # Indicate successful disk save
