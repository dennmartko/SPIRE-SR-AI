import os
import gc
import warnings

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from sklearn.model_selection import train_test_split
from reproject import reproject_exact, reproject_adaptive, reproject_interp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from scripts.utils.file_utils import save_input_image_with_header_to_fits, save_target_image_catalog_to_fits

warnings.filterwarnings("ignore")

########################
#####   CONFIG   #######
########################

class Config:
      ''' Set default values for configuration parameters'''

      # The following lists have to be synchronized along the indices. However, positional permutations are allowed.
      interp_pixscale_l = [1., 1., 1., 1., 1] # In arcseconds, for each image class
      instr_noise_l = [14e-6, 2e-3, 2e-3, 2e-3, 0.]#[14e-6, 2e-3, 2e-3, 2e-3, 0.] #[14e-6, 3e-3, 3e-3, 3e-3, 0.] #[14e-6, 2e-3, 2e-3, 2e-3, 0.]# In Jy
      fwhm_l = [5.7, 18.1, 24.9, 36.6, 7.9] # In arcseconds
      class_names = ["24", "250", "350", "500", "500SR"] # Do not change names, permutations are allowed!
      class_types = ["input", "input", "input", "input", "target"]

      # Paths
      parent_out_dir = r"/mnt/d/SPIRE-SR-AI/data/processed" #r"/scratch/p317470/SRHerschel500/data/processed" # r"/scratch/p317470/SRHerschel500/data/processed" #r"/scratch-shared/dkoopmans" #r"E:\SRHerschel500\DATA" # Output directory of Data, change if needed to
      dataset_dir_name = "50deg_shark_sides_spritz" # Directory name of generated dataset
      dir_to_data_maps = r"/mnt/d/SRHerschel500/data/raw/sim datamaps" #r"/scratch/p317470/SRHerschel500/data/raw/sim datamaps" # #r"/scratch/p317470/SRHerschel500/data/raw/sim datamaps" #r"/mnt/d/SRHerschel500/data/raw/sim datamaps" #r"/scratch-shared/dkoopmans/sim_datamaps" #r"E:\SRHerschel500\RAW DATA\sim datamaps" # Path to simulation datamaps

      # Instrument information
      instrument_l = ["MIPS24", "SPIRE250", "SPIRE350", "SPIRE500", "SR_SPIRE500"]
      instrument_names = ["Spitzer MIPS", "Herschel SPIRE", "Herschel SPIRE", "Herschel SPIRE", "SR Herschel SPIRE"]

      # Cutout dimensions
      input_cutout_dims = (256, 256)
      target_cutout_dims = (256, 256)

      # Global Configuration for Training, Validation, and Testing (order is important)
      # Has to sum to 1.0 and individual 0.0 ratios are not possible --> either use very small value or use gen_obs_data.py which is more general to generate a full (Test) set
      split = [0.8, 0.1, 0.1]

      # The code relies heavily on multiprocessing to provide the data in a timely matter
      # By default, max 6 cores should be used for interpolation and can not be changed. This is due to the high memory usage.
      # However, on HPCs, this can be changed
      # Here, N_CPU indicates number of cores to be used for cutout generation: I/O (FITs saving) and source pre-detection.
      N_CPU = 10 # Number of cores available for multi processing, Usually TOTAL CORES - 1
      N_CPU_INTERP = 6 # Number of cores for interpolation, CAUTION: memory explodes with the number of cores

      # Do not change!
      CRVAL_offset1 = 0.64
      CRVAL_offset2 = 0.67

######################
###  WORKING CLASS ###
######################

class ProcessDataSet():
    def __init__(self, prefix):
        self.prefix = prefix

        self.fname = lambda idx: f"{self.prefix}_" + f"{Config.instrument_l[idx]}" + "_smoothed_Jy_beam.fits"
        
        # Configuration dictionary for generating cutouts
        self.gen_cutout_config = {"Training": {"color": 'white', "pos_center": [], "save_path": None}, 
                             "Validation": {"color": 'orange', "pos_center": [], "save_path": None},
                               "Test" : {"color": 'green', "pos_center": [], "save_path": None}}

        # Final configurations
        self.prepare_dirs()
        self.configure_cutouts()

    def prepare_dirs(self):
        # Grab the current directory and prepare subdirectories
        cdir = Config.parent_out_dir
        self.fig_path = os.path.join(os.path.join(cdir, f"{Config.dataset_dir_name} - PreProcFigs"), self.prefix)
        self.data_path = os.path.join(os.path.join(cdir, f"{Config.dataset_dir_name}"), self.prefix)
        self.train_path = os.path.join(self.data_path, "Train")
        self.val_path = os.path.join(self.data_path, "Validation")
        self.test_path = os.path.join(self.data_path, "Test")

        self.gen_cutout_config["Training"]["save_path"] = self.train_path
        self.gen_cutout_config["Validation"]["save_path"] = self.val_path
        self.gen_cutout_config["Test"]["save_path"] = self.test_path

        # First time object creation --> create DIRS for saving
        if not os.path.isdir(os.path.join(cdir, f"{Config.dataset_dir_name} - PreProcFigs")):
            os.mkdir(os.path.join(cdir, f"{Config.dataset_dir_name} - PreProcFigs"))
        if not os.path.isdir(os.path.join(cdir, f"{Config.dataset_dir_name}")):
            os.mkdir(os.path.join(cdir, f"{Config.dataset_dir_name}"))
        if not os.path.isdir(self.fig_path):
            os.mkdir(self.fig_path)
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
        if not os.path.isdir(self.test_path):
            os.mkdir(self.test_path)

        # class subdirs
        for cl in Config.class_names:
            train_FITS_path = os.path.join(self.train_path, f"{cl}")
            val_FITS_path = os.path.join(self.val_path, f"{cl}")
            test_FITS_path = os.path.join(self.test_path, f"{cl}")

            if not os.path.isdir(train_FITS_path):
                os.mkdir(train_FITS_path)
            if not os.path.isdir(val_FITS_path):
                os.mkdir(val_FITS_path)
            if not os.path.isdir(test_FITS_path):
                os.mkdir(test_FITS_path)

        # For each target class create a mask subdir
        for i, cl_type in enumerate(Config.class_types):
            if cl_type == 'target':
                train_FITS_path = os.path.join(self.train_path, f"{Config.class_names[i]}_mask")
                val_FITS_path = os.path.join(self.val_path, f"{Config.class_names[i]}_mask")

                if not os.path.isdir(train_FITS_path):
                    os.mkdir(train_FITS_path)
                if not os.path.isdir(val_FITS_path):
                    os.mkdir(val_FITS_path)

    def configure_cutouts(self):
        # Load a temporary datamap
        hdu = fits.open(os.path.join(Config.dir_to_data_maps, self.fname(0)), memmap=False)
        hdr = hdu[0].header

        # Calculate the distance between the centers of adjacent cutouts
        DISTANCE_BETWEEN_CENTERS = (Config.input_cutout_dims[0]*Config.interp_pixscale_l[0])/3600 # degrees
        
        START_X = hdr["CRVAL1"] - Config.CRVAL_offset1
        END_X = hdr["CRVAL1"] + Config.CRVAL_offset2
        START_Y = hdr["CRVAL2"] - Config.CRVAL_offset1
        END_Y = hdr["CRVAL2"] + Config.CRVAL_offset2

        ra_l = np.arange(START_X, END_X, DISTANCE_BETWEEN_CENTERS)
        dec_l = np.arange(START_Y, END_Y, DISTANCE_BETWEEN_CENTERS)

        self.cutout_center_l = [SkyCoord(ra*units.degree, dec*units.degree, frame="fk5") for ra in ra_l for dec in dec_l]
        indices = np.arange(0, len(self.cutout_center_l), 1).tolist()
        # Pre-computed training, validation and testing split
        train_indices, test_val_indices = train_test_split(indices, test_size=Config.split[1] + Config.split[2], random_state=10)
        val_indices, test_indices = train_test_split(test_val_indices, test_size=Config.split[2]/(Config.split[1] + Config.split[2]), random_state=10)

        self.gen_cutout_config["Training"]["pos_center"] = [self.cutout_center_l[i] for i in train_indices]
        self.gen_cutout_config["Validation"]["pos_center"] = [self.cutout_center_l[i] for i in val_indices]
        self.gen_cutout_config["Test"]["pos_center"] = [self.cutout_center_l[i] for i in test_indices]

    def interp_datamaps(self, datamaps, original_header, interp_pixscale):
        # Copy header
        new_header = original_header.copy()

        # Compute scaling
        original_pix_scale = abs(original_header["CDELT1"] * 3600)
        scaler = original_pix_scale / interp_pixscale

        # Update header with rescaled WCS
        w_hdu = WCS(original_header)
        new_hdr = w_hdu[::1/scaler, ::1/scaler].to_header()

        for key in ["CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]:
            new_header[key] = new_hdr[key]

        new_header["NAXIS1"] = int(np.round(original_header["NAXIS1"] * scaler + 0.5))
        new_header["NAXIS2"] = int(np.round(original_header["NAXIS2"] * scaler + 0.5))

        # Interpolation using new header WCS
        datamaps = reproject_exact((datamaps, original_header), new_header, parallel=Config.N_CPU_INTERP, return_footprint=False)
        # datamaps = reproject_interp((datamaps, original_header), new_header, parallel=Config.N_CPU_INTERP, return_footprint=False, order='bilinear')

        datamaps[np.isnan(datamaps)] = 0 # Regions outside the map set to 0.

        interp_wcs = WCS(new_header) # Create a new WCS object
        return interp_wcs, datamaps
    
    def add_noise(self, sigma_instr_noise, img_data):
        noisy_img = img_data.copy() + np.random.normal(0, sigma_instr_noise, (img_data.shape[0],img_data.shape[1]))
        return noisy_img
    
    def generate_cutouts(self, size, interp_datamap, interp_wcs, cl, cl_type, fwhm, pix_scale, purpose, ax, img_ID_iter = 0):
        # Get the Training/Validation/Testing config
        config = self.gen_cutout_config.get(purpose, "Training")

        # Spawn daemon processes for fast post-processing and FITS saving
        executor = ProcessPoolExecutor(max_workers=Config.N_CPU)

        futures = []

        for pos in config["pos_center"]:
            # Make cutout based on given center
            cutout = Cutout2D(interp_datamap, position=pos, size=size, wcs=interp_wcs, mode="strict", copy=True)

            # Cutout projection on datamap
            if ax is not None:
                cutout.plot_on_original(color=config["color"], ax=ax, alpha=0.8)

            # Post-processing and disk saving
            # Execute the queue'd jobs
            if cl_type != "target":
                future = executor.submit(save_input_image_with_header_to_fits, cutout.data, img_ID_iter, cl, cutout.wcs, config['save_path'])
            else:
                future = executor.submit(save_target_image_catalog_to_fits, cutout.data, img_ID_iter, cl, pix_scale, fwhm, cutout.wcs, purpose, config['save_path'])

            futures.append(future)

            # Increase the file number
            img_ID_iter += 1
        
        # confirm the closure of all jobs
        _ = [future.result() for future in futures]
        executor.shutdown(cancel_futures=True) # ensure proper shutdown
        return img_ID_iter

    def run(self):
        # List of datamaps with 15 x 15 arcminute coverage
        self.fov_interp_cutout_list = []
        self.fov_original_cutout_list = []

        # We process the data maps one by one. Only the first one will be used for validation, Testing.
        # The other maps are used for augmented training samples
        for idx, cl in enumerate(Config.class_names):
            # Load the map
            hdu = fits.open(os.path.join(Config.dir_to_data_maps, self.fname(idx)), memmap=False)
            original_size = int(np.round(Config.input_cutout_dims[0]*Config.interp_pixscale_l[idx]/(hdu[0].header["CDELT1"]*3600)))
            datamaps_w_noise = self.add_noise(Config.instr_noise_l[idx], hdu[0].data)
            if Config.class_types[idx] != "target":
                interp_wcs, interp_datamaps_w_noise = self.interp_datamaps(datamaps_w_noise, hdu[0].header, Config.interp_pixscale_l[idx])
                size = Config.input_cutout_dims
            else:
                # We do not have to interpolate the target class
                interp_wcs = WCS(hdu[0].header)
                interp_datamaps_w_noise = datamaps_w_noise
                size = Config.target_cutout_dims

            img_ID_iter = 0
            # Create a figure cutout overlay, to verify correct Training/Val/Test split for each class
            # Only do this for the first datamap, which will not be augmented and hence also used for validation and Testing
            # THEREFORE; augment_ID = 0 means NO AUGMENTATION
            # img_ID_iter tracks the file ID number for saving FITS files.
            fig_cutout_overlay, ax = self.create_cutout_projection_figure(interp_datamaps_w_noise, interp_wcs, idx)
            _ = self.generate_cutouts(size, interp_datamaps_w_noise, interp_wcs, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Training", ax, img_ID_iter = img_ID_iter)
            _ = self.generate_cutouts(size, interp_datamaps_w_noise, interp_wcs, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Validation", ax, img_ID_iter = 0)
            _ = self.generate_cutouts(size, interp_datamaps_w_noise, interp_wcs, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Test", ax, img_ID_iter = 0)
            
            # This is used for a Sample Mosaic plot showing pre-processing vs Post-processing
            self.fov_interp_cutout_list.append(Cutout2D(interp_datamaps_w_noise, position=self.cutout_center_l[0], size=size, wcs=interp_wcs, mode="strict"))
            self.fov_original_cutout_list.append(Cutout2D(hdu[0].data, position=self.cutout_center_l[0], size=(original_size, original_size), wcs=WCS(hdu[0].header), mode="strict"))

            # Save the cutout overlay figure
            ax.coords[0].set_format_unit('deg')

            fig_cutout_overlay.savefig(f"{self.fig_path}" + f"/CutoutPlot_On_Original_{cl}.pdf", dpi=300)
            plt.close(fig_cutout_overlay)

            # Clear memory
            gc.collect()

        # Mosaic Plot
        self.class_mosaic_plot()

    def create_cutout_projection_figure(self, data_map, w_data_map, idx):
        # Initiate figure for cutout projection
        fig_cutout_overlay = plt.figure(figsize=(10,10))
        ax = fig_cutout_overlay.add_subplot(111, projection=w_data_map)

        if Config.class_names[idx] == "24":
            im_ax = ax.imshow(np.array(data_map) * 1000, cmap="viridis", vmin=0, vmax=1)
        elif Config.class_names[idx] == "500SR":
            im_ax = ax.imshow(np.array(data_map) * 1000, cmap="viridis", vmin=0, vmax=25)
        else:
            im_ax = ax.imshow(np.array(data_map) * 1000, cmap="viridis", vmin=0, vmax=60)

        ax.set_ylabel("DEC (deg)", fontsize=10)
        ax.set_xlabel("RA (deg)", fontsize=10)
        
        ax.tick_params()
        white_box = patches.Rectangle((0.01, 0.01), 0.45, 0.05, linewidth=1.5, edgecolor='#414a4c', facecolor='white', transform=ax.transAxes, zorder=10)
        ax.add_patch(white_box)

        if Config.class_types[idx] != "target":
            text_content = rf"{Config.instrument_names[idx]} {Config.class_names[idx]}$\mu m$"
        else:
            text_content = rf"{Config.instrument_names[idx]} {500}$\mu m$"

        ax.text(0.05, 0.03, text_content, color='#414a4c', transform=ax.transAxes, ha='left', va='center', fontsize=16, zorder=11)
        cax = fig_cutout_overlay.add_axes([ax.get_position().x1+0.02, ax.get_position().y0, 0.03, ax.get_position().y1 - ax.get_position().y0])  # Adjust the position of the colorbar if needed
        cbar = fig_cutout_overlay.colorbar(im_ax, cax=cax)
        cbar.set_label('mJy/beam', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        return fig_cutout_overlay, ax


    def class_mosaic_plot(self):
        dataset = "SPRITZ" if "SPRITZ" in self.prefix else ("SIDES" if "SIDES" in self.prefix else "SHARK")
        bands = [f"{band} μm" if band != "500SR" else f"SR 500 μm" for band in Config.class_names]
        fov = '256" × 256"'

        data = {'before': [d.data for d in self.fov_original_cutout_list], 
                'after': [d.data for d in self.fov_interp_cutout_list]
        }

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))  # 2 rows for before and after, 5 columns for bands

        # Configure the figure
        fig.subplots_adjust(wspace=0.01, hspace=0.05)

        # Plot the data
        for process_idx, process in enumerate(['before', 'after']):
            for col, (band, ax) in enumerate(zip(bands, axes[process_idx])):
                if col == 0:
                    im = ax.imshow(data[process][col]*1000, cmap='afmhot', vmin=0, vmax=1)
                elif col >0 and col <4:
                    im = ax.imshow(data[process][col], cmap='afmhot', vmin=0, vmax=50/1000)
                else:
                    im = ax.imshow(data[process][col], cmap='afmhot', vmin=0, vmax=10/1000)
                ax.text(0.05, 0.95, f'{band}\nFov: {fov}', color='white', fontsize=10, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
                ax.axis('off')

                # Adding process label on the left side
                if col == 0:
                    ax.text(-0.68, 0.5, f'{dataset} ({process})', va='center', ha='left', fontsize=12, transform=ax.transAxes, weight='bold')

        # Adjusting colorbars with equal width
        cbar_width = 0.015

        # Adding colorbars for 24 μm and SR 500 μm
        cbar_24 = fig.colorbar(axes[0][0].images[0], ax=[axes[i][0] for i in range(2)], orientation='horizontal', pad=0.01, aspect=30, fraction=cbar_width)
        cbar_24.set_label('mJy/beam')

        # 250/350/500 micron colorbar shared across the columns
        # We will use the `cbar_width` to maintain equal width of colorbars
        # Adjust the `fraction` to maintain visual consistency
        cbar_250_350_500 = fig.colorbar(axes[0][1].images[0], ax=[axes[0][1], axes[0][2], axes[0][3], axes[1][1], axes[1][2], axes[1][3]], orientation='horizontal', pad=0.01, aspect=80, fraction=cbar_width)
        cbar_250_350_500.set_label('Jy/beam')

        # SR 500 micron colorbar
        cbar_sr_500 = fig.colorbar(axes[0][4].images[0], ax=[axes[i][4] for i in range(2)], orientation='horizontal', pad=0.01, aspect=30, fraction=cbar_width)
        cbar_sr_500.set_label('Jy/beam')
        fig.savefig(f"{self.fig_path}" + f"/class_mosaic_plot.png", dpi=300)
        plt.close(fig)


##########################
###  CODE IS RUN HERE ####
##########################
SHARK = [f"SHARK_{i+1}" for i in range(0, 12)]
SIDES = [f"SIDES_{i+1}" for i in range(0, 12)]
SPRITZ = [f"SPRITZ"]
prefixes = SPRITZ + SHARK + SIDES  # Prefixes of the datamaps. Check the code for "fname" for details on standard formatting of files. CTRL + F --> "fname"

if __name__ == "__main__":
    for i, prefix in tqdm(enumerate(prefixes), desc="Processing...", total=len(prefixes)):
        ProcessDataSet(prefix).run()