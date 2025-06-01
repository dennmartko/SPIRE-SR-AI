import os
import gc
import warnings

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
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
    parent_out_dir = r"/mnt/g/data/PhD Projects/SR" #r"/scratch/p317470/SRHerschel500/data/processed" # r"/scratch/p317470/SRHerschel500/data/processed" #r"/scratch-shared/dkoopmans" # Output directory of Data, change if needed to
    dataset_dir_name = "120deg2_shark_sides" # Directory name of generated dataset
    dir_to_data_maps = r"/mnt/g/data/PhD Projects/SR/sim_datamaps" #r"/scratch/p317470/SRHerschel500/data/raw/sim datamaps" # #r"/scratch/p317470/SRHerschel500/data/raw/sim datamaps" #r"/scratch-shared/dkoopmans/sim_datamaps" # Path to simulation datamaps

    # Instrument information
    instrument_prefixes = ["MIPS24", "SPIRE250", "SPIRE350", "SPIRE500", "SR_SPIRE500"] # prefixes used in filenames of datamaps
    instrument_names = ["Spitzer MIPS", "Herschel SPIRE", "Herschel SPIRE", "Herschel SPIRE", "SR Herschel SPIRE"] # Names listed on plots

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
    N_CPU = 12 # Number of cores available for multi processing, Usually TOTAL CORES - 1
    N_CPU_INTERP = 5 # Number of cores for interpolation, CAUTION: memory explodes with the number of cores

    # # Do not change!
    # CRVAL_offset1 = 0.64
    # CRVAL_offset2 = 0.67

    # Indicate whether to include noise in the output images.
    include_noise = True

######################
###  WORKING CLASS ###
######################

class ProcessDataSet():
    def __init__(self, dataset_prefix):
        self.dataset_prefix = dataset_prefix

        self.fname = lambda idx: f"{self.dataset_prefix}_" + f"{Config.instrument_prefixes[idx]}" + "_smoothed_Jy_beam.fits"
        
        # Configuration dictionary for generating cutouts
        self.gen_cutout_config = {"Training": {"color": 'white', "pos_center": [], "save_path": None}, 
                             "Validation": {"color": 'orange', "pos_center": [], "save_path": None},
                               "Test" : {"color": 'green', "pos_center": [], "save_path": None}}

        # Final configurations
        self.prepare_dirs()
        self.configure_cutouts()

    def prepare_dirs(self):
        # Grab the output directory and prepare subdirectories
        cdir = Config.parent_out_dir
        base_fig_dir = os.path.join(cdir, f"{Config.dataset_dir_name} - PreProcFigs")
        base_data_dir = os.path.join(cdir, f"{Config.dataset_dir_name}")

        # Define primary directories for current individual dataset
        self.fig_path = os.path.join(base_fig_dir, self.dataset_prefix)
        self.data_path = os.path.join(base_data_dir, self.dataset_prefix)

        # Define second-level directories
        self.train_path = os.path.join(self.data_path, "Train")
        self.val_path = os.path.join(self.data_path, "Validation")
        self.test_path = os.path.join(self.data_path, "Test")

        # Update the configuration with save paths
        self.gen_cutout_config["Training"]["save_path"] = self.train_path
        self.gen_cutout_config["Validation"]["save_path"] = self.val_path
        self.gen_cutout_config["Test"]["save_path"] = self.test_path

        # Helper function to create directory if it does not exist
        def create_dir(path):
            os.makedirs(path, exist_ok=True)

        # Create primary and second-level directories
        create_dir(base_fig_dir)
        create_dir(base_data_dir)
        create_dir(self.fig_path)
        create_dir(self.data_path)
        create_dir(self.train_path)
        create_dir(self.val_path)
        create_dir(self.test_path)

        # Create class subdirectories for each Train/Validation/Test subdirectory
        for cl in Config.class_names:
            for subset in [self.train_path, self.val_path, self.test_path]:
                create_dir(os.path.join(subset, cl))

        # Create mask subdirectories for target classes used by the loss functions
        for i, cl_type in enumerate(Config.class_types):
            if cl_type == 'target':
                for subset in [self.train_path, self.val_path]:
                    create_dir(os.path.join(subset, f"{Config.class_names[i]}_mask"))


    def configure_cutouts(self):
        # Load a temporary datamap, use the highest spatial/angular resolution datamap
        hdu = fits.open(os.path.join(Config.dir_to_data_maps, self.fname(0)), memmap=False)
        hdr = hdu[0].header

        w = WCS(hdr)

        # Image dimensions (FITS data shape is (ny, nx))
        ny, nx = hdu[0].data.shape[0], hdu[0].data.shape[1]
        nx_deg, ny_deg = nx*abs(hdr["CDELT1"]), ny*abs(hdr["CDELT2"])

        # Compute centers along X and Y in pixel coordinates
        center_offset_x = Config.input_cutout_dims[0] // 2 * 1/3600
        center_offset_y = Config.input_cutout_dims[1] // 2 * 1/3600

        centers_x = np.arange(center_offset_x, nx_deg - center_offset_x, Config.input_cutout_dims[0] * 1/3600)
        centers_y = np.arange(center_offset_y, ny_deg - center_offset_y, Config.input_cutout_dims[1] * 1/3600)

        # Compute meshgrid of center coordinates for the cutouts
        grid_x, grid_y = np.meshgrid(centers_x, centers_y)

        grid_x, grid_y = grid_x / abs(hdr["CDELT1"]), grid_y / abs(hdr["CDELT2"])

        # Compute the corresponding ra and dec coordinates
        grid_ra, grid_dec = w.all_pix2world(grid_x, grid_y, 0)

        fig_centers = plt.figure(figsize=(8, 8))
        ax_centers = fig_centers.add_subplot(111, projection=w)
        # Display the datamap in grayscale as a background
        ax_centers.imshow(hdu[0].data, origin='lower', cmap='gray', alpha=0.7)
        # Overplot the cutout center positions as red circles (using world coordinates)
        ax_centers.scatter(grid_ra.flatten(), grid_dec.flatten(), s=50, edgecolor='red', facecolor='none', transform=ax_centers.get_transform('world'))
        ax_centers.set_title("Cutout Centers on the Sky")
        # Save the figure in the figures directory defined in the object
        plt.savefig(os.path.join(self.fig_path, "cutout_centers_overplot.png"))
        plt.close(fig_centers)

        # Create class-wide list containing the Skycoordinates of the cutouts
        self.cutout_centers = [SkyCoord(ra*u.degree, dec*u.degree, frame="fk5") for ra, dec in zip(grid_ra.flatten(), grid_dec.flatten())]

        # Now split the cutouts according to the training, validation and test split ratios
        indices = np.arange(0, len(self.cutout_centers), 1).tolist()

        train_indices, test_val_indices = train_test_split(indices, train_size=Config.split[0], test_size=Config.split[1] + Config.split[2], random_state=10)
        val_indices, test_indices = train_test_split(test_val_indices, train_size = Config.split[1]/(Config.split[1] + Config.split[2]), test_size=Config.split[2]/(Config.split[1] + Config.split[2]), random_state=10)

        self.gen_cutout_config["Training"]["pos_center"] = [self.cutout_centers[i] for i in train_indices]
        self.gen_cutout_config["Validation"]["pos_center"] = [self.cutout_centers[i] for i in val_indices]
        self.gen_cutout_config["Test"]["pos_center"] = [self.cutout_centers[i] for i in test_indices]

        hdu.close() # Close the HDU to free up memory

    def interp_data(self, datamaps, original_header, interp_pixscale):
        # Calculate the scaling factor from the original pixel scale to the desired interpolated pixel scale
        scale_factor = abs(original_header["CDELT1"]*3600) / interp_pixscale

        # Compute new dims
        new_naxis1 = int(np.round(original_header["NAXIS1"] * scale_factor + 0.5))
        new_naxis2 = int(np.round(original_header["NAXIS2"] * scale_factor + 0.5))
        
        # Manually compute new CRPIX
        new_crpix1 = (original_header["CRPIX1"] - 1) * scale_factor + 1
        new_crpix2 = (original_header["CRPIX2"] - 1) * scale_factor + 1

        # Create new WCS object for interpolated datamap
        w_interp = WCS(naxis=2)
        w_interp.wcs.crval = [original_header["CRVAL1"], original_header["CRVAL2"]]
        w_interp.wcs.crpix = [new_crpix1, new_crpix2]
        w_interp.wcs.cdelt = [np.sign(original_header["CDELT1"]) * interp_pixscale/3600, 
                              np.sign(original_header["CDELT2"]) * interp_pixscale/3600]
        w_interp.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w_interp.wcs.cunit = [u.deg, u.deg]


        # Interpolation using new header WCS
        interp_data = reproject_exact(
            (datamaps, original_header),
            w_interp.to_header(), 
            shape_out=(new_naxis2, new_naxis1),
            parallel=Config.N_CPU_INTERP, 
            return_footprint=False)

        # Replace NaNs, representing regions outside the original map, with zeros
        interp_data[np.isnan(interp_data)] = 0
        return w_interp, interp_data
    
    def add_noise(self, sigma_instr_noise, img_data):
        noisy_img = img_data + np.random.normal(0, sigma_instr_noise, (img_data.shape[0],img_data.shape[1]))
        return noisy_img

    def generate_cutouts(self, size, interp_datamap, w_interp, cl, cl_type, fwhm, pix_scale, purpose, ax, img_ID_iter = 0):
        # Get the Training/Validation/Testing config
        config = self.gen_cutout_config.get(purpose, "Training")

        # Spawn daemon processes for fast post-processing and FITS saving
        executor = ProcessPoolExecutor(max_workers=Config.N_CPU)

        futures = []

        for pos in config["pos_center"]:
            # Make cutout based on given center
            cutout = Cutout2D(interp_datamap, position=pos, size=size, wcs=w_interp, mode="strict", copy=True)

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

        # We process the datamaps one by one. 
        for idx, cl in enumerate(Config.class_names):
            # Context of the datamap
            with fits.open(os.path.join(Config.dir_to_data_maps, self.fname(idx)), memmap=False) as hdu:
                # define the datamap and add noise if indicated in the config class
                datamap = hdu[0].data if not Config.include_noise else self.add_noise(Config.instr_noise_l[idx], hdu[0].data)

                # Interpolate the datamap for the input classes.
                if Config.class_types[idx] == "input":
                    w_interp, interp_data = self.interp_data(datamap, hdu[0].header, Config.interp_pixscale_l[idx])
                    size = Config.input_cutout_dims
                else:
                    w_interp = WCS(hdu[0].header)
                    interp_data = datamap
                    size = Config.target_cutout_dims

                # Create an image of the interpolated data with projected cutout outlines for verification.
                fig_cutout_overlay, ax = self.create_cutout_projection_figure(interp_data, w_interp, idx)

                # Generate the Training/Validation/Test cutouts and store to disk
                _ = self.generate_cutouts(size, interp_data, w_interp, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Training", ax, img_ID_iter = 0)
                _ = self.generate_cutouts(size, interp_data, w_interp, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Validation", ax, img_ID_iter = 0)
                _ = self.generate_cutouts(size, interp_data, w_interp, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Test", ax, img_ID_iter = 0)

            # Store a cutout sample for a fixed ID for verification purposes
            cutout_size_native_map = int(np.round(Config.input_cutout_dims[0]*Config.interp_pixscale_l[idx]/(hdu[0].header["CDELT1"]*3600)))
            self.fov_interp_cutout_list.append(Cutout2D(interp_data, position=self.cutout_centers[0], size=size, wcs=w_interp, mode="strict").data)
            self.fov_original_cutout_list.append(Cutout2D(hdu[0].data, position=self.cutout_centers[0], size=(cutout_size_native_map, cutout_size_native_map), wcs=WCS(hdu[0].header), mode="strict").data)


            # Finalise the cutout overlay figure and save to disk in the figure folder
            ax.coords[0].set_format_unit('deg')

            fig_cutout_overlay.savefig(f"{self.fig_path}" + f"/CutoutPlot_On_{cl}.pdf", dpi=300)
            plt.close(fig_cutout_overlay)

            # Clear memory
            gc.collect()

        # Mosaic Plot
        self.class_mosaic_plot()

    def create_cutout_projection_figure(self, data_map, w_data_map, idx):
        # Initiate figure for cutout projection
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection=w_data_map)

        # Scale data once and define vmin / vmax conditions
        scaled_data = np.array(data_map) * 1000
        if Config.class_names[idx] == "24":
            vmax = 1
        elif Config.class_names[idx] == "500SR":
            vmax = 25
        else:
            vmax = 60

        im_ax = ax.imshow(scaled_data, cmap="viridis", vmin=0, vmax=vmax)

        # Configure axis labels and ticks
        ax.set_ylabel("DEC (deg)", fontsize=10)
        ax.set_xlabel("RA (deg)", fontsize=10)
        ax.tick_params()

        # Add a white box patch overlay
        white_box = patches.Rectangle(
            (0.01, 0.01), 0.45, 0.05, linewidth=1.5,
            edgecolor='#414a4c', facecolor='white',
            transform=ax.transAxes, zorder=10
        )
        ax.add_patch(white_box)

        # Determine text content based on class type
        instrument_name = Config.instrument_names[idx]
        if Config.class_types[idx] == "input":
            text_content = rf"{instrument_name} {Config.class_names[idx]}$\mu m$"
        else:
            text_content = rf"{instrument_name} {500}$\mu m$"

        # Place text box indicating instrument and wavelength
        ax.text(
            0.05, 0.03, text_content, color='#414a4c',
            transform=ax.transAxes, ha='left', va='center',
            fontsize=16, zorder=11
        )

        # Setup a colorbar
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1+0.02, pos.y0, 0.03, pos.y1 - pos.y0])
        cbar = fig.colorbar(im_ax, cax=cax)
        cbar.set_label('mJy/beam', fontsize=10)
        cbar.ax.tick_params(labelsize=8)

        return fig, ax

    def class_mosaic_plot(self):
        dataset = "SPRITZ" if "SPRITZ" in self.dataset_prefix else ("SIDES" if "SIDES" in self.dataset_prefix else "SHARK")
        bands = [f"{band} μm" if band != "500SR" else f"SR 500 μm" for band in Config.class_names]
        fov = '256" × 256"'

        data = {'before': self.fov_original_cutout_list, 
                'after': self.fov_interp_cutout_list
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
                    im = ax.imshow(data[process][col]*1000, cmap='afmhot', vmin=0, vmax=50)
                else:
                    im = ax.imshow(data[process][col]*1000, cmap='afmhot', vmin=0, vmax=10)
                ax.text(0.05, 0.95, f'{band}\nFov: {fov}', color='white', fontsize=10, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.5))
                ax.axis('off')

                # Adding process label on the left side
                if col == 0:
                    ax.text(-0.68, 0.5, f'{dataset} ({process})', va='center', ha='left', fontsize=12, transform=ax.transAxes, weight='bold')

        # Adjusting colorbars with equal width
        cbar_width = 0.015

        # Adding colorbars for 24 μm
        cbar_24 = fig.colorbar(axes[0][0].images[0], ax=[axes[i][0] for i in range(2)], orientation='horizontal', pad=0.01, aspect=30, fraction=cbar_width)
        cbar_24.set_label('mJy/beam')

        # 250/350/500 micron colorbar shared across the columns
        # We will use the `cbar_width` to maintain equal width of colorbars
        # Adjust the `fraction` to maintain visual consistency
        cbar_250_350_500 = fig.colorbar(axes[0][1].images[0], ax=[axes[0][1], axes[0][2], axes[0][3], axes[1][1], axes[1][2], axes[1][3]], orientation='horizontal', pad=0.01, aspect=80, fraction=cbar_width)
        cbar_250_350_500.set_label('mJy/beam')

        # SR 500 μm colorbar
        cbar_sr_500 = fig.colorbar(axes[0][4].images[0], ax=[axes[i][4] for i in range(2)], orientation='horizontal', pad=0.01, aspect=30, fraction=cbar_width)
        cbar_sr_500.set_label('mJy/beam')
        fig.savefig(f"{self.fig_path}" + f"/cutout_mosaic_overview.png", dpi=300)
        plt.close(fig)


##########################
###  CODE IS RUN HERE ####
##########################
SHARK = [f"SHARK_{i+1}" for i in range(0, 30)]
SIDES = [f"SIDES_{i+1}" for i in range(0, 30)]
#SPRITZ = [f"SPRITZ"]
prefixes = SIDES + SHARK # Prefixes of the datamaps. Check the code for "fname" for details on standard formatting of files. CTRL + F --> "fname"

if __name__ == "__main__":
    for i, prefix in tqdm(enumerate(prefixes), desc="Processing...", total=len(prefixes)):
        dataset = ProcessDataSet(prefix)
        dataset.run()

        del dataset;