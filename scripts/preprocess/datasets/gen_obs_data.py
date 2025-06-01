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



warnings.filterwarnings("ignore")


class Config:
    ''' Set default values for configuration parameters'''

    # The following lists have to be synchronized along the indices. However, positional permutations are allowed.
    interp_pixscale_l = [1., 1., 1., 1., 1.] # In arcseconds, for each image class
    fwhm_l = [5.7, 18.1, 24.9, 36.6, 7.9] #[5.7, 18.1, 24.9, 36.6, 7.9] # In arcseconds
    hdu_idx_l = [0, 1, 1, 1, 0] #[0, 1, 1, 1, 0] # indices of the fits header unit containing the data map
    class_names = ["24", "250", "350", "500", "450"] # ["24", "250", "350", "500", "450"] # Do not change names, use wavelengths.
    class_types = ["input", "input", "input", "input", "target"] #["input", "input", "input", "input", "target"]

    # Paths
    parent_out_dir = r"/mnt/g/data/PhD Projects/SR" # Output directory of data set, change if needed to
    dataset_dir_name = "cosmos" # Directory name of generated dataset
    dir_to_data_maps = r"/mnt/g/data/PhD Projects/SR/obs_datamaps" # Path to observation datamaps

    # Instrument information
    file_names = ["mips_24_GO3_sci_10_interp_bkg_subtracted.fits", "COSMOS-Nest_image_250_SMAP_v6.0_interp_bkg_subtracted.fits", "COSMOS-Nest_image_350_SMAP_v6.0_interp_bkg_subtracted.fits",
                    "COSMOS-Nest_image_500_SMAP_v6.0_interp_bkg_subtracted.fits", "SCS_450_corr.fits"] # "SCS_450_corr.fits"
    instrument_prefixes = ["MIPS24", "SPIRE250", "SPIRE350", "SPIRE500", "SCUBA450"]
    instrument_names = ["Spitzer MIPS", "Herschel SPIRE", "Herschel SPIRE", "Herschel SPIRE", "JCMT SCUBA-2"]

    # Cutout dimensions
    input_cutout_dims = (256, 256)
    target_cutout_dims = (256, 256)

    # The code relies heavily on multiprocessing to provide the data in a timely matter
    # By default, max 6 cores should be used for interpolation and can not be changed. This is due to the high memory usage.
    # However, on HPCs, this can be changed
    # Here, N_CPU indicates number of cores to be used for cutout generation: I/O (FITs saving) and source pre-detection.
    N_CPU = 12 # Number of cores available for multi processing, Usually TOTAL CORES - 1
    N_CPU_INTERP = 5 # Number of cores for interpolation, CAUTION: memory explodes with the number of cores

    # Define the area from which cutouts are generated
    ## In degrees

    # WHEN INCLUDING SCUBA-2, UNCOMMENT THESE COORDINATES
    START_X = 149.833
    START_Y = 2.16
    END_X =  150.333
    END_Y = 2.666

    # When super-resolving the Herschel COSMOS data, UNCOMMENT THESE COORDINATES
    # START_X = 149.35
    # START_Y = 1.5
    # END_X =  150.85
    # END_Y = 2.966


class ProcessDataSet():
    def __init__(self, dataset_prefix):
        self.dataset_prefix = dataset_prefix
       
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
        self.test_path = os.path.join(self.data_path, "Test")

        # Update the configuration with save paths
        self.gen_cutout_config["Test"]["save_path"] = self.test_path

        # Helper function to create directory if it does not exist
        def create_dir(path):
            os.makedirs(path, exist_ok=True)

        # Create primary and second-level directories
        create_dir(base_fig_dir)
        create_dir(base_data_dir)
        create_dir(self.fig_path)
        create_dir(self.data_path)
        create_dir(self.test_path)

        # Create class subdirectories for each Train/Validation/Test subdirectory
        for cl in Config.class_names:
            for subset in [self.test_path]:
                create_dir(os.path.join(subset, cl))

    def configure_cutouts(self):
        # Load a temporary datamap, use the highest spatial/angular resolution datamap
        hdu = fits.open(os.path.join(Config.dir_to_data_maps, Config.file_names[0]), memmap=False)
        hdr = hdu[0].header

        w = WCS(hdr)

        plate_scale_keys = ["CDELT1", "CDELT2"] if "CDELT1" in hdr else ["CD1_1", "CD2_2"]


        # Compute centers along X and Y in pixel coordinates
        center_offset_xpix = Config.input_cutout_dims[0] // 2
        center_offset_ypix = Config.input_cutout_dims[1] // 2

        # Convert the world coordinates (RA, DEC) defined in Config to pixel coordinates
        # The corners are: lower left, lower right, upper right, and upper left.
        corners_world = np.array([
            [Config.START_X, Config.START_Y],  # lower left
            [Config.END_X, Config.START_Y],    # lower right
            [Config.END_X, Config.END_Y],      # upper right
            [Config.START_X, Config.END_Y]     # upper left
        ])
        corners_pix = w.all_world2pix(corners_world, 0)

        # Determine the pixel boundaries from the world-to-pixel converted corners
        min_x, max_x = np.min(corners_pix[:, 0]), np.max(corners_pix[:, 0])
        min_y, max_y = np.min(corners_pix[:, 1]), np.max(corners_pix[:, 1])

        # Interpolation factor
        factor = abs(hdr[plate_scale_keys[0]] * 3600) / Config.interp_pixscale_l[0]

        # Generate pixel coordinate arrays stepping by center offset values
        x_vals = np.arange(min_x + center_offset_xpix, max_x - center_offset_xpix, Config.input_cutout_dims[0]/factor)
        y_vals = np.arange(min_y + center_offset_ypix, max_y - center_offset_ypix, Config.input_cutout_dims[1]/factor)

        # Create the meshgrid in pixel coordinates
        grid_pix_x, grid_pix_y = np.meshgrid(x_vals, y_vals)

        # Convert the pixel grid to world coordinates (RA, Dec)
        grid_ra, grid_dec = w.all_pix2world(grid_pix_x, grid_pix_y, 0)

        fig_centers = plt.figure(figsize=(8, 8))
        ax_centers = fig_centers.add_subplot(111, projection=w)
        # Display the datamap in grayscale as a background
        ax_centers.imshow(hdu[Config.hdu_idx_l[0]].data, origin='lower', cmap='gray', alpha=0.7)
        # Overplot the cutout center positions as red circles (using world coordinates)
        ax_centers.scatter(grid_ra.flatten(), grid_dec.flatten(), s=50, edgecolor='red', facecolor='none', transform=ax_centers.get_transform('world'))
        ax_centers.set_title("Cutout Centers on the Sky")
        # Save the figure in the figures directory defined in the object
        plt.savefig(os.path.join(self.fig_path, "cutout_centers_overplot.png"))
        plt.close(fig_centers)

        # Create class-wide list containing the Skycoordinates of the cutouts
        self.cutout_centers = [SkyCoord(ra*u.degree, dec*u.degree, frame="fk5") for ra, dec in zip(grid_ra.flatten(), grid_dec.flatten())]

        self.gen_cutout_config["Test"]["pos_center"] = self.cutout_centers

        hdu.close() # Close the HDU to free up memory

    def interp_data(self, datamaps, original_header, interp_pixscale):
        # Check if original map uses CDELT1/CDELT2 or CD1_1/CD2_2
        plate_scale_keys = ["CDELT1", "CDELT2"] if "CDELT1" in original_header else ["CD1_1", "CD2_2"]
        
        # Calculate the scaling factor from the original pixel scale to the desired interpolated pixel scale
        scale_factor = abs(original_header[plate_scale_keys[0]]*3600) / interp_pixscale

        # Compute new dims, add 0.5 since scale_factor may not be an integer
        new_naxis1 = int(np.round(original_header["NAXIS1"] * scale_factor + 0.5))
        new_naxis2 = int(np.round(original_header["NAXIS2"] * scale_factor + 0.5))

        # Manually compute new CRPIX, note astropy uses 1-based indexing
        new_crpix1 = (original_header["CRPIX1"] - 1) * scale_factor + 1
        new_crpix2 = (original_header["CRPIX2"] - 1) * scale_factor + 1

        # Create new WCS object for interpolated datamap
        w_interp = WCS(naxis=2)
        w_interp.wcs.crval = [original_header["CRVAL1"], original_header["CRVAL2"]]
        w_interp.wcs.crpix = [new_crpix1, new_crpix2]
        w_interp.wcs.cdelt = [np.sign(original_header[plate_scale_keys[0]]) * interp_pixscale/3600, 
                              np.sign(original_header[plate_scale_keys[1]]) * interp_pixscale/3600]
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
        # Most important for observational data maps
        interp_data[np.isnan(interp_data)] = 0
        return w_interp, interp_data
        
    def generate_cutouts(self, size, interp_datamap, w_interp, cl, cl_type, fwhm, pix_scale, purpose, ax, img_ID_iter = 0):
        # Get the Testing config
        config = self.gen_cutout_config.get(purpose)

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
        self.fov_interp_cutout_list = []
        self.fov_original_cutout_list = []

        # We process the datamaps one by one. 
        for idx, (cl, hdu_index) in enumerate(zip(Config.class_names, Config.hdu_idx_l)):
            # Context of the datamap
            with fits.open(os.path.join(Config.dir_to_data_maps, Config.file_names[idx]), memmap=False) as hdu:
                # define the datamap and add noise if indicated in the config class
                datamap = hdu[hdu_index].data

                # Interpolate the datamap for the input classes.
                if Config.class_types[idx] == "input":
                    w_interp, interp_data = self.interp_data(datamap, hdu[hdu_index].header, Config.interp_pixscale_l[idx])
                    size = Config.input_cutout_dims
                else:
                    w_interp = WCS(hdu[hdu_index].header)
                    interp_data = datamap
                    size = Config.target_cutout_dims

                # Create an image of the interpolated data with projected cutout outlines for verification.
                fig_cutout_overlay, ax = self.create_cutout_projection_figure(interp_data, w_interp, idx)

                # Generate the Training/Validation/Test cutouts and store to disk
                _ = self.generate_cutouts(size, interp_data, w_interp, cl, Config.class_types[idx], Config.fwhm_l[idx], Config.interp_pixscale_l[idx], "Test", ax, img_ID_iter = 0)

            plate_scale_keys = ["CDELT1", "CDELT2"] if "CDELT1" in hdu[hdu_index].header else ["CD1_1", "CD2_2"]


            # Store a cutout sample for a fixed ID for verification purposes
            factor = abs(hdu[hdu_index].header[plate_scale_keys[0]] * 3600) / Config.interp_pixscale_l[idx]
            cutout_size_native_map = int(np.round(Config.input_cutout_dims[0] / factor + 0.5))
            self.fov_interp_cutout_list.append(Cutout2D(interp_data, position=self.cutout_centers[0], size=size, wcs=w_interp, mode="strict").data)
            self.fov_original_cutout_list.append(Cutout2D(hdu[hdu_index].data, position=self.cutout_centers[0], size=(cutout_size_native_map, cutout_size_native_map), wcs=WCS(hdu[hdu_index].header), mode="strict").data)


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
        elif Config.class_names[idx] == "450":
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
            text_content = rf"{instrument_name} {450}$\mu m$"

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
        dataset = "COSMOS"
        bands = [f"{band} μm" if band != "450" else f"450 μm" for band in Config.class_names]
        fov = '256" × 256"'

        data = {'before': self.fov_original_cutout_list, 
                'after': self.fov_interp_cutout_list
        }

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=len(bands), figsize=(15, 6))  # 2 rows for before and after, 5 columns for bands

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
        cbar_250_350_450 = fig.colorbar(axes[0][1].images[0], ax=[axes[0][1], axes[0][2], axes[0][3], axes[1][1], axes[1][2], axes[1][3]], orientation='horizontal', pad=0.01, aspect=80, fraction=cbar_width)
        cbar_250_350_450.set_label('mJy/beam')

        # 450 μm colorbar
        if len(axes[0]) > 4:
            cbar_sr_450 = fig.colorbar(axes[0][4].images[0], ax=[axes[i][4] for i in range(2)], orientation='horizontal', pad=0.01, aspect=30, fraction=cbar_width)
            cbar_sr_450.set_label('mJy/beam')
        fig.savefig(f"{self.fig_path}" + f"/cutout_mosaic_overview.png", dpi=300)
        plt.close(fig)


##########################
###  CODE IS RUN HERE ####
##########################
prefixes = ["cosmos_spitzer_spire_scuba"] # Name for subdirectory.
if __name__ == "__main__":
    for i, prefix in tqdm(enumerate(prefixes), desc="Processing...", total=len(prefixes)):
        dataset = ProcessDataSet(prefix)
        dataset.run()

        del dataset;
