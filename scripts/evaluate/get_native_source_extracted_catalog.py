import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from astropy.modeling import functional_models
from photutils.psf import PSFPhotometry, SourceGrouper, make_psf_model
from photutils.detection import DAOStarFinder
import re
from astropy.table import Table

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

def load_cutout_wcs(path):
    """Return dict filename → WCS object."""
    fns = [f for f in os.listdir(path) if f.lower().endswith('.fits')]
    wcs_dict = {}

    with progress:
        task = progress.add_task("Loading cutout WCS objects...", total=len(fns))
        for fn in fns:
            hdr = fits.open(os.path.join(path, fn))[0].header
            match = re.search(r'_(\d+)\.fits$', fn)
            if match:
                file_id = int(match.group(1))
            wcs_dict[file_id] = WCS(hdr)
            progress.update(task, advance=1)
    print(f"Loaded WCS for {len(wcs_dict)} cutouts.")
    return wcs_dict

def normalize_ra(ra):
    """Normalize RA to [0, 360) degrees."""
    return np.mod(ra, 360)

def get_catalog_bounds(ra, dec):
    """Return RA/Dec bounds for a catalog, correctly handling RA wraparound."""
    ra = normalize_ra(np.array(ra))
    dec = np.array(dec)

    # Sort RA and find minimal interval that includes all points
    ra_sorted = np.sort(ra)
    ra_diff = np.diff(np.concatenate([ra_sorted, [ra_sorted[0] + 360]]))
    max_gap_index = np.argmax(ra_diff)

    # Exclude the largest gap — get the smallest bounding RA interval
    ra_min = ra_sorted[(max_gap_index + 1) % len(ra_sorted)]
    ra_max = ra_sorted[max_gap_index]

    ra_bounds = (ra_min, ra_max + 360) if ra_min > ra_max else (ra_min, ra_max)
    dec_bounds = (np.min(dec), np.max(dec))

    return ra_bounds, dec_bounds


def get_image_bounds_from_wcs(wcs, shape):
    """Given a WCS and image shape (ny, nx), return RA/Dec bounds."""
    ny, nx = shape

    # Define corner pixel coordinates
    corners_pix = np.array([
        [0, 0],           # bottom-left
        [0, nx - 1],      # bottom-right
        [ny - 1, 0],      # top-left
        [ny - 1, nx - 1], # top-right
    ])

    # Convert pixel to sky coordinates
    sky_coords = wcs.pixel_to_world(corners_pix[:, 1], corners_pix[:, 0])
    ra = normalize_ra(sky_coords.ra.deg)
    dec = sky_coords.dec.deg

    return get_catalog_bounds(ra, dec)


def ra_interval_overlap(ra1, ra2):
    """Check for overlap between two RA intervals, accounting for wraparound."""
    # Normalize to [0, 360)
    ra1_min, ra1_max = np.mod(ra1[0], 360), np.mod(ra1[1], 360)
    ra2_min, ra2_max = np.mod(ra2[0], 360), np.mod(ra2[1], 360)

    def expand_interval(rmin, rmax):
        if rmin <= rmax:
            return [(rmin, rmax)]
        else:
            # Wraps around 0: e.g., 350° to 10° becomes two segments
            return [(rmin, 360), (0, rmax)]

    seg1 = expand_interval(ra1_min, ra1_max)
    seg2 = expand_interval(ra2_min, ra2_max)

    # Check all combinations of segments for overlap
    for s1_min, s1_max in seg1:
        for s2_min, s2_max in seg2:
            if s1_max >= s2_min and s2_max >= s1_min:
                return True
    return False

def dec_interval_overlap(dec1, dec2):
    """Simple 1D overlap for Dec."""
    return not (dec1[1] < dec2[0] or dec2[1] < dec1[0])

def bounds_overlap(cat_bounds, img_bounds):
    """Check for RA/Dec overlap with proper RA wraparound."""
    ra_cat, dec_cat = cat_bounds
    ra_img, dec_img = img_bounds
    return ra_interval_overlap(ra_cat, ra_img) and dec_interval_overlap(dec_cat, dec_img)

def cutout_overlaps_catalog(wcs, shape, ra_cat, dec_cat):
    """Check if an image defined by WCS and shape overlaps a catalog."""
    img_bounds = get_image_bounds_from_wcs(wcs, shape)
    cat_bounds = get_catalog_bounds(ra_cat, dec_cat)
    return bounds_overlap(img_bounds, cat_bounds)

# Define the custom 2D Gaussian PSF model
def configure_psf_photometry(fwhm, threshold, image_data, fit_shape=(5,5)):
    # Calculate standard deviations from FWHM
    x_stddev = fwhm / 2.355
    y_stddev = fwhm / 2.355
    
    # Create an unprepared Gaussian 2D model
    psf_model_unprepared = functional_models.Gaussian2D(
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=0
    )
    
    # Prepare the PSF model with appropriate parameters
    psf_model = make_psf_model(
        psf_model_unprepared,
        x_name="x_mean",
        y_name="y_mean",
        flux_name="amplitude",
        normalize=False
    )
    
    # Initialize the SourceGrouper
    grouper = SourceGrouper(min_separation=1.5 * fwhm)
    
    # Use DAOStarFinder to detect initial sources
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold)
    init_sources = finder(image_data)
    
    # Intercept None types
    if init_sources is None:
        return None

    # Set initial flux values
    init_sources["flux"] = init_sources["peak"]

    init_sources = init_sources[(init_sources["xcentroid"] >= 0) & (init_sources["xcentroid"] <= (image_data.shape[1]))]
    init_sources = init_sources[(init_sources["ycentroid"] >= 0) & (init_sources["ycentroid"] <= (image_data.shape[0]))]
    
    # Configure PSFPhotometry without finder
    psfphot = PSFPhotometry(
        psf_model=psf_model,
        fit_shape=fit_shape,
        finder=None,
        grouper=grouper
    )
    
    # Perform photometry on the image data
    phot = psfphot(image_data, init_params=init_sources)
    return phot

def source_extraction(data: np.ndarray, wcs: WCS, fwhm_pix: float, threshold: float = 2e-3) -> pd.DataFrame:
    """
    Perform PSF photometry and return a catalog DataFrame with 'ra','dec','source_flux'.
    Returns None if no sources found.
    """
    phot_table = configure_psf_photometry(fwhm_pix, threshold, data)
    if phot_table is None or len(phot_table) == 0:
        return None

    # Filter by flags and extract positions and fluxes
    mask = phot_table['flags'] <= 1
    if not np.any(mask):
        return None

    pts = phot_table[mask]
    coords = np.vstack((pts['x_fit'], pts['y_fit'])).T
    world = wcs.wcs_pix2world(coords, 0)
    df = pd.DataFrame({
        'ra': world[:, 0],
        'dec': world[:, 1],
        'source_flux': pts['flux_fit']
    })

    return df

def extract_catalog_from_datamap(fn: str, fwhm: float, data_maps_dir: str, wcs_dict: dict):
    path = os.path.join(data_maps_dir, fn)
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    wcs_datamap = WCS(header)

    # Convert FWHM from arcsec to pixels
    pix_scale = abs(header['CDELT1']) * 3600 # may throw error for newer versions, use projection func from astropy to make it return pix_scale regardless of convention
    fwhm_pix = fwhm / pix_scale

    # Run extraction
    df_cat = source_extraction(data, wcs_datamap, fwhm_pix)
    results = []

    # Iterate over the cutouts and find sources from the catalog belonging whose coordinates lie within these cutouts
    for file_id, w_cut in wcs_dict.items():
        # First, check whether the cutout overlaps with the catalog
        is_overlap = cutout_overlaps_catalog(w_cut, CUTOUT_SHAPE, df_cat['ra'].values, df_cat['dec'].values)

        if is_overlap:
            # Convert all catalog coordinates to x, y coordinates using the cutout WCS
            x, y = w_cut.all_world2pix(df_cat['ra'], df_cat['dec'], 0, ra_dec_order=True)
            # Mask the sources that within the cutout region
            mask = (0 <= x) & (x < CUTOUT_SHAPE[1]) & (0 <= y) & (y < CUTOUT_SHAPE[0])
            if not np.any(mask):
                continue
            df_sub = df_cat[mask].copy()
            df_sub['file_id'] = file_id
            results.append(df_sub)
    return results

def get_native_source_extracted_catalog(data_maps_dir: str, wcs_cut: dict, bands: list, fwhm_list: list, out_files: list, max_workers: int = 10):

    for band, fwhm, out_file in zip(bands, fwhm_list, out_files):
        pattern = os.path.join(data_maps_dir, f"*_{band}*.fits")
        files = [os.path.basename(f) for f in glob.glob(pattern) if '_SR_' not in f]

        all_entries = []
        with progress, ProcessPoolExecutor(max_workers=max_workers) as executor:
            task = progress.add_task(f"Extracting sources for {band}...", total=len(files))
            futures = {executor.submit(extract_catalog_from_datamap, fname, fwhm, data_maps_dir, wcs_cut): fname for fname in files}

            for future in as_completed(futures):
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                except Exception as e:
                    print(f"Error processing {futures[future]}: {e}")
                finally:
                    progress.update(task, advance=1)

        if all_entries:
            df_out = pd.concat(all_entries, ignore_index=True)
            df_out.rename(columns={'source_flux': 'source_flux_native'}, inplace=True)
            out_path = os.path.join(catalog_output_dir, out_file)
            Table.from_pandas(df_out).write(out_path, format='fits', overwrite=True)
            print(f"Saved catalog with {len(df_out)} entries to {out_file}.")

            n_unique = df_out['file_id'].nunique()
            print(f"Recovered {n_unique} / {len(wcs_cut)} cutouts")

            # Identify and print the list of cutouts with no sources to verify
            recovered_cutouts = set(df_out['file_id'].unique())
            all_cutouts = set(wcs_cut.keys())
            missing_cutouts = sorted(all_cutouts - recovered_cutouts)
            print(f"Cutouts with no sources ({len(missing_cutouts)}): {missing_cutouts}")

        else:
            print(f"No sources found for band {band}.")


if __name__ == '__main__':
    # ——— Config ———
    # Paths containing data
    sim_catalogs_path   = "/mnt/g/data/PhD Projects/SR/sim_catalogs"
    test_cutouts_path   = "/mnt/g/data/PhD Projects/SR/120deg2_shark_sides/Test/500SR"
    catalog_output_dir  = "/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim"
    data_maps_dir       = "/mnt/g/data/PhD Projects/SR/sim_datamaps"
    CUTOUT_SHAPE        = (256, 256)  # (ny, nx)

    # Indicate for which bands a catalog should be calculated
    bands = ["SPIRE500"]

    # Indicate corresponding FWHM for source extraction
    FWHM = [36.6] # In arcseconds

    # Indicate the output file name
    out_files = [f"{band}_native_catalog.fits" for band in bands]

    # Obtain the cutout wcs objects together with file_id
    wcs_cut = load_cutout_wcs(test_cutouts_path)

    # Get the native source extracted catalog(s)
    get_native_source_extracted_catalog(data_maps_dir=data_maps_dir, wcs_cut=wcs_cut, bands=bands, fwhm_list=FWHM, out_files=out_files, max_workers = 12)