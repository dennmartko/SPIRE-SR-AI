import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    sky_coords = wcs.all_pix2world(corners_pix[:, 1], corners_pix[:, 0], 0)
    # print(sky_coords)
    ra = normalize_ra(sky_coords[0])
    dec = sky_coords[1]

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

def cutout_overlaps_catalog(wcs, shape, cat_bounds):
    """Check if an image defined by WCS and shape overlaps a catalog."""
    img_bounds = get_image_bounds_from_wcs(wcs, shape)
    return bounds_overlap(img_bounds, cat_bounds)

def process_sim_file(fn, wcs_dict):
    """Return list of DataFrames for all cutouts overlapping this sim file."""
    path = os.path.join(sim_catalogs_path, fn)
    sim_cat = fits.open(path, memmap=False)[1].data

    # Calculate the catalog boundary once. Speeds up ~x100.
    cat_bounds = get_catalog_bounds(sim_cat['ra'], sim_cat['dec'])

    results = []
    # Iterate over the cutouts and find sources from the catalog belonging whose coordinates lie within these cutouts
    for file_id, w_cut in wcs_dict.items():
        # First, check whether the cutout overlaps with the catalog
        is_overlap = cutout_overlaps_catalog(w_cut, CUTOUT_SHAPE, cat_bounds)

        if is_overlap:
            # Convert all catalog coordinates to x, y coordinates using the cutout WCS
            x, y = w_cut.all_world2pix(sim_cat['ra'], sim_cat['dec'], 0, ra_dec_order=True)
            # Mask the sources that within the cutout region
            mask = (0 <= x) & (x < CUTOUT_SHAPE[1]) & (0 <= y) & (y < CUTOUT_SHAPE[0])
            if not mask.any():
                continue
            df_sub = pd.DataFrame(sim_cat[mask])
            df_sub['file_id'] = file_id
            results.append(df_sub)
    return results

if __name__ == '__main__':
    # ——— Config ———
    # Paths containing data
    sim_catalogs_path   = "/mnt/g/data/PhD Projects/SR/sim_catalogs"
    test_cutouts_path   = "/mnt/g/data/PhD Projects/SR/120deg2_shark_sides/Test/500SR"
    catalog_output_dir  = "/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim"
    catalog_file_name   = "120_deg2_shark_sides_input_test_catalog.fits"
    CUTOUT_SHAPE        = (256, 256)  # (ny, nx)
    max_workers         = 10 # Number of processes to use. Each process requires 500-1000MB. Noteto self: can further reduce memory usage by introducing flux cut.

    # Load all cutout WCS
    wcs_dict = load_cutout_wcs(test_cutouts_path)

    # Process each sim catalog in parallel
    sim_cat_files = [f for f in os.listdir(sim_catalogs_path) if f.lower().endswith('.fits')]
    all_entries = []

    with progress as prog, ProcessPoolExecutor(max_workers=max_workers) as executor:
        task = progress.add_task(f"Processing simulated catalogs...", total=len(sim_cat_files))
        futures = {executor.submit(process_sim_file, fn, wcs_dict): fn for fn in sim_cat_files}

        for future in as_completed(futures):
            all_entries.extend(future.result())
            prog.update(task, advance=1)

    # assemble the input catalog for the test set
    catalog_df = pd.concat(all_entries, ignore_index=True) if all_entries else pd.DataFrame()
    n_unique   = catalog_df['file_id'].nunique()
    print(f"Recovered {n_unique} / {len(wcs_dict)} cutouts")

    # save to FITS
    if not catalog_df.empty:

        tbl = Table.from_pandas(catalog_df)
        out_path = os.path.join(catalog_output_dir, catalog_file_name)
        tbl.write(out_path, overwrite=True)
        print("Wrote catalog to", out_path)