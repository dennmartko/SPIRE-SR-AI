import numpy as np
import pandas as pd
from astropy.wcs import WCS
from astropy.modeling import functional_models
from photutils.psf import PSFPhotometry, SourceGrouper, make_psf_model
from photutils.detection import DAOStarFinder

from concurrent.futures import ProcessPoolExecutor, as_completed

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

    group_labels = grouper._group_sources(init_sources["xcentroid"], init_sources["ycentroid"])
    unique, counts = np.unique(group_labels, return_counts=True)
    max_members = counts.max()

    # Prevent the fitter to be stuck in a very long fitting procedure
    ## If this is triggered, the images are bad anyways
    if max_members > 25:
        grouper = None

    # Set initial flux values
    init_sources["flux"] = init_sources["peak"]

    # We only fit sources >8" from the border
    init_sources = init_sources[(init_sources["xcentroid"] >= 8) & (init_sources["xcentroid"] <= (image_data.shape[0]-8))]
    init_sources = init_sources[(init_sources["ycentroid"] >= 8) & (init_sources["ycentroid"] <= (image_data.shape[0]-8))]
    
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

def process_single_prediction(fwhm, threshold, prediction, wcs, idx):
    phot = configure_psf_photometry(fwhm, threshold, np.squeeze(prediction))
    if phot is None or len(phot) == 0:
        return None

    phot = phot[phot["flags"] <= 1 ] #1
    source_flux = np.array(phot["flux_fit"])
    xpix = phot["x_fit"].tolist()
    ypix = phot["y_fit"].tolist()
    tr = np.transpose(np.vstack((phot["x_fit"], phot["y_fit"])))
    tr_world = wcs.wcs_pix2world(tr, 0)
    ra = tr_world[:, 0].tolist()
    dec = tr_world[:, 1].tolist()
    image_ids = [idx] * len(source_flux)
    
    return source_flux, xpix, ypix, ra, dec, image_ids

def construct_SR_source_catalog_deprecated(cl, fwhm, threshold, predictions, wcs_arr, N_CPU):
    SR_cat = {f"S{cl}": [], "ra": [], "dec": [], "xpix": [], "ypix": [], "ImageID": []}
    
    with ProcessPoolExecutor(max_workers=N_CPU) as executor:
        futures = [
            executor.submit(process_single_prediction, fwhm, threshold, predictions[i], wcs_arr[i], i)
            for i in range(len(predictions))
        ]
        
        for future in futures:
            result = future.result()
            if result:
                source_flux, xpix, ypix, ra, dec, image_ids = result
                SR_cat[f"S{cl}"].extend(source_flux)
                SR_cat["xpix"].extend(xpix)
                SR_cat["ypix"].extend(ypix)
                SR_cat["ra"].extend(ra)
                SR_cat["dec"].extend(dec)
                SR_cat["ImageID"].extend(image_ids)

    return pd.DataFrame(data=SR_cat, columns=list(SR_cat.keys()))

def source_extraction(data: np.ndarray, wcs: WCS, file_id: int, fwhm_pix: float, threshold: float = 2e-3) -> pd.DataFrame:
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
        'source_flux': pts['flux_fit'],
        'file_id': file_id
    })

    return df

def construct_source_catalog(images, wcs_dict, fwhm_pix, threshold=2e-3, N_CPU=10, progress=None, task_desc=None):

    results = []
    progress_task = progress.add_task(task_desc, total=int(images.shape[0])) if progress else None

    # We need to give file_id given that we retireve the processes as they get completed (which is not chronological)
    with ProcessPoolExecutor(max_workers=N_CPU) as executor:
        futures = [
            executor.submit(source_extraction, image, wcs_dict[file_id], file_id, fwhm_pix, threshold) for image, file_id in zip(images, wcs_dict.keys())
        ]
    
        for future in as_completed(futures):
            results.append(future.result())
            if progress is not None and progress_task is not None:
                progress.update(progress_task, advance=1)
                #progress.refresh()

    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame([])
    return df