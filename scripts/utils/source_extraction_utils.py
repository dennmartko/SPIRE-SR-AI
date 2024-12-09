import numpy as np
import pandas as pd

from astropy.modeling import functional_models
from photutils.psf import PSFPhotometry, SourceGrouper, make_psf_model
from photutils.detection import DAOStarFinder

from concurrent.futures import ProcessPoolExecutor

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

def construct_SR_source_catalog(cl, fwhm, threshold, predictions, wcs_arr, N_CPU):
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