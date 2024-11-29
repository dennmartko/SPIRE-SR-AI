import numpy as np
import pandas as pd

from astropy.modeling import functional_models
from photutils.psf import PSFPhotometry, SourceGrouper, make_psf_model
from photutils.detection import DAOStarFinder

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