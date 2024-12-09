import os
import time

import numpy as np
import pandas as pd
import astropy.units as u

from astropy.coordinates import SkyCoord
from astroML.crossmatch import crossmatch_angular

def calculate_flux_statistics(x_values, y_values, num_bins):
    # Create an array to store the median and percentile values
    results = np.zeros((num_bins, 4))

    # Bin the x_values
    bin_edges = np.logspace(np.log10(2e-3), np.log10(100e-3), num_bins + 1, base=10)
    bin_indices = np.digitize(x_values, bin_edges)

    # Calculate median and 1 sigma percentile for each bin
    for i in range(1, num_bins + 1):
        in_bin_mask = (bin_indices == i)

        flux_bin = y_values[in_bin_mask]

        median_value = np.median(flux_bin)
        percentile_1sigma = np.percentile(flux_bin, [16, 84])

        results[i - 1] = [np.mean(bin_edges[i - 1:i + 1]), median_value, percentile_1sigma[0], percentile_1sigma[1]]

    return results

def construct_matched_catalog(cat1, cat2, target_cl, max_distance):
    '''Construct a matched catalog, where the second column contains the target catalog with which the first column is matched.
       This function can be used interchangeably for the Target and SR catalogs, only requiring column names S500 and S500SR respectively.
       max_radius is the maximum matching radius in degrees. The distances in the returned catalog are in arcseconds.'''
    # Match cat1 with cat2, using max_radius in arcseconds
    distance_to_cat2, indices_of_cat2 = crossmatch_angular(cat1[["ra", "dec"]].values, cat2[["ra", "dec"]].values, max_distance=max_distance)
    matches_mask_to_cat2 = ~np.isinf(distance_to_cat2)

    matched_cat_col1 = f"S{target_cl[:-2]}" if f"S{target_cl[:-2]}" in cat1.columns else f"S{target_cl}"
    matched_cat_col2 = f"S{target_cl}" if f"S{target_cl}" in cat2.columns else f"S{target_cl[:-2]}"

    matched_cat = pd.DataFrame({
        matched_cat_col1: cat1[matches_mask_to_cat2][matched_cat_col1].values,
        matched_cat_col2: cat2.iloc[indices_of_cat2[matches_mask_to_cat2]][matched_cat_col2].values,
        "distance": distance_to_cat2[matches_mask_to_cat2]*3600
    })
    return matched_cat

def cross_match_catalogs(source_cat, target_cat, flux_col_source, flux_col_target, search_radius=4):
    """
    Cross-matches two catalogs and keeps the sources that are closest in flux if there are multiple matches for one of the sources in the target catalog.

    Parameters:
    source_cat (pd.DataFrame): Source catalog with ra, dec, and flux columns.
    target_cat (pd.DataFrame): Target catalog with ra, dec, and flux columns.
    flux_col_source (str): Name of the flux column in the source catalog.
    flux_col_target (str): Name of the flux column in the target catalog.
    search_radius (float): Search radius in arcseconds. Default is 4 arcseconds.

    Returns:
    pd.DataFrame: A DataFrame with the matched fluxes from both catalogs.
    """

    # Convert to SkyCoord objects
    source_coords = SkyCoord(ra=source_cat['ra'].values * u.degree, dec=source_cat['dec'].values * u.degree)
    target_coords = SkyCoord(ra=target_cat['ra'].values * u.degree, dec=target_cat['dec'].values * u.degree)

    # Perform the search around sky with a given radius
    target_indices, source_indices, _, _ = source_coords.search_around_sky(target_coords, search_radius / 3600 * u.degree)

    # Create a DataFrame with matched indices and corresponding fluxes
    matches = pd.DataFrame({
        'target_idx': target_indices,
        'source_idx': source_indices,
        flux_col_target: target_cat.iloc[target_indices][flux_col_target].values,
        flux_col_source: source_cat.iloc[source_indices][flux_col_source].values
    })

    # Calculate the absolute difference in flux
    matches['flux_diff'] = np.abs(matches[flux_col_source] - matches[flux_col_target])

    # Find the closest flux match for each unique target index
    closest_matches = matches.loc[matches.groupby('target_idx')['flux_diff'].idxmin()]

    # Create the matched catalog
    matched_catalog = pd.DataFrame({
        flux_col_source: closest_matches[flux_col_source],
        flux_col_target: closest_matches[flux_col_target]
    })
    return matched_catalog
