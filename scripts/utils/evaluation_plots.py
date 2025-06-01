import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


from scripts.utils.metrics import (
    calculate_flux_statistics, 
    cross_match_catalogs, 
    construct_matched_catalog)

# --- Implemented functions ---
def plot_image_grid(
    images,
    cat_sr_images,
    cat_target_images,
    region_size=16,
    vmins=None,
    vmaxs=None,
    cmap='viridis',
    pad_w=0.04,
    pad_h=0.04,
    region_color='red',
    save_path=None
):
    """
    Plot a grid of images with per-row colorbars of aligned height, inset zooms of a highlighted region,
    adjustable padding between images, row labels on the first column, and customizable region highlight color.
    """
    nrows, ncols, H, W = images.shape

    # Labels for each row to display on the first column
    row_labels = ['High-resolution', 'Super-resolved', 'Residual']

    # Normalize vmins/vmaxs to lists
    if vmins is None:
        vmins = [None] * nrows
    elif np.isscalar(vmins):
        vmins = [vmins] * nrows
    if vmaxs is None:
        vmaxs = [None] * nrows
    elif np.isscalar(vmaxs):
        vmaxs = [vmaxs] * nrows

    # Extra width ratio for colorbar column
    extra = 0.05
    fig_width = 4 * (ncols + extra)
    fig_height = 4 * nrows

    # Create grid with extra column for colorbars, set spacing
    fig, axes = plt.subplots(
        nrows,
        ncols + 1,
        figsize=(fig_width, fig_height),
        gridspec_kw={'width_ratios': [1] * ncols + [extra]}
    )
    fig.subplots_adjust(wspace=pad_w, hspace=pad_h)

    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            im = ax.imshow(images[i, j], vmin=vmins[i], vmax=vmaxs[i], cmap=cmap, origin='lower')
            ax.set_xticks([])
            ax.set_yticks([])

            # From the target catalog take the third brightest object
            cat_sr = cat_sr_images[j]
            cat_target = cat_target_images[j]

            cat_target = cat_target[np.argsort(cat_target[:, -1])]
            x0, y0 = int(np.round(cat_target[-2, 0], 0)) - region_size//2 , int(np.round(cat_target[-2, 1], 0)) - region_size//2

            # Draw rectangle around region
            rect = plt.Rectangle(
                (x0, y0),
                region_size,
                region_size,
                edgecolor=region_color,
                fill=False,
                linewidth=2
            )
            ax.add_patch(rect)

            # Inset zoom of the region
            axins = inset_axes(
                ax,
                width="30%",
                height="30%",
                loc='lower left',
                borderpad=0.5,
            )
            patch = images[i, j][y0:y0 + region_size, x0:x0 + region_size]
            axins.imshow(patch, vmin=vmins[i], vmax=vmaxs[i], cmap=cmap, origin='lower', interpolation=None)
            axins.set_xticks([])
            axins.set_yticks([])

            # Set inset border color
            for spine in axins.spines.values():
                spine.set_edgecolor(region_color)
                spine.set_linewidth(2)

            # Only overplot the sources for target/sr (apply for rows 1 and 2)
            if i >= 0:
                # Create an offset such that scatter point does not go beyond image border
                offset = 2
                # calculate all sources that fall in highlighted region
                cat_sr_region = cat_sr[(cat_sr[:, 0] > x0+offset) &
                                               (cat_sr[:, 0] < x0 + region_size - offset) &
                                               (cat_sr[:, 1] > y0 + offset) &
                                               (cat_sr[:, 1] < y0 + region_size - offset)]
                
                cat_target_region = cat_target[(cat_target[:, 0] > x0) &
                                               (cat_target[:, 0] < x0 + region_size - offset) &
                                               (cat_target[:, 1] > y0) &
                                               (cat_target[:, 1] < y0 + region_size - offset)]
            
                # Plot all positions in the highlighted region
                if i == 0:
                    axins.scatter(cat_target_region[:,0]-x0, cat_target_region[:, 1]-y0, marker='x', s=80, color='blue')
                if i >= 1:
                    axins.scatter(cat_sr_region[:,0]-x0, cat_sr_region[:, 1]-y0, marker='o', s=80, facecolors='none', edgecolors='green', linewidths=2)
                    axins.scatter(cat_target_region[:,0]-x0, cat_target_region[:, 1]-y0, marker='x', s=80, color='blue')
        # Set row label on first column
        ax_label = axes[i, 0]
        ax_label.set_ylabel(
            row_labels[i],
            rotation=90,
            labelpad=15,
            fontweight='bold',
            color='#333333',
            va='center',
            fontsize=16
        )

        # Colorbar axis in extra column
        cax = axes[i, -1]
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')
        cax.set_xticks([])
        cb.set_label('Flux (mJy)', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_binned_iqr(S_in_list, S_out_list, bool_scatterplots, legend_labels, xlabel, ylabel, colors=None, bins=10, save_path=None):
    """
    Plot binned median with IQR shading for 1 sigma on a log-scaled X axis.
    """
    plt.figure(figsize=(8, 6))

    # If no colors provided, use default cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for S_in, S_out, bool_scatterplot, label, color in zip(S_in_list, S_out_list, bool_scatterplots, legend_labels, colors):

        # Calculate the y-values
        flux_ratios = (S_out - S_in)/S_in

        flux_statistics = calculate_flux_statistics(S_in, flux_ratios, bins)
        centers, p50, p16, p84 = flux_statistics[:, 0], flux_statistics[:, 1], flux_statistics[:, 2], flux_statistics[:, 3]

        # Plot median line and IQR region
        plt.plot(
            centers, p50,
            linestyle='-', color=color, label=label
        )

        plt.fill_between(
            centers, p16, p84,
            alpha=0.3, color=color
        )

        if bool_scatterplot and sum(bool_scatterplots) <= 1: # Only 1 scatterplot can be shown
            plt.scatter(S_in, flux_ratios, s=0.25, color='black', alpha=0.1, marker='o')

    # Horizontal line at zero (ideal)
    plt.axhline(0, linestyle='--', linewidth=2, color='#0e1111', alpha=0.7, label="1:1")

    # Log-scale X axis
    plt.xscale('log')

    # Set axis limits
    plt.yticks(np.arange(-1, 1 + 0.2, 0.2))
    plt.xlim([2, 80])
    plt.ylim([-1, 1])

    # tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    # axis labels
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def contourplot_completeness_reliability(input_cat, output_cat,
                                  flux_col_input, flux_col_output,
                                  bins, levels, search_radius=4, save_path=None):
    """
    Computes completeness and reliability across flux bins and plots them
    with a linear color scale.
    """
        # Cross‑match catalogs
    matched_from_input = cross_match_catalogs(
        output_cat, input_cat, flux_col_output, flux_col_input, search_radius
    )
    matched_from_output = cross_match_catalogs(
        input_cat, output_cat, flux_col_input, flux_col_output, search_radius
    )

    # Prepare the table for the flux bins
    flux_bins = [(low, high) for low, high in zip(bins[:-1], bins[1:])]
    n_bins    = len(flux_bins)
    zero_list = [0] * n_bins
    nan_list  = [np.nan] * n_bins

    metrics_df = pd.DataFrame({
        'Flux bin':  flux_bins,
        'TPc':       zero_list,   # true positives for completeness
        'FNc':       zero_list,   # false negatives for completeness
        'C':         nan_list,    # completeness = TPc/(TPc+FNc)
        'cum_C':     nan_list,   # cumulative completeness
        'TPr':       zero_list,   # true positives for reliability
        'FPr':       zero_list,   # false positives for reliability
        'R':         nan_list,    # reliability = TPr/(TPr+FPr)
        'cum_R':     nan_list,   # cumulative reliability
    })

    # Compute bin centers, completeness, and reliability
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        # --- completeness side
        in_bin     = (input_cat[flux_col_input]  >= low) & (input_cat[flux_col_input]  < high)
        n_in       = in_bin.sum()
        tp_c       = ((matched_from_input[flux_col_input] >= low) &
                      (matched_from_input[flux_col_input] < high)).sum()
        fn_c       = n_in - tp_c

        # --- reliability side
        out_bin    = (output_cat[flux_col_output] >= low) & (output_cat[flux_col_output] < high)
        n_out      = out_bin.sum()
        tp_r       = ((matched_from_output[flux_col_output] >= low) &
                      (matched_from_output[flux_col_output] < high)).sum()
        fp_r       = n_out - tp_r

        # --- completeness & reliability fractions
        comp = tp_c / n_in if n_in > 0 else np.nan
        rel  = tp_r / n_out if n_out > 0 else np.nan

        # Fill row `idx`
        metrics_df.at[idx, 'TPc'] = tp_c
        metrics_df.at[idx, 'FNc'] = fn_c
        metrics_df.at[idx,  'C']  = comp

        metrics_df.at[idx, 'TPr'] = tp_r
        metrics_df.at[idx, 'FPr'] = fp_r
        metrics_df.at[idx,  'R']  = rel


    # Triangulate the points in C–R space
    valid = (~np.isnan(metrics_df['C'])) & (~np.isnan(metrics_df['R']))
    triang = mtri.Triangulation(metrics_df.loc[valid, 'C'], metrics_df.loc[valid, 'R'])
    bin_centers = np.array([0.5 * (low + high) for low, high in flux_bins])
    bin_centers = bin_centers[valid.values]

    fig, ax = plt.subplots(figsize=(6, 5))
    cntr = ax.tricontourf(triang, bin_centers,
                          levels=levels,
                          cmap='viridis')  # default linear normalization

    # Linear colorbar
    cbar = fig.colorbar(
        cntr,
        orientation='horizontal',
        pad=0.15,
        shrink=0.8,
        aspect=30
    )
    cbar.set_label('Flux bin center', labelpad=8)

    ax.set_xlabel('Completeness')
    ax.set_ylabel('Reliability')
    ax.set_title('Flux contours in Completeness–Reliability space')

    ax.set_xlim([0.6, 1.])
    ax.set_ylim([0.6, 1.])
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_completeness_reliability(input_cat, output_cat,
                                  flux_col_input, flux_col_output,
                                  bins, search_radius=4, save_path=None):
    """
    Computes completeness and reliability across flux bins and plots them
    with a linear color scale.
    """

    # Cross‑match catalogs
    matched_from_input = cross_match_catalogs(
        output_cat, input_cat, flux_col_output, flux_col_input, search_radius
    )
    matched_from_output = cross_match_catalogs(
        input_cat, output_cat, flux_col_input, flux_col_output, search_radius
    )

    # Prepare the table for the flux bins
    flux_bins = [(low, high) for low, high in zip(bins[:-1], bins[1:])]
    n_bins    = len(flux_bins)
    zero_list = [0] * n_bins
    nan_list  = [np.nan] * n_bins

    metrics_df = pd.DataFrame({
        'Flux bin':  flux_bins,
        'TPc':       zero_list,   # true positives for completeness
        'FNc':       zero_list,   # false negatives for completeness
        'C':         nan_list,    # completeness = TPc/(TPc+FNc)
        'cum_C':     nan_list,   # cumulative completeness
        'TPr':       zero_list,   # true positives for reliability
        'FPr':       zero_list,   # false positives for reliability
        'R':         nan_list,    # reliability = TPr/(TPr+FPr)
        'cum_R':     nan_list,   # cumulative reliability
    })

    # Compute bin centers, completeness, and reliability
    for idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        # --- completeness side
        in_bin     = (input_cat[flux_col_input]  >= low) & (input_cat[flux_col_input]  < high)
        n_in       = in_bin.sum()
        tp_c       = ((matched_from_input[flux_col_input] >= low) &
                      (matched_from_input[flux_col_input] < high)).sum()
        fn_c       = n_in - tp_c

        # --- reliability side
        out_bin    = (output_cat[flux_col_output] >= low) & (output_cat[flux_col_output] < high)
        n_out      = out_bin.sum()
        tp_r       = ((matched_from_output[flux_col_output] >= low) &
                      (matched_from_output[flux_col_output] < high)).sum()
        fp_r       = n_out - tp_r

        # --- completeness & reliability fractions
        comp = tp_c / n_in if n_in > 0 else np.nan
        rel  = tp_r / n_out if n_out > 0 else np.nan

        # Fill row `idx`
        metrics_df.at[idx, 'TPc'] = tp_c
        metrics_df.at[idx, 'FNc'] = fn_c
        metrics_df.at[idx,  'C']  = comp

        metrics_df.at[idx, 'TPr'] = tp_r
        metrics_df.at[idx, 'FPr'] = fp_r
        metrics_df.at[idx,  'R']  = rel

    # Compute cumulative True Positives, False Negatives etc
    metrics_df["TPr_cum"] = metrics_df["TPr"][::-1].cumsum()[::-1]
    metrics_df["FPr_cum"] = metrics_df["FPr"][::-1].cumsum()[::-1]
    metrics_df["TPc_cum"] = metrics_df["TPc"][::-1].cumsum()[::-1]
    metrics_df["FNc_cum"] = metrics_df["FNc"][::-1].cumsum()[::-1]

    # Do not compute cumulative metrics if there are no matches
    mask = (metrics_df["TPr"] + metrics_df["FPr"]) == 0
    metrics_df.loc[mask, "TPr_cum"] = np.nan
    metrics_df.loc[mask, "FPr_cum"] = np.nan

    for i in range(len(metrics_df)):
        # Compute the cumulative Completeness and Reliability
        den_C = metrics_df.loc[i, "TPc_cum"] + metrics_df.loc[i, "FNc_cum"]
        metrics_df.at[i, 'cum_C'] = metrics_df.loc[i, "TPc_cum"] / den_C if den_C > 0 else np.nan

        den_R = metrics_df.loc[i, "TPr_cum"] + metrics_df.loc[i, "FPr_cum"]
        metrics_df.at[i, 'cum_R'] = metrics_df.loc[i, "TPr_cum"] / den_R if den_R > 0 else np.nan

    # Append another column containing denominators
    metrics_df['den_C'] = metrics_df['TPc'] + metrics_df['FNc']
    metrics_df['den_R'] = metrics_df['TPr'] + metrics_df['FPr']
    metrics_df['den_C_cum'] = metrics_df['TPc_cum'] + metrics_df['FNc_cum']
    metrics_df['den_R_cum'] = metrics_df['TPr_cum'] + metrics_df['FPr_cum']

    # Plots
    threshold = 2. # mJy
    bin_centers = np.array([0.5 * (low + high) for low, high in flux_bins])
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))

    axs[0].errorbar(bin_centers, metrics_df['R'], yerr=np.sqrt(metrics_df['R'] * (1 - metrics_df['R'])/metrics_df['den_R']), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$R(S_{bin})$")
    
    axs[0].errorbar(bin_centers, metrics_df['cum_R'], yerr=np.sqrt(metrics_df['cum_R'] * (1 - metrics_df['cum_R'])/metrics_df['den_R_cum']), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$R(\geq S_{bin})$")
    
    axs[1].errorbar(bin_centers, metrics_df['C'], yerr=np.sqrt(metrics_df['C'] * (1 - metrics_df['C'])/metrics_df['den_C']), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$C(S_{bin})$")
    
    axs[1].errorbar(bin_centers, metrics_df['cum_C'], yerr=np.sqrt(metrics_df['cum_C'] * (1 - metrics_df['cum_C'])/metrics_df['den_C_cum']), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$C(\geq S_{bin})$")
    
    # Interpolate cumulative reliability and completeness at 10 mJy (bin_centers are in mJy)
    flux_threshold = 10.
    interp_R = np.interp(flux_threshold, bin_centers, metrics_df['cum_R'])
    interp_C = np.interp(flux_threshold, bin_centers, metrics_df['cum_C'])

    # Create annotation string
    annotation_R = f"R(>10mJy) = {interp_R:.2f}"
    annotation_C = f"C(>10mJy) = {interp_C:.2f}"

    # Place the annotation on both subplots
    axs[0].text(10, 0.50, annotation_R,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    axs[1].text(10, 0.50, annotation_C,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Set the x-axis to log scale and add a minor grid
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[0].xaxis.set_tick_params(labelbottom=True)
    axs[0].yaxis.set_tick_params(labelleft=True)
    axs[0].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')
    axs[1].xaxis.set_tick_params(labelbottom=True)
    axs[1].yaxis.set_tick_params(labelleft=True)
    axs[1].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')
    
    # Set the axis labels and title for the main plot
    axs[0].set_xlabel('Super-resolved Source Flux (mJy)', fontsize=10)
    axs[1].set_xlabel('Target Source Flux (mJy)', fontsize=10)
    
    axs[0].set_ylabel('Reliability', fontsize=10)
    axs[1].set_ylabel('Completeness', fontsize=10)
    
    axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].tick_params(direction='in', axis='x', which='both', labelsize=10)
    axs[1].tick_params(direction='in', axis='x', which='both', labelsize=10)
    axs[0].tick_params(axis='y', which='both', labelsize=10)
    axs[1].tick_params(axis='y', which='both', labelsize=10)
    
    axs[0].set_ylim([0., 1.02])
    axs[1].set_ylim([0., 1.02])
    axs[0].set_xlim([np.min(threshold)-0.1, np.max(bin_centers)+0.05])
    axs[1].set_xlim([np.min(threshold)-0.1, np.max(bin_centers)+0.05])
    axs[0].set_xticks(np.array([threshold, 10, 100]))
    axs[1].set_xticks(np.array([threshold, 10, 100]))
    
    axs[0].legend(loc='lower right', frameon=False)
    axs[1].legend(loc='lower right', frameon=False)
    
    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def pos_flux_plot(input_fluxes, output_fluxes, offsets, xbins, ybins,
                  xlabel=r"$(S_{SR} - S_{Target})/S_{Target}$",
                  ylabel="Offset ('')",
                  interpolation=True, save_path=None):
    '''Plots the positional offset against source flux reproduction.
       Shows systematics and focuses on the region around perfect performance.
       Assumes that the input fluxes are in mJy and the offsets are in arcseconds.
    '''
    # Compute relative flux error
    rel_flux_err = (output_fluxes - input_fluxes) / input_fluxes

    # 2D histogram
    h, xedges, yedges = np.histogram2d(rel_flux_err, offsets, bins=(xbins, ybins))
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    # Prepare colormap
    cmap = plt.colormaps['plasma'].with_extremes(bad=plt.colormaps['plasma'](0.))

    # Set up figure and axes
    fig = plt.figure(figsize=(7.5, 7.5))
    grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.5,
                        width_ratios=[1, 5, 5, 5, 5])
    ax_main = fig.add_subplot(grid[1:4, 1:4])
    ax_xhist = fig.add_subplot(grid[0, 1:4], sharex=ax_main)
    ax_yhist = fig.add_subplot(grid[1:4, 4], sharey=ax_main)
    ax_cb    = fig.add_subplot(grid[1:4, 0])

    # Adjust colorbar position
    cb_pos = ax_cb.get_position()
    main_pos = ax_main.get_position()
    ax_cb.set_position([cb_pos.x0 - 0.03,
                        main_pos.y0,
                        cb_pos.width,
                        main_pos.height])

    # Interpolate if requested
    if interpolation:
        xi = np.linspace(xcenters[0], xcenters[-1], 500)
        yi = np.linspace(ycenters[0], ycenters[-1], 500)
        interp = RegularGridInterpolator((xcenters, ycenters), h,
                                         bounds_error=False,
                                         fill_value=0, method='cubic')
        Xg, Yg = np.meshgrid(xi, yi, indexing='ij')
        h = interp(np.column_stack((Xg.ravel(), Yg.ravel()))).reshape(Xg.shape)
        h = np.clip(h, 0, None)  # Avoid negative values
        xcenters, ycenters = xi, yi
    else:
        xi, yi = xcenters, ycenters

    # Mask zeros and plot
    mask = np.ma.masked_where(h.T == 0, h.T)
    pcm = ax_main.pcolormesh(xi, yi, mask, cmap=cmap)
    cb = fig.colorbar(pcm, ax=ax_main, cax=ax_cb, location='left')
    cb.set_label('Number of Matches', fontsize=12)

    # Marginal distributions
    counts_x = np.sum(h.T, axis=0)
    counts_y = np.sum(h.T, axis=1)
    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]

    pdf_x = counts_x / (counts_x.sum() * dx)
    pdf_y = counts_y / (counts_y.sum() * dy)
    cdf_x = np.cumsum(counts_x) / counts_x.sum()
    cdf_y = np.cumsum(counts_y) / counts_y.sum()

    # Plot marginals and CDFs
    ax_xhist.plot(xi, pdf_x, color='blue', lw=1)
    ax_xhist.twinx().plot(xi, cdf_x, color='red', linestyle='--', lw=1)
    ax_xhist.set_ylabel('PDF', rotation=-90, color='blue', labelpad=18)
    ax_xhist.yaxis.label.set_color('blue')

    ax_yhist.plot(pdf_y, yi, color='blue', lw=1)
    ax_yhist.twiny().plot(cdf_y, yi, color='red', linestyle='--', lw=1)
    ax_yhist.set_xlabel('PDF', color='blue')
    ax_yhist.xaxis.label.set_color('blue')

    # Labels and grid
    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    ax_xhist.set_xlim(ax_main.get_xlim())
    ax_yhist.set_ylim(ax_main.get_ylim())
    ax_xhist.grid(alpha=0.4, linestyle='--')
    ax_yhist.grid(alpha=0.4, linestyle='--')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Deprecated functions ---
    # --- These functions are deprecated and should be removed in future versions. ---
















def flux_reproduction_plot_deprecated(target_cat, sr_cat, target_cl, matching_distance, save_path):
    """
    Generates a flux reproduction plot comparing target and super-resolved catalogs.

    Parameters:
        target_cat (pd.DataFrame): Target catalog containing flux values.
        sr_cat (pd.DataFrame): Super-resolved catalog containing flux values.
        matching_distance (float): Maximum matching distance for catalog cross-matching (in degrees).
        save_path (str): Path to save the generated plot.
    """
    num_bins = 7

    # Cross-match the catalogs and filter by target flux threshold
    matched_cat = cross_match_catalogs(sr_cat, target_cat, f"S{target_cl}", f"S{target_cl[:-2]}")
    filtered_matched_cat = matched_cat[matched_cat[f"S{target_cl[:-2]}"].values <= 0.1]  # Filter for S500 <= 100 mJy

    # Calculate flux statistics
    flux_ratios = filtered_matched_cat[f"S{target_cl}"].values / filtered_matched_cat[f"S{target_cl[:-2]}"].values - 1
    flux_statistics = calculate_flux_statistics(
        filtered_matched_cat[f"S{target_cl[:-2]}"].values,
        flux_ratios,
        num_bins
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.fill_between(
        flux_statistics[:, 0] * 1000, flux_statistics[:, 2], flux_statistics[:, 3],
        alpha=0.8, color="#8b0023"
    )
    ax.plot(
        flux_statistics[:, 0] * 1000, flux_statistics[:, 1],
        linestyle='-', marker='o', color="#8b0023",
        markersize=4, markeredgecolor="#0e1111"
    )

    # Set labels, scales, and limits
    ax.set_xlabel(r"$S_{Target}$ (mJy)")
    ax.set_ylabel(r"$(S_{SR} - S_{Target}) / S_{Target}$")
    ax.set_xscale('log')
    ax.set_yticks(np.arange(-1, 1.2, 0.2))
    ax.set_xlim([np.min(flux_statistics[:, 0] * 1000), 100])
    ax.set_ylim([-0.8, 0.8])

    # Add a horizontal line at y=0 and text annotation
    ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='#0e1111', ls='-', lw=3, alpha=0.7)
    ax.text(
        0.95, 0.2,
        f"Model: Unet-Resnet34-Tr\nMatching radius: {matching_distance * 3600:.1f}''\nMatches: {filtered_matched_cat.shape[0]}",
        ha='right', va='top', transform=ax.transAxes, fontsize=8
    )

    # Save and close the plot
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def pos_flux_cornerplot_deprecated(target_cat, sr_cat, target_cl, save_path, **plot_args):
    '''Plots the Positional Offset against the source flux reproduction.
        The plot displays any systematics, and zooms in at the region around the perfect performance.
        This region is specified by the plot arguments.
    '''
    # Define the colormap for the 2D distribution plot, NaNs (0 counts) are set to white color.
    cmap = plt.get_cmap("plasma")
    cmap.set_bad(cmap(0.))

    # Create the 2D blind distribution plot
    fig = plt.figure(figsize=(7.5, 7.5))
    grid = plt.GridSpec(4, 5, hspace=0.3, wspace=0.3, width_ratios=[1,5,5,5,5])
    ax_main = fig.add_subplot(grid[1:4, 1:4])
    ax_xhist = fig.add_subplot(grid[0, 1:4], sharex=ax_main)
    ax_yhist = fig.add_subplot(grid[1:4, 4], sharey=ax_main)
    ax_colorbar = fig.add_subplot(grid[1:4, 0])
   
    binsx, binsy = 50, 50
    xbins = np.linspace(-0.5, 0.5, binsx + 1)
    ybins = np.linspace(0, 4, binsy + 1)


    matched_cat = construct_matched_catalog(target_cat, sr_cat, target_cl, max_distance=4)
    # matched_cat = cross_match_catalogs(sr_cat, target_cat, "S500SR", "S500")

    filtered_matched_cat = matched_cat.copy()

    H_blind, xedges_blind, yedges_blind = np.histogram2d((filtered_matched_cat[f"S{target_cl}"] - filtered_matched_cat[f'S{target_cl[:-2]}'])/filtered_matched_cat[f'S{target_cl[:-2]}'], filtered_matched_cat["distance"], bins=(xbins, ybins))

    x_centers = (xbins[:-1] + xbins[1:]) / 2
    y_centers = (ybins[:-1] + ybins[1:]) / 2

    H = np.round(H_blind, 0).astype(np.int32)

    mask = np.ma.masked_where(H.T == 0, H.T)
    pcol = ax_main.pcolormesh(xedges_blind, yedges_blind, mask, cmap=cmap, norm=LogNorm())#, shading='gouraud')
    cb = fig.colorbar(pcol, ax=ax_main, cax=ax_colorbar, location='left')
    pos_cb = ax_colorbar.get_position()
    pos_main = ax_main.get_position()
    ax_colorbar.set_position([pos_cb.x0 - 0.05, pos_main.y0, 0.03, pos_main.y1 - pos_main.y0])
    cb.set_label('Number of Matches', fontsize=12)

    # Set the axis limits for the 2D distribution plot
    ax_main.set_xlim(-0.5, 0.5)
    ax_main.set_ylim(0, 4)

    # Create the x and y marginal plots
    Xhist = np.sum(H.T, axis=0)/np.sum(H.T)
    Yhist = np.sum(H.T, axis=1)/np.sum(H.T)

    cumulative_prob_reproduction_ratio = np.cumsum(Xhist)
    cumulative_prob_offset = np.cumsum(Yhist)

    ax_xhist_y2 = ax_xhist.twinx();
    ax_yhist_y2 = ax_yhist.twiny();

    ax_xhist_y2.plot(x_centers, cumulative_prob_reproduction_ratio, color='red', linestyle='--', lw=1)
    ax_xhist.plot(x_centers, Xhist, color='blue', linestyle='-', lw=1)
    ax_yhist_y2.plot(cumulative_prob_offset, y_centers, color='red', linestyle='--', lw=1)
    ax_yhist.plot(Yhist, y_centers, color='blue', linestyle='-', lw=1)

    # Set ticks
    ax_xhist.tick_params(axis='y', which='both', labelsize=10, colors='blue')
    ax_xhist.tick_params(axis='x', which='both', labelsize=10)
    ax_xhist_y2.tick_params(axis='y', which='both', labelsize=10, colors='red')
    ax_xhist_y2.yaxis.label.set_color('blue')
    ax_yhist_y2.xaxis.label.set_color('red')
    ax_yhist.yaxis.label.set_color('blue')
    ax_yhist.yaxis.label.set_color('red')

    ax_yhist.tick_params(axis='y', which='both', labelsize=10)
    ax_yhist.tick_params(axis='x', which='both', labelsize=10, colors='blue', rotation=-90)
    ax_yhist_y2.tick_params(axis='x', which='both', labelsize=10, colors='red', rotation=-90)

    ax_main.tick_params(axis='y', which='both', labelsize=10)
    ax_main.tick_params(axis='x', which='both', labelsize=10)
    ax_main.tick_params(axis='y', which='both', labelsize=10)

    # Set the axis labels
    ax_main.set_xlabel(r"$(S_{SR} - S_{Target})/S_{Target}$", fontsize=12)
    ax_main.set_ylabel('Offset (\'\')', fontsize=12)
    ax_xhist.set_ylabel('PDF', fontsize=12, color='blue', rotation=-90, labelpad=18)
    ax_yhist.set_xlabel('PDF', fontsize=12, color='blue')
    ax_xhist_y2.set_ylabel('CDF', fontsize=12, color='red', rotation=-90, labelpad=12)
    ax_yhist_y2.set_xlabel('CDF', fontsize=12, color='red')

    ax_yhist.grid(which='both', alpha=0.4, color='lightgrey', linestyle='--')
    ax_xhist.grid(which='both', alpha=0.4, color='lightgrey', linestyle='--')

    xticks_xhist = np.round(np.arange(0., np.max(Xhist), np.max(Xhist)/4), 2)
    xticks_yhist = np.round(np.arange(0., np.max(Yhist), np.max(Yhist)/4), 2)
    ax_xhist.set_yticks(xticks_xhist)
    ax_yhist.set_xticks(xticks_yhist)
    ax_xhist_y2.set_yticks(np.arange(0, 1.2, 0.2))
    ax_yhist_y2.set_xticks(np.arange(0, 1.2, 0.2))

    # Display the plot
    fig.savefig(save_path, dpi=400, bbox_inches = "tight")
    plt.close(fig)

def CompletenessReliabilityPlot(target_cat, sr_cat, target_cl, max_distance, save_path):
    max_bin_value = 80 # mJy
    num_bins = 20
    log_min = np.log10(1)
    log_max = np.log10(max_bin_value)
    log_spacing = (log_max - log_min) / num_bins
    half_width = 0.5 * log_spacing
    flux_bin_edges = np.logspace(log_min - half_width, log_max + half_width, num_bins + 1)/1000

    ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
    flux_bins = list(zip(flux_bin_edges[:-1], flux_bin_edges[1:]))
    zero_list = np.zeros(len(flux_bins))

    # Construct the confusion matrix or df
    confusion_df = {'Flux bins': flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'C': zero_list, 'R': zero_list}
    confusion_df = pd.DataFrame(confusion_df)

    # Completeness
    # matched_cat = construct_matched_catalog(target_cat, sr_cat, max_distance=max_distance)
    matched_cat = cross_match_catalogs(sr_cat, target_cat, f"S{target_cl}", f"S{target_cl[:-2]}")

    filtered_matched_cat = matched_cat[matched_cat[f"S{target_cl[:-2]}"].values <= 100/1000]
    filtered_target_cat = target_cat[target_cat[f"S{target_cl[:-2]}"].values <= 100/1000]

    bin_inds_matched = np.digitize(filtered_matched_cat[f"S{target_cl[:-2]}"].values, flux_bin_edges, right=False)
    bin_inds_target = np.digitize(filtered_target_cat[f"S{target_cl[:-2]}"].values, flux_bin_edges, right=False)
    for i in range(1, len(flux_bins)):
        confusion_df.loc[i, "TPc"] = np.sum((bin_inds_matched == i))
        confusion_df.loc[i, "FNc"] = np.sum((bin_inds_target == i)) - confusion_df.loc[i, "TPc"]

    # Reliability
    matched_cat = construct_matched_catalog(sr_cat, target_cat, target_cl, max_distance=max_distance)
    filtered_matched_cat = matched_cat[matched_cat[f"S{target_cl}"].values <= 100/1000]

    filtered_sr_cat = sr_cat[sr_cat[f"S{target_cl}"].values <= 100/1000]

    bin_inds_matched = np.digitize(filtered_matched_cat[f"S{target_cl}"].values, flux_bin_edges, right=False)
    bin_inds_sr = np.digitize(filtered_sr_cat[f"S{target_cl}"].values, flux_bin_edges, right=False)
    for i in range(1, len(flux_bins)):
        confusion_df.loc[i, "TPr"] = np.sum((bin_inds_matched == i))
        confusion_df.loc[i, "FPr"] = np.sum((bin_inds_sr == i)) - confusion_df.loc[i, "TPr"]

    # Compute cumulative True Positives, False Negatives etc
    confusion_df["TPr_cum"] = confusion_df["TPr"][::-1].cumsum()[::-1]
    confusion_df["FPr_cum"] = confusion_df["FPr"][::-1].cumsum()[::-1]
    confusion_df["TPc_cum"] = confusion_df["TPc"][::-1].cumsum()[::-1]
    confusion_df["FNc_cum"] = confusion_df["FNc"][::-1].cumsum()[::-1]

    # Create the cumulative Completeness and Reliability columns
    confusion_df["C_cum"] = 0.
    confusion_df["R_cum"] = 0.

    # Calculate completeness and reliability
    ## If needed resolve zero-occurences
    for i in range(len(confusion_df['Flux bins'])):
        if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
            confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])

        if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
            confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
        
        if confusion_df.loc[i, 'TPc_cum'] + confusion_df.loc[i, 'FNc_cum'] != 0:
            confusion_df.loc[i, 'C_cum'] = confusion_df.loc[i, 'TPc_cum']/(confusion_df.loc[i, 'TPc_cum'] + confusion_df.loc[i, 'FNc_cum'])

        if confusion_df.loc[i, 'TPr_cum'] + confusion_df.loc[i, 'FPr_cum'] != 0:
            confusion_df.loc[i, 'R_cum'] = confusion_df.loc[i, 'TPr_cum']/(confusion_df.loc[i, 'TPr_cum'] + confusion_df.loc[i, 'FPr_cum'])


    ##### PLOT #####
    threshold = 2. # mJy
    flux = np.array([((bin[0] + bin[1])/2) for bin in flux_bins]) * 1000

    # Reliability Plot
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 4))

    # Plot the Reliability and Completeness lines with error bars
    NsamplesR = confusion_df['TPr'] + confusion_df['FPr']
    NsamplesC = confusion_df['TPc'] + confusion_df['FNc']

    NsamplesR_cum = confusion_df['TPr_cum'] + confusion_df['FPr_cum']
    NsamplesC_cum = confusion_df['TPc_cum'] + confusion_df['FNc_cum']

    axs[0].errorbar(flux, confusion_df['R'], yerr=np.sqrt(confusion_df['R'] * (1- confusion_df['R']) / NsamplesR), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$R(S_{SR, bin})$")

    axs[1].errorbar(flux, confusion_df['C'], yerr=np.sqrt(confusion_df['C'] * (1- confusion_df['C']) / NsamplesC), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, label=r"$C(S_{Target, bin})$")
    axs[0].errorbar(flux, confusion_df['R_cum'], yerr=np.sqrt(confusion_df['R_cum'] * (1- confusion_df['R_cum']) / NsamplesR_cum), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, color='green', label=r"$R(\geq S_{SR, bin})$")

    axs[1].errorbar(flux, confusion_df['C_cum'], yerr=np.sqrt(confusion_df['C_cum'] * (1- confusion_df['C_cum']) / NsamplesC_cum), marker='o', markersize=4,
                    capsize=0, capthick=0, elinewidth=1, linewidth=1, fillstyle='none', linestyle='dashed', alpha=0.6, color='green', label=r"$C(\geq S_{Target, bin})$")

    # Set the x-axis to log scale and add a minor grid
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    axs[0].xaxis.set_tick_params(labelbottom=True)
    axs[0].yaxis.set_tick_params(labelleft=True)
    axs[0].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')
    axs[1].xaxis.set_tick_params(labelbottom=True)
    axs[1].yaxis.set_tick_params(labelleft=True)
    axs[1].grid(which='both', alpha=0.3, color='lightgrey', linestyle='--')

    # Set the axis labels and title for the main plot
    axs[0].set_xlabel('Super-resolved Source Flux (mJy)', fontsize=10)
    axs[1].set_xlabel('Target Source Flux (mJy)', fontsize=10)

    axs[0].set_ylabel('Reliability', fontsize=10)
    axs[1].set_ylabel('Completeness', fontsize=10)

    axs[0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1].set_yticks(np.arange(0, 1.1, 0.1))
    axs[0].tick_params(direction='in', axis='x', which='both', labelsize=10)
    axs[1].tick_params(direction='in', axis='x', which='both', labelsize=10)
    axs[0].tick_params(axis='y', which='both', labelsize=10)
    axs[1].tick_params(axis='y', which='both', labelsize=10)

    axs[0].set_ylim([0., 1.02])
    axs[1].set_ylim([0., 1.02])
    axs[0].set_xlim([np.min(threshold)-0.1, np.max(flux)+0.05])
    axs[1].set_xlim([np.min(threshold)-0.1, np.max(flux)+0.05])
    axs[0].set_xticks(np.array([threshold, 10, 100]))
    axs[1].set_xticks(np.array([threshold, 10, 100]))

    axs[0].legend(loc='lower right', frameon=False)
    axs[1].legend(loc='lower right', frameon=False)


    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=0.3)

    # Save the plot
    fig.savefig(save_path, dpi=350, edgecolor='white', facecolor='white', bbox_inches='tight')
    plt.close(fig)


def get_highlighted_sources(catalog, wcs_arr, imageID, target_cl, boxsize=30, imgDIM=(256, 256)):
    flux_col_name = "S450" if "S450" in catalog.columns else (f"S{target_cl[:-2]}" if f"S{target_cl[:-2]}" in catalog.columns else f"S{target_cl}")

    # Obtain the catalog containing sources for the given image
    catalog = catalog[catalog["ImageID"] == imageID].copy()

    # Obtain coordinates of the sources in pixels
    catalog[["xpix", "ypix"]]= wcs_arr[imageID].wcs_world2pix(catalog[["ra", "dec"]].values, 0)

    # Get all sources within the applicable region for highlighting
    catalog = catalog[(boxsize*1.5 < catalog["xpix"]) & (catalog["xpix"] < (imgDIM[0] - boxsize*1.5)) & 
                    (boxsize*1.5 < catalog["ypix"]) & (catalog["ypix"] < (imgDIM[1] - boxsize*1.5))]

    # Return None, None if catalog has fewer than 2 sources
    if catalog.shape[0] == 0:
        return None, None

    # Find the index of the brightest source
    brightest_idx = np.argmax(catalog[flux_col_name].values)
    brightest_source = catalog[["xpix", "ypix"]].iloc[brightest_idx].values
    
    # Find another source that is atleast a distance 2*boxsize from the brightest source
    distances = np.linalg.norm(catalog[["xpix", "ypix"]].values - brightest_source, axis=1)
    
    closest_far_sources = distances >= 2*boxsize

    if np.sum(distances >= 2*boxsize) == 0:
        return None, None
    
    closest_far_source = catalog[["xpix", "ypix"]].iloc[closest_far_sources].values[0]
    
    return brightest_source, closest_far_source


def PlotSuperResolvedImage(cat, wcs_arr, target_cl, Ysample, imageID, save_path, boxsize=30, selected_sources=None, target_sources_in_region=None):
    '''Plot the Target or SuperResolved Image (Ysample, Shape: (H,W,1)) in the same style as the Input images.
       The plot will automatically determine two sources to be highlighted.
       When using this function for a SuperResolved Image, use these two sources instead by passing it
       to selected_sources. This ensures that the highlighted sources are synchronized.
       boxsize parameter (in pixels) specifies Height and Width of the highlighted regions.'''
    
    target_bool = True if selected_sources is None else False

    # Check for selected_sources --> Indicates Target (None) or Prediction (list of 2 coordinate arrays)
    if selected_sources is None:
        # Pick two sources from the catalog, which will be highlighted
        brightest_source, closest_far_source = get_highlighted_sources(cat, wcs_arr, imageID, target_cl, boxsize=boxsize)

        if brightest_source is None or closest_far_source is None:
            return None, None

        selected_sources = [brightest_source, closest_far_source]

    img_label = r'450 $\mu m$ (Target)' if "S450" in cat.columns else (rf'S{{target_cl[:-2]}} $\mu m$ (Target)' if f'S{target_cl[:-2]}' in cat.columns else rf'S{{target_cl[:-2]}} $\mu m$ (Super-resolved)')

    ##### PLOT #####
    fig = plt.figure(figsize=(6, 9))
    fig.patch.set_facecolor('black')

    # Create grid layout, fixed do not change
    gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.05, hspace=0.05)
    axs = [
        fig.add_subplot(gs[:2, :]),
        fig.add_subplot(gs[-1, -1]),
        fig.add_subplot(gs[-1, -2])
    ]    
    colors = ['white', '#FFD43E', '#45E0A5']

    # Main plot, change vmax if necessary
    axs[0].imshow(Ysample, origin='lower', cmap='afmhot', aspect='equal', vmin=0, vmax=15e-3)

    # Create colorbar
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(axs[0].get_images()[0], cax=cax, orientation='vertical')
    cb.outline.set_edgecolor('white')
    cb.ax.yaxis.set_tick_params(color='white')
    cb.set_label('Jy/beam', color='white')
    plt.setp(plt.getp(cax.axes, 'yticklabels'), color='white')

    # Similarly to the plot function for input images, we assume 256x256 images.
    axs[0].text(5, 256-5, img_label, color='white', ha='left', va='top', weight='bold', fontsize=12)

    for i in range(3):
        for loc in ['left', 'right', 'bottom', 'top']:
            axs[i].spines[loc].set_color(colors[i])
            axs[i].spines[loc].set_linewidth(2)
            axs[i].set(yticklabels=[], xticklabels=[])  # remove the tick labels
            axs[i].tick_params(left=False, bottom=False)  # remove the ticks  

    # Draw highlighted areas
    cat_img = cat[cat["ImageID"] == imageID]
    if target_bool:
        target_sources_in_region = [[], []]

    for i in range(0, 2):
        # Draw rectangle on main plot, plot highlighted region
        rect = plt.Rectangle((selected_sources[i][0] - boxsize/2, selected_sources[i][1] - boxsize/2), boxsize, boxsize, fill=False, edgecolor=colors[i+1])
        axs[0].add_patch(rect)
        axs[i + 1].imshow(Ysample[int(np.round(rect.xy[1])):int(np.round(rect.xy[1]+boxsize)), int(np.round(rect.xy[0])):int(np.round(rect.xy[0]+boxsize))], origin='lower', cmap='afmhot', aspect='equal', vmin=0, vmax=15e-3)
        # draw markers for target and super-resolved sources
        # both shown in SR plot, only target shown in target plot.
        # Scale by size?    
        offset = 2# small offset because a marker has a size that can extend beyond the border which gives bad plots.
        cat_region = cat_img[
            (cat_img["xpix"] - (rect.xy[0] + offset) > 0) & 
            (cat_img["xpix"] - (rect.xy[0] - offset) < boxsize) & 
            (cat_img["ypix"] - (rect.xy[1] + offset) > 0) & 
            (cat_img["ypix"] - (rect.xy[1] - offset) < boxsize)]
        
        if target_bool:
            target_sources_in_region[i] = cat_region[["xpix", "ypix"]].values

        if target_bool and cat_region.values.shape[0] != 0:
            axs[i + 1].scatter(cat_region["xpix"] - rect.xy[0], cat_region["ypix"] - rect.xy[1], marker='+', s=200, linewidth=2, color="#1F51FF")
        elif not target_bool and cat_region.values.shape[0] != 0:
            axs[i + 1].scatter(cat_region["xpix"] - rect.xy[0], cat_region["ypix"] - rect.xy[1], facecolor='none', marker='o', s=200, linewidth=2, edgecolors="#00FF00")
            
        if not target_bool:
            axs[i + 1].scatter(target_sources_in_region[i][:, 0] - rect.xy[0], target_sources_in_region[i][:, 1] - rect.xy[1], marker='+', s=200, linewidth=2, color="#1F51FF")
    
    # Save figure
    plt.savefig(save_path, bbox_inches = "tight")
    plt.close(fig)

    return selected_sources, target_sources_in_region

def PlotInputImages(Xsample, boxsize, selected_sources, save_path):
    '''Plot the input images used in the model. Xsample is a Tensor/ndarray of shape (H,W,C).
    Height, Width, Channels. C = 4 for inclusion of 24 microns, else 3 (only SPIRE).
    This function needs to be modified if used with different H,W dimensions.'''
    Nimages = Xsample.shape[-1]

    # Input based grid allowing also for only SPIRE input
    if Nimages == 4:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
        wavelengths = [r'24 $\mu m$', r'250 $\mu m$', r'350 $\mu m$', r'500 $\mu m$ (native)']
    else:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5), constrained_layout=True)
        wavelengths = [r'250 $\mu m$', r'350 $\mu m$', r'500 $\mu m$ (native)']

    # Nice black background
    fig.patch.set_facecolor('black')

    ##### PLOT #####
    cmap = 'afmhot'
    colors = ['#FFD43E', '#45E0A5']
    # Draw highlighted areas
    for ax, i, wavelength in zip(axs.flat, range(Nimages) , wavelengths):
        if ax == axs.flatten()[0] and Nimages == 4:
            im = ax.imshow(Xsample[:, :, i], cmap=cmap, vmin=0, vmax=5e-4, origin='lower')
        else:
            im = ax.imshow(Xsample[:, :, i], cmap=cmap, vmin=0, vmax=50e-3, origin='lower')
        
        # Add wavelength text in the upper left corner
        ax.text(5, 256-5, wavelength, color='white', ha='left', va='top', weight='bold', fontsize=16)
        
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append axes to the right of ax, with 5% width of ax
        if Nimages == 4:
            if ax == axs.flatten()[0] or ax == axs.flatten()[2]:
                cax = divider.append_axes("left", size="5%", pad=0.05)
            else:
                cax = divider.append_axes("right", size="5%", pad=0.05)
        else:
            cax = divider.append_axes("right", size="5%", pad=0.05)

        # Create colorbar in the appended axes
        # Tick locations can be set with the kwarg `ticks`
        # and the format of the ticklabels with kwarg `format`
        if Nimages == 4:
            if ax == axs.flatten()[0] or ax == axs.flatten()[2]:
                cb = fig.colorbar(im, cax=cax, orientation='vertical', location='left')
                cax.yaxis.set_ticks_position('left')
                cb.set_label('Jy/beam', color='white')
            else:
                cb = fig.colorbar(im, cax=cax, orientation='vertical', location='right')
                cb.set_label('Jy/beam', color='white')
        else:
            cb = fig.colorbar(im, cax=cax, orientation='vertical', location='right')
            cb.set_label('Jy/beam', color='white')
        # set colorbar edgecolor 
        cb.outline.set_edgecolor('white')

        # Set colorbar label color to white
        cax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cax.axes, 'yticklabels'), color='white')

        # Draw highlighted region on the input maps
        for k in range(0, 2):
            rect = plt.Rectangle((selected_sources[k][0] - boxsize/2, selected_sources[k][1] - boxsize/2), boxsize, boxsize, fill=False, edgecolor=colors[k], lw=2)
            ax.add_patch(rect)


    # Adding white frames around each image
    for i in range(4):
        for loc in ['left', 'right', 'bottom', 'top']:
            axs.flatten()[i].spines[loc].set_color("white")
            axs.flatten()[i].spines[loc].set_linewidth(2)
            axs.flatten()[i].set(yticklabels=[], xticklabels=[])  # remove the tick labels
            axs.flatten()[i].tick_params(left=False, bottom=False)  # remove the ticks  

    plt.savefig(save_path, bbox_inches = "tight")
    plt.close(fig)

def scuba_recovery_plot(matched_catalog, save_path=None):
    # Set up the subplots with shared y-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    # Define plot limits and a true line for 1:1 recovery
    xmax = 30  # mJy
    true_line = np.linspace(0, xmax + 5, 100)
    # Plot the first dataset
    ax1.errorbar(
        matched_catalog["S450"], 
        matched_catalog["S500SR"], 
        xerr=matched_catalog["S450_total_err"], 
        markersize=6, 
        fmt='o', 
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        color='blue', 
        label="Matched SCUBA-2 Sources", 
        alpha=0.8
    )
    ax1.plot(
        true_line, 
        true_line,
        'r--', 
        label="1:1 Recovery", 
        linewidth=2
    )

    ax1.set_xlabel(r"SCUBA-2 $450 \mu m$ Source Flux $S_{in}$ [mJy]", fontsize=12)
    ax1.set_ylabel(r"Super-resolved $500 \mu m$ Source Flux $S_{SR}$ [mJy]", fontsize=12)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_xlim([1, xmax + 5])
    ax1.set_ylim([1, xmax + 5])
    ax1.legend(fontsize=12, frameon=False, loc='upper left')
    
    # Plot the second dataset
    ax2.errorbar(
        matched_catalog["S500corr"], 
        matched_catalog["S500SR"], 
        xerr=matched_catalog["S500corr_total_err"], 
        markersize=6, 
        fmt='o', 
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        color='blue', 
        label="Matched SCUBA-2 Sources", 
        alpha=0.8
    )

    ax2.plot(
        true_line, 
        true_line,
        'r--', 
        label="1:1 Recovery", 
        linewidth=2
    )

    ax2.set_xlabel(r"SCUBA-2 Converted $500 \mu m$ Source Flux $S_{in}$ [mJy]", fontsize=12)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_xlim([1, xmax + 5])
    ax2.legend(fontsize=12, frameon=False, loc='upper left')
    
    # Adjust spacing between plots to remove horizontal space
    plt.subplots_adjust(wspace=0)
    ax1.grid(True)
    ax2.grid(True)
    
    # Save the figure
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close(fig)