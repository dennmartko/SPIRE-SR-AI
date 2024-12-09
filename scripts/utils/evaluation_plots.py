import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

from scripts.utils.metrics import (
    calculate_flux_statistics, 
    cross_match_catalogs, 
    construct_matched_catalog)

def flux_reproduction_plot(target_cat, sr_cat, target_cl, matching_distance, save_path):
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

def pos_flux_cornerplot(target_cat, sr_cat, target_cl, save_path, **plot_args):
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
    xbins = np.linspace(plot_args['ReproductionRatio_min'], plot_args['ReproductionRatio_max'], binsx + 1)
    ybins = np.linspace(0, plot_args['max_distance'], binsy + 1)


    matched_cat = construct_matched_catalog(target_cat, sr_cat, target_cl, max_distance=plot_args["max_distance"])
    # matched_cat = cross_match_catalogs(sr_cat, target_cat, "S500SR", "S500")

    filtered_matched_cat = matched_cat[matched_cat[f"S{target_cl[:-2]}"].values <= 100/1000]

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
    ax_main.set_xlim(plot_args['ReproductionRatio_min'], plot_args['ReproductionRatio_max'])
    ax_main.set_ylim(0, plot_args['max_distance'])

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