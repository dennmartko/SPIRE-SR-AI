import os
import gc
import argparse
import pandas as pd
from tqdm import tqdm
from astropy.table import Table, vstack
from astropy.io import fits
import numpy as np

SELECTED_COLS = ['ra', 'dec', 'S24', 'S250', 'S350', 'S500']
max_i, max_j = 6, 8  # max_j is used only for j=8 section in SIDES tiles

def load_catalog(tile_i, tile_j):
    filename = f'/mnt/g/data/PhD Projects/SR/pysides_from_uchuu catalogs/pySIDES_from_uchuu_tile_{tile_i}_{tile_j}.fits'
    return Table.read(filename)

def process_tiles(output_dir):
    """Process SIDES dataset catalogs from tile pairs and save at most 30 catalogs."""
    os.makedirs(output_dir, exist_ok=True)
    sides_count = 0

    # Process j pairs: (0,1), (2,3), (4,5), (6,7)
    for i in tqdm(range(max_i + 1), desc='Processing SIDES j pairs'):
        if sides_count >= 30:
            break
        for j in range(0, 8, 2):
            if sides_count >= 30:
                break
            tables = []
            for jj in [j, j + 1]:
                try:
                    cat = load_catalog(i, jj)
                    # Rename SIDES columns to match new naming conventions
                    cat.rename_column('S24', 'SMIPS24')
                    cat.rename_column('S250', 'SSPIRE250')
                    cat.rename_column('S350', 'SSPIRE350')
                    cat.rename_column('S500', 'SSPIRE500')
                    tables.append(cat[['ra', 'dec', 'SMIPS24', 'SSPIRE250', 'SSPIRE350', 'SSPIRE500']])
                except Exception as e:
                    print(f"Skipping tile ({i}, {jj}): {e}")
            if len(tables) == 2:
                merged = vstack(tables)
                sides_count += 1
                fname = os.path.join(output_dir, f'SIDES_{sides_count}_cat.fits')
                merged.write(fname, overwrite=True)
                del merged, tables
                gc.collect()

    # Process j=8 tiles: merge (i,8) with (i+1,8) for even i; if unpaired, do not write that tile.
    i = 0
    while i <= max_i and sides_count < 30:
        tables = []
        try:
            cat = load_catalog(i, max_j)
            cat.rename_column('S24', 'SMIPS24')
            cat.rename_column('S250', 'SSPIRE250')
            cat.rename_column('S350', 'SSPIRE350')
            cat.rename_column('S500', 'SSPIRE500')
            tables.append(cat[['ra', 'dec', 'SMIPS24', 'SSPIRE250', 'SSPIRE350', 'SSPIRE500']])
        except Exception as e:
            print(f"Skipping tile ({i}, {max_j}): {e}")
        if i + 1 <= max_i:
            try:
                cat = load_catalog(i + 1, max_j)
                cat.rename_column('S24', 'SMIPS24')
                cat.rename_column('S250', 'SSPIRE250')
                cat.rename_column('S350', 'SSPIRE350')
                cat.rename_column('S500', 'SSPIRE500')
                tables.append(cat[['ra', 'dec', 'SMIPS24', 'SSPIRE250', 'SSPIRE350', 'SSPIRE500']])
            except Exception as e:
                print(f"Skipping tile ({i+1}, {max_j}): {e}")
        if len(tables) == 2:
            merged = vstack(tables)
            sides_count += 1
            fname = os.path.join(output_dir, f'SIDES_{sides_count}_cat.fits')
            merged.write(fname, overwrite=True)
            del merged, tables
            gc.collect()
        i += 2


def load_shark_catalog():
    """Load and merge the SHARK catalogs from chunks."""
    file_path_1 = '/mnt/g/data/PhD Projects/SR/Shark-deep-opticalLightcone-AtLAST-FIR.txt'
    file_path_2 = '/mnt/g/data/PhD Projects/SR/Shark-deep-opticalLightcone-AtLAST.txt'

    flux_iter = pd.read_csv(
        file_path_1,
        sep=r'\s+|\t',
        header=None,
        engine='python',
        skiprows=8,
        names=["SMIPS24", "SSPIRE250", "SSPIRE350", "SSPIRE500"],
        usecols=[0, 4, 5, 7],
        chunksize=int(5e6),
        dtype={"SMIPS24": np.float64, "SSPIRE250": np.float64,
               "SSPIRE350": np.float64, "SSPIRE500": np.float64},
    )
    pos_iter = pd.read_csv(
        file_path_2,
        sep=r'\s+|\t',
        header=None,
        engine='python',
        skiprows=12,
        names=["dec", "ra"],
        usecols=[0, 1],
        chunksize=int(5e6),
        dtype={"dec": np.float64, "ra": np.float64}
    )

    merged_chunks = []
    chunk_counter = 0
    for pos_chunk, flux_chunk in zip(pos_iter, flux_iter):
        chunk_counter += 1
        print(f"Processing SHARK chunk {chunk_counter}")
        flux_chunk = flux_chunk / 1000  # Convert mJy to Jy
        pos_chunk = pos_chunk.reset_index(drop=True)
        flux_chunk = flux_chunk.reset_index(drop=True)
        merged_chunks.append(pd.concat([pos_chunk, flux_chunk], axis=1))

    print(f"Loaded {chunk_counter} chunks. Concatenating SHARK catalog...")
    merged_cat = pd.concat(merged_chunks, ignore_index=True)
    print("Finished loading SHARK catalog.")
    return merged_cat

def process_shark_catalog(output_dir):
    """Cut the SHARK catalog into subregions and write at most 30 catalogs."""
    print("Starting SHARK catalog processing...")
    shark_cat = load_shark_catalog()

    # Define region bounds and cut size (delta)
    ra_start, ra_end = 211.5, 223.5
    dec_start, dec_end = -4.5, 4.5
    delta_ra, delta_dec = 2, 1

    shark_count = 0
    for dec_min in np.arange(dec_start, dec_end, delta_dec):
        dec_max = dec_min + delta_dec
        for ra_min in np.arange(ra_start, ra_end, delta_ra):
            if shark_count >= 30:
                break
            ra_max = ra_min + delta_ra
            print(f"Processing SHARK subregion {shark_count+1}: RA {ra_min}-{ra_max}, Dec {dec_min}-{dec_max}")
            subset = shark_cat[(shark_cat['ra'] >= ra_min) & (shark_cat['ra'] < ra_max) &
                               (shark_cat['dec'] >= dec_min) & (shark_cat['dec'] < dec_max)]
            if not subset.empty:
                table = Table.from_pandas(subset)
                shark_count += 1
                fname = os.path.join(output_dir, f'SHARK_{shark_count}_cat.fits')
                table.write(fname, overwrite=True)
                print(f"Saved SHARK subregion {shark_count} to {fname}")
        if shark_count >= 30:
            break

    print("Finished processing SHARK catalog.")

if __name__ == '__main__':
    output_dir = '/mnt/g/data/PhD Projects/SR/sim_catalogs'  # Set your desired output directory here
    # To process SIDES catalogs, uncomment the next line:
    process_tiles(output_dir)
    gc.collect()
    #process_shark_catalog(output_dir)