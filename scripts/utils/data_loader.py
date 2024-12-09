import tensorflow as tf
import numpy as np
import os
from astropy.io import fits
from glob import glob
from astropy.wcs import WCS

def parse_fits_numpy(file_path):
    """Reads FITS file using numpy and returns the data from HDU index 0."""
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float32)
    return data

def parse_fits(file_path):
    """Wrapper to parse FITS using TensorFlow's py_function."""
    data = tf.numpy_function(parse_fits_numpy, [file_path], tf.float32)
    return data

def load_data(input_paths, target_paths):
    """Loads synchronized input images and target image."""
    # Convert tf.Tensor (input_paths) to a Python list
    input_paths = tf.unstack(input_paths)  # Breaks input_paths into a list of tensors
    
    # Use tf.numpy_function to process each path
    inputs = [parse_fits(path) for path in input_paths]
    inputs = tf.stack(inputs, axis=-1)  # Stack input images into one tensor
    
    # Process the output path
    target_paths = tf.unstack(target_paths)  # Breaks input_paths into a list of tensors

    # Use tf.numpy_function to process each path
    targets = [parse_fits(path) for path in target_paths]
    targets = tf.stack(targets, axis=-1)  # Stack input images into one tensor

    return inputs, targets


def match_input_output(directory, input_classes, target_classes, include_mask):
    """
    Matches input and output images based on their IDs.
    Returns paired input and output paths.
    """
    input_files = {cls: sorted(glob(os.path.join(directory, cls, '*.fits'))) for cls in input_classes}
    output_files = {cls: sorted(glob(os.path.join(directory, cls, '*.fits'))) for cls in target_classes}
    mask_files = {cls: sorted(glob(os.path.join(directory, f'{cls}_mask', '*.fits'))) for cls in target_classes}

    # Match by ID
    input_pairs = []
    output_list = []

    for i in range(len(output_files[target_classes[0]])):  # Match based on the first target class size
        matched_inputs = [input_files[cls][i] for cls in input_classes]
        matched_outputs = [output_files[cls][i] for cls in target_classes]

        if include_mask:
            matched_masks = [mask_files[cls][i] for cls in target_classes]
            matched_outputs.extend(matched_masks)

        input_pairs.append(matched_inputs)
        output_list.append(matched_outputs)

    return input_pairs, output_list

# Initialize random flip layer outside the function
random_flip = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=42)

def augment(inputs, labels):
    # Concatenate inputs and labels along the batch dimension
    concatenated = tf.concat([inputs, labels], axis=-1)
    # Apply random flip
    concatenated = random_flip(concatenated)

    # Apply random 90-degree rotations
    num_rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    concatenated = tf.image.rot90(concatenated, k=num_rotations)

    inputs, labels = tf.split(concatenated, [tf.shape(inputs)[-1], tf.shape(labels)[-1]], axis=-1)
    
    return inputs, labels

def create_dataset(directory, input_class_names, target_class_names, batch_size, is_training=True, include_mask=True):
    """Creates a tf.data.Dataset for given directory."""
    input_pairs, output_list = match_input_output(directory, input_class_names, target_class_names, include_mask)
    # Create TensorFlow Dataset
    file_ds = tf.data.Dataset.from_tensor_slices((input_pairs, output_list))
    if is_training:
        file_ds = file_ds.shuffle(buffer_size=len(output_list[0]))
    
    dataset = file_ds.map(
        lambda input_paths, output_path: load_data(input_paths, output_path),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache the dataset after loading and processing FITS files
    dataset = dataset.cache()
    
    # Apply data augmentation if training
    if is_training:
        dataset = dataset.map(
            lambda inputs, output: augment(inputs, output),
            num_parallel_calls=tf.data.AUTOTUNE # Note, this parallelism can cause minor within-batch shuffling
        ) # Think about doing this after the batch call?

    
    # Define the dataset's structure with TensorSpec
    input_spec = tf.TensorSpec(shape=(256, 256, len(input_class_names)), dtype=tf.float32)
    output_spec = tf.TensorSpec(shape=(256, 256, len(target_class_names)*2), dtype=tf.float32)
    dataset = dataset.map(
        lambda inputs, output: (tf.ensure_shape(inputs, input_spec.shape), tf.ensure_shape(output, output_spec.shape))
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    size = dataset.cardinality()

    # if is_training:
    #     dataset = dataset.repeat()
    return dataset, size

def load_input_data_asarray(indices, classes, path, tensor_shape_X):
    """
    Loads input data into a numpy array from a directory containing the class FITS files.
    
    Parameters:
    - indices: List of indices for the data files.
    - classes: List of class names corresponding to subdirectories or file prefixes.
    - path: Base directory path containing the class data.
    - tensor_shape_X: Shape of the tensor to hold `data_X`.
    
    Returns:
    - data_X: Numpy array containing input data.
    """
    data_X = np.zeros((len(indices), ) + tensor_shape_X, dtype=np.float32)

    for idx, i in enumerate(indices):
        for k, class_name in enumerate(classes):
            file_path = os.path.join(path, f"{class_name}/{class_name}_{i}.fits")
            with fits.open(file_path, memmap=False) as hdu:
                data_X[idx][:, :, k] = hdu[0].data

    return data_X

def load_target_data_asarray(indices, target_classes, path, tensor_shape_Y):
    """
    Loads target data and source catalogs for multiple target classes, and stores WCS information per index.

    Parameters:
    - indices: List of indices for the data files.
    - target_classes: List of target class names corresponding to subdirectories or file prefixes.
    - path: Base directory path containing the target class data.
    - tensor_shape_Y: Shape of the tensor to hold `data_Y`.
    
    Returns:
    - data_Y: Numpy array containing target data.
    - source_catalogs: Dictionary mapping target class names to their respective source catalogs.
    - wcs_array: List of WCS objects, one for each index.
    """
    data_Y = np.zeros((len(indices), ) + tensor_shape_Y, dtype=np.float32)
    source_catalogs = {f"S{class_name[:-2]}": [] for class_name in target_classes}
    wcs_array = [None] * len(indices)

    for idx, i in enumerate(indices):
        for class_name in target_classes:
            file_path = os.path.join(path, f"{class_name}/{class_name}_{i}.fits")
            with fits.open(file_path, memmap=False) as hdu:
                # Load target data
                data_Y[idx, :, :, target_classes.index(class_name)] = hdu[0].data
                
                # Store WCS only once per index
                if wcs_array[idx] is None:
                    wcs_array[idx] = WCS(hdu[0].header)
                
                # Process source catalog from the second HDU
                if len(hdu) > 1:  # Ensure there is a catalog extension
                    catalog = np.array([list(row) for row in hdu[1].data])
                    catalog = np.column_stack((catalog, np.full(len(catalog), i)))  # Add index info
                    source_catalogs[f"S{class_name[:-2]}"].append(catalog)

    # Combine catalogs for each class into a single array
    for band in source_catalogs:
        if source_catalogs[band]:
            source_catalogs[band] = np.vstack(source_catalogs[band])
        else:
            source_catalogs[band] = np.empty((0, 0))

    return data_Y, source_catalogs, wcs_array
