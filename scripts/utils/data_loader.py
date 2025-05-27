import tensorflow as tf
import numpy as np
import os
import re
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

def augment_tf(data):
    # Apply random flip
    data = random_flip(data)

    # Apply random 90-degree rotations
    num_rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    data = tf.image.rot90(data, k=num_rotations)
    return data

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
    output_spec = tf.TensorSpec(shape=(256, 256, len(target_class_names)*2), dtype=tf.float32) # Factor 2 because each target class has a mask
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


@tf.function
def _load_and_stack(*paths):
    def load_fits(fp):
        fp = fp.numpy().decode("utf-8")
        with fits.open(fp) as hdul:
            return np.expand_dims(hdul[0].data.astype(np.float32), axis=-1)

    imgs = []
    for fp in paths:
        img = tf.py_function(load_fits, [fp], tf.float32)
        img.set_shape([256, 256, 1])
        imgs.append(img)

    img = tf.concat(imgs, axis=-1)  # Expected shape: [256, 256, num_channels]
    return img

def split_input_labels(tensor, input_classes, target_classes):
    """
    Splits a concatenated tensor into inputs and labels.
    Assumes that the tensor's last dimension is ordered with input channels first
    and then label channels (which may include auxiliary channels like masks).

    Parameters:
    - tensor: A tf.Tensor of shape (..., channels) (may include a batch dimension).
    - input_classes: List of input class names.
    - target_classes: List of target class names.
      (For example, if each target class provides two channels, the remainder channels are split accordingly.)

    Returns:
    - inputs: Sub-tensor corresponding to the input channels.
    - labels: Sub-tensor corresponding to the label channels.
    """
    input_channels = len(input_classes)
    # The rest channels are considered labels.
    inputs = tensor[..., :input_channels]
    labels = tensor[..., input_channels:]
    return inputs, labels

def create_dataset_tf(directory, input_classes, target_classes, batch_size, is_training=True):
    # Make file‐list Datasets, sorted so that channels align
    classes = input_classes + target_classes + [target_cl + "_mask" for target_cl in target_classes]

    datasets = {}
    for cls in classes:
        # get all .fits files for this class
        pattern = f"{directory}/{cls}/*fits"
        files = glob(pattern)
        # sort on the numeric fileID extracted from filenames like "class_fileID.fits"
        files.sort(key=lambda fp: int(re.search(r"_(\d+)\.fits$", fp).group(1)))
        datasets[cls] = tf.data.Dataset.from_tensor_slices(files)

    files_ds = tf.data.Dataset.zip(tuple(datasets[cl] for cl in classes))

    if is_training:
        dataset = (
            files_ds
            .map(_load_and_stack, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .shuffle(1024)
            .map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        dataset = (
            files_ds
            .map(_load_and_stack, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    return dataset, dataset.cardinality()


def load_input_data_asarray(indices, classes, path, tensor_shape_X, progress=None):
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

    progress_task = progress.add_task("Loading input data...", total=len(indices)) if progress else None

    for idx, i in enumerate(indices):
        for k, class_name in enumerate(classes):
            file_path = os.path.join(path, f"{class_name}/{class_name}_{i}.fits")
            with fits.open(file_path, memmap=False) as hdu:
                data_X[idx][:, :, k] = hdu[0].data
        if progress:
            progress.update(progress_task, advance=1)

    return data_X

def load_target_data_asarray_deprecated(indices, target_classes, path, tensor_shape_Y):
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

def load_target_data_asarray(indices, target_classes, path, tensor_shape_Y, progress=None):
    """
    Loads target data and source catalogs for multiple target classes, and stores WCS information per index.

    Parameters:
    - indices: List of indices for the data files.
    - target_classes: List of target class names corresponding to subdirectories or file prefixes.
    - path: Base directory path containing the target class data.
    - tensor_shape_Y: Shape of the tensor to hold `data_Y`.
    
    Returns:
    - data_Y: Numpy array containing target data.
    - wcs_array: dictionary of WCS objects, one for each index.
    """
    # Initialize target data array and dictionary for WCS objects
    data_Y = np.zeros((len(indices), ) + tensor_shape_Y, dtype=np.float32)
    wcs_dict = {}

    progress_task = progress.add_task("Loading target data...", total=len(indices)) if progress else None
    for idx, file_id in enumerate(indices):
        for class_name in target_classes:
            file_path = os.path.join(path, f"{class_name}/{class_name}_{file_id}.fits")
            with fits.open(file_path, memmap=False) as hdu:
                # Load target data
                data_Y[idx, :, :, target_classes.index(class_name)] = hdu[0].data
                
                # Store WCS only once per index
                if file_id not in wcs_dict:
                    wcs_dict[file_id] = WCS(hdu[0].header)
        if progress:
            progress.update(progress_task, advance=1)

    return data_Y, wcs_dict


# Function to create patches
@tf.function
def extract_patches(input_tensor, patch_size):
    # Extract patches
    patches = tf.image.extract_patches(
        images=input_tensor,
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, patch_size[0], patch_size[1], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    # Get dimensions
    batch_size = tf.shape(input_tensor)[0]
    num_patches_h = tf.shape(patches)[1]
    num_patches_w = tf.shape(patches)[2]
    num_patches = num_patches_h * num_patches_w
    # patch_dim = patch_size[0] * patch_size[1] * tf.shape(input_tensor)[-1]
    
    # Reshape to desired shape (Batchsize * n, patch_size[0], patch_size[1], C)
    patches = tf.reshape(patches, [batch_size * num_patches, patch_size[0], patch_size[1], -1])
    return patches

# Function to reconstruct from patches
@tf.function
def reconstruct_from_patches(patches, original_shape, patch_size):
    batch_size = original_shape[0]
    patch_h, patch_w = patch_size
    num_patches_h = original_shape[1] // patch_h
    num_patches_w = original_shape[2] // patch_w
    num_channels = original_shape[3]

    # Reshape patches back to grid structure
    patches = tf.reshape(patches, [batch_size, num_patches_h, num_patches_w, patch_h, patch_w, num_channels])
    patches = tf.transpose(patches, perm=[0, 1, 3, 2, 4, 5])  # Rearrange to match original dimensions
    reconstructed = tf.reshape(patches, [batch_size, original_shape[1], original_shape[2], num_channels])
    return reconstructed