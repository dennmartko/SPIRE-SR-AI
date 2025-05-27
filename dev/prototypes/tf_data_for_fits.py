import tensorflow as tf
import glob
import re
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

# 1) Make three file‐list Datasets, sorted so they align
base_dir = "/mnt/d/SPIRE-SR-AI/data/processed/dummy_set/Train"
classes = ["24", "250", "350", "500", "500SR"]

datasets = {}
for cls in classes:
    # get all .fits files for this class
    pattern = f"{base_dir}/{cls}/*fits"
    files = glob.glob(pattern)
    # sort on the numeric fileID extracted from filenames like "class_fileID.fits"
    files.sort(key=lambda fp: int(re.search(r"_(\d+)\.fits$", fp).group(1)))
    datasets[cls] = tf.data.Dataset.from_tensor_slices(files)

files_ds = tf.data.Dataset.zip(tuple(datasets[cl] for cl in classes))

# 3) Define a parser that loads each band and stacks them
@tf.function
def _load_and_stack(*paths):
    def load_fits(fp):
        fp = fp.numpy().decode("utf-8")
        with fits.open(fp) as hdul:
            # load data from the primary HDU and ensure it's float32
            return np.expand_dims(hdul[0].data.astype(np.float32), axis=-1)

    imgs = []
    for fp in paths:
        # wrap the python function so it can be used in the graph
        img = tf.py_function(load_fits, [fp], tf.float32)
        # optionally, set the shape if known, e.g., img.set_shape([H, W])
        img.set_shape([256, 256, 1])
        imgs.append(img)

    # concatenate into H×W×4x1 tensor
    img = tf.concat(imgs, axis=-1)
    return img

# 4) Build your final dataset
dataset = (
    files_ds
    .shuffle(1024)                      # for training
    .map(_load_and_stack, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

# display 5 samples from the dataset
for batch in dataset.take(1):
    print(f"Batch shape: {batch.shape}")

    imgs = batch.numpy()  # shape: (batch_size, 256, 256, num_channels)
    for i in range(min(5, imgs.shape[0])):
        img = imgs[i]
        num_channels = img.shape[-1]
        # create subplots: one for each channel
        fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
        # if there is only one channel, axes is not a list
        if num_channels == 1:
            axes = [axes]
        for j, ax in enumerate(axes):
            ax.imshow(img[..., j], cmap='gray')
            ax.set_title(f"Sample {i} - Channel {j}")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"./visualisation_{i}.png", dpi=300)
        plt.close()
