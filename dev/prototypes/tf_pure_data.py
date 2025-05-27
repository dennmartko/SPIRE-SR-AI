import tensorflow as tf
import numpy as np
from astropy.io import fits
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from astropy.io import fits

import os, glob

base_dir = "/mnt/d/SPIRE-SR-AI/data/processed/dummy_set/Train"
bands    = ["24", "250", "350", "500", "500SR"]  # the 4 you care about

# Collect the set of IDs present in *all* four band folders
common_ids = None
for b in bands:
    fps = glob.glob(f"{base_dir}/{b}/*.fits")
    ids  = { os.path.basename(fp).rsplit("_",1)[1].replace(".fits","") for fp in fps }
    common_ids = ids if common_ids is None else common_ids & ids

common_ids = sorted(common_ids, key=int)
print(f"Found {len(common_ids)} samples present in all bands.")

import numpy as np
import tensorflow as tf
from astropy.io import fits

out_path = "/mnt/d/SPIRE-SR-AI/data/train_4band.tfrecord"
writer   = tf.io.TFRecordWriter(out_path)

for fid in tqdm(common_ids):
    # load & stack the four bands into shape (H, W, 4)
    layers = []
    layers = [fits.getdata(os.path.join(base_dir, b, f"{b}_{fid}.fits")).astype(np.float32) for b in bands]
    stack = np.stack(layers, axis=-1)              # shape: (H, W, 4)

    h, w, c = stack.shape
    feature = {
      "height"  : tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
      "width"   : tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
      "channels": tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
      "image"   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[stack.tobytes()])),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(ex.SerializeToString())

writer.close()
print("TFRecord written:", out_path)

base_dir = "/mnt/d/SPIRE-SR-AI/data/processed/dummy_set/Train"
classes = ["24", "250", "350", "500", "500SR"]
cls_dirs = [f"{base_dir}/{cls}" for cls in classes]
bands = [24, 250, 350, 500, 500]

def _parse_example(serialized):
    features = {
      "height"  : tf.io.FixedLenFeature([], tf.int64),
      "width"   : tf.io.FixedLenFeature([], tf.int64),
      "channels": tf.io.FixedLenFeature([], tf.int64),
      "image"   : tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(serialized, features)
    img = tf.io.decode_raw(parsed["image"], tf.float32)
    img = tf.reshape(img, [parsed["height"], parsed["width"], parsed["channels"]])
    # now img has shape (H, W, 4)! 
    return img

dataset = (
    tf.data.TFRecordDataset("/mnt/d/SPIRE-SR-AI/data/train_4band.tfrecord")
      .shuffle(1024)
      .map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(32)
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
