from PIL import Image, ImageDraw
import numpy as np

def create_image_mask(coordinates, image_dimensions, radius):
    # Create a new image with the specified dimensions
    img = Image.new('L', image_dimensions, 0)  # 'L' mode for grayscale image

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Draw circles around each coordinate
    for coord in coordinates:
        x, y = coord
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)

    return np.array(img).astype(np.float32)

def generate_image_masks(cat_img, image_dimensions, radius=8):
    # Iterate over unique ImageIDs in the target catalog
    if cat_img.shape[0] != 0:
        # Create a mask for the current ImageID
        galx_mask = create_image_mask(cat_img[["xpix", "ypix"]].values, image_dimensions, radius)
        # Store the mask in the galx_masks array
    else:
        galx_mask = np.zeros(image_dimensions, dtype=np.float32)
    return galx_mask

def generate_image_masks_old(target_cat, train_indices, image_dimensions, radius):
    # Initialize an array to store masks
    galx_masks = np.zeros((len(train_indices), *image_dimensions), dtype=np.float32)

    # Iterate over unique ImageIDs in the target catalog
    for ID in train_indices:
        # Select rows corresponding to the current ImageID
        target_cat_img0 = target_cat[target_cat["ImageID"] == ID]
        if target_cat_img0.shape[0] != 0:
            # Create a mask for the current ImageID
            galx_mask = create_image_mask(target_cat_img0[["xpix", "ypix"]].values, image_dimensions, radius)
            # Store the mask in the galx_masks array
            galx_masks[ID] = galx_mask
        else:
            galx_masks[ID] = np.zeros(image_dimensions, dtype=np.float32)
    return galx_masks