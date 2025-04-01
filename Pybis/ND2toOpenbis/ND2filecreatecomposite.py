import os
import numpy as np
from tifffile import imwrite
from skimage.io import imread
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO)

logging.basicConfig(level=logging.INFO)

def create_composite_images_for_all_channels(output_dir, num_channels, channel_names=None, dimensions=None):
    """
    Creates labeled composite images for all channels, arranging rows as time points and columns as positions.
    
    For 3D samples, it uses the middle Z-plane (z_index computed from dimensions).
    For 2D samples (dimensions['Z'] == 1 or dimensions is None), it uses the only available plane.
    
    Args:
        output_dir (str): The directory where position directories exist.
        num_channels (int): The number of channels.
        channel_names (list, optional): Custom channel names.
        dimensions (dict, optional): ND2 dimensions; expected to contain key 'Z'. If not provided, assumed 2D.
    """
    if not channel_names:
        channel_names = [f"Channel {i+1}" for i in range(num_channels)]
    
    # Determine if sample is 2D or 3D based on dimensions
    if dimensions is not None:
        z_dim = dimensions.get("Z", 1)
    else:
        z_dim = 1  # Assume 2D if dimensions not provided

    # Compute z_index: if 2D (z_dim == 1) then use 0, else choose middle plane.
    if z_dim == 1:
        z_index = 0
    else:
        z_index = z_dim // 2

    logging.info(f"Creating composite images using z_index = {z_index} (Z-dim = {z_dim})")
    
    # List of position directories (assumes each position has its own subfolder in output_dir)
    position_dirs = sorted(
        [os.path.join(output_dir, d) for d in os.listdir(output_dir)
         if os.path.isdir(os.path.join(output_dir, d))]
    )
    if not position_dirs:
        logging.error("No position directories found in the specified output_dir.")
        return

    logging.info(f"Found {len(position_dirs)} positions.")
    
    # For each channel, gather images from each position.
    for channel_idx, channel_name in enumerate(channel_names):
        logging.info(f"Creating composite image for {channel_name}")
        all_time_images = []
        time_labels = None

        # Process each position directory
        for position_dir in position_dirs:
            logging.info(f"Processing position directory: {position_dir}")
            tiff_files = sorted([
                os.path.join(position_dir, f) 
                for f in os.listdir(position_dir) if f"channel_{channel_idx}" in f
            ])
            logging.info(f"Found {len(tiff_files)} TIFF files for channel {channel_idx} in {position_dir}")
            if not tiff_files:
                logging.warning(f"No TIFF files for channel {channel_idx} found in {position_dir}. Skipping.")
                continue
            # ... process files ...


            # For each file, load image; if sample is 3D, extract the plane at z_index.
            position_images = []
            for f in tiff_files:
                img = imread(f)
                if img.ndim == 3 and z_dim > 1:
                    # For 3D, select the specified z-plane.
                    img = img[z_index]
                position_images.append(img)
            # Assume time_labels can be extracted from filenames of the first valid position.
            if time_labels is None and position_images:
                time_labels = [os.path.basename(f).split("_")[2] for f in tiff_files]
            all_time_images.append(position_images)
        
        if not all_time_images:
            logging.warning(f"No valid images found for {channel_name}. Skipping this channel.")
            continue

        # Use the minimum number of time points across all positions.
        min_rows = min(len(images) for images in all_time_images)
        if time_labels is None:
            time_labels = [f"Time {i+1}" for i in range(min_rows)]
        elif len(time_labels) > min_rows:
            time_labels = time_labels[:min_rows]

        n_rows = min_rows
        n_cols = len(all_time_images)
        # Use the shape of the first image in the first position as a reference.
        sample_image = all_time_images[0][0]
        composite_height = n_rows * sample_image.shape[0]
        composite_width = n_cols * sample_image.shape[1]
        composite_image = np.zeros((composite_height, composite_width), dtype=np.uint16)

        # Fill the composite image grid.
        for row_idx in range(n_rows):
            for col_idx, position_images in enumerate(all_time_images):
                if row_idx >= len(position_images):
                    logging.warning(f"Position at index {col_idx} does not have enough time points.")
                    continue
                img = position_images[row_idx]
                row_start = row_idx * img.shape[0]
                row_end = row_start + img.shape[0]
                col_start = col_idx * img.shape[1]
                col_end = col_start + img.shape[1]
                composite_image[row_start:row_end, col_start:col_end] = img

        # Save the composite image as a 16-bit PNG.
        output_png_path = os.path.join(output_dir, f"{channel_name}.png")
        imwrite(output_png_path, composite_image.astype(np.uint16))
        logging.info(f"Saved composite image for {channel_name} at {output_png_path}")

def draw_labels_on_image(image, time_labels, position_labels, label_height, label_width):
    """
    Adds row (time) and column (position) labels to the composite image.
    """
    image = (image / np.max(image) * 65535).astype(np.uint16)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()
    
    for row_idx, label in enumerate(time_labels):
        y_pos = label_height + row_idx * (image.shape[0] - label_height) // len(time_labels)
        draw.text((10, y_pos), label, fill="white", font=font)
    
    for col_idx, label in enumerate(position_labels):
        x_pos = label_width + col_idx * (image.shape[1] - label_width) // len(position_labels)
        draw.text((x_pos, 10), label, fill="white", font=font)
    
    return np.array(pil_image)

