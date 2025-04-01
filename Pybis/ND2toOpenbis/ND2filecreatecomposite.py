import os
import numpy as np
import math
from tifffile import imwrite
from skimage.io import imread
from PIL import Image, ImageDraw, ImageFont
import logging

logging.basicConfig(level=logging.INFO)

def create_composite_images_for_all_channels(output_dir, num_channels, channel_names=None, dimensions=None, max_positions_per_row=10):
    """
    Creates labeled composite images for all channels.
    
    Instead of placing all positions in one row, this function arranges positions in a grid.
    For each channel, it uses the first image (or time point) from each position to form the grid.
    
    Args:
        output_dir (str): Directory where the extracted images are stored.
        num_channels (int): Number of channels.
        channel_names (list, optional): Custom channel names.
        dimensions (dict, optional): ND2 dimensions.
        max_positions_per_row (int): Maximum number of positions per row.
    """
    if not channel_names:
        channel_names = [f"Channel {i+1}" for i in range(num_channels)]
    
    # Get the list of position directories in the output folder.
    position_dirs = sorted([os.path.join(output_dir, d) for d in os.listdir(output_dir)
                              if os.path.isdir(os.path.join(output_dir, d))])
    if not position_dirs:
        logging.error("No position directories found in the specified output_dir.")
        return

    logging.info(f"Found {len(position_dirs)} position directories.")
    
    # For each channel, build a composite from the first image (e.g. time point 0) in each position.
    for channel_idx, channel_name in enumerate(channel_names):
        logging.info(f"Creating composite image for {channel_name}")
        position_images = []  # This will hold one image per position for the given channel.
        for pos_dir in position_dirs:
            # Assume filenames contain "channel_{channel_idx}" and a time indicator.
            # We'll filter for files that match and pick the first one.
            files = sorted([os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
                            if f"channel_{channel_idx}" in f])
            logging.info(f"  {pos_dir}: found {len(files)} files for channel {channel_idx}")
            if not files:
                logging.warning(f"  No files for channel {channel_idx} in {pos_dir}")
                continue
            # Use the first file as the representative image.
            try:
                img = imread(files[0])
                # If the image is 3D and dimensions indicate 3D, extract the middle plane
                if img.ndim == 3 and dimensions and dimensions.get("Z", 1) > 1:
                    z_index = dimensions["Z"] // 2
                    img = img[z_index]
                position_images.append(img)
            except Exception as e:
                logging.error(f"  Failed to read image {files[0]}: {e}")
        
        if not position_images:
            logging.warning(f"No valid images found for channel {channel_idx}. Skipping composite creation for this channel.")
            continue
        
        # Arrange position_images in a grid.
        n_positions = len(position_images)
        n_cols = min(max_positions_per_row, n_positions)
        n_rows = math.ceil(n_positions / max_positions_per_row)
        logging.info(f"Arranging {n_positions} positions in {n_rows} rows and {n_cols} columns.")
        
        # Use the shape of the first image as reference.
        sample_image = position_images[0]
        cell_height, cell_width = sample_image.shape[:2]
        composite_height = n_rows * cell_height
        composite_width = n_cols * cell_width
        # Create a blank composite image (assuming grayscale 16-bit images).
        composite_image = np.zeros((composite_height, composite_width), dtype=np.uint16)
        
        # Place each image into the composite grid.
        for idx, img in enumerate(position_images):
            row_idx = idx // max_positions_per_row
            col_idx = idx % max_positions_per_row
            y0 = row_idx * cell_height
            x0 = col_idx * cell_width
            # Ensure image dimensions match; if not, you may need to resize.
            composite_image[y0:y0+cell_height, x0:x0+cell_width] = img
        output_png_path = os.path.join(output_dir, f"{channel_name}.png")
        imwrite(output_png_path, composite_image)
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

