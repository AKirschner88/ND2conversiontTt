import os
import numpy as np
from tifffile import imwrite
from nd2 import ND2File
import logging

logging.basicConfig(level=logging.INFO)

def calculate_frame_index(dimensions, position_idx, t_idx, z_idx):
    """Calculate the global frame index based on dimensions.
    
    For 3D samples: index = z_idx + position_idx * Z + t_idx * (P * Z)
    For 2D samples (Z==1): index = position_idx + t_idx * P
    """
    z_dim = dimensions.get("Z", 1)
    P = dimensions.get("P", 1)
    if z_dim == 1:
        return position_idx + t_idx * P
    else:
        return z_idx + position_idx * z_dim + t_idx * P * z_dim

def extract_images_with_nd2_plugin(nd2_file, save_dir="output", date="241203", initials="USER"):
    """
    Extracts the middle Z-stack for the first and last time points of each position for all channels,
    saving them as grayscale TIFF images.
    For 2D samples, it extracts the single available timepoint image.
    """
    from nd2 import ND2File  # Ensure ND2File is imported
    from tifffile import imwrite
    import numpy as np
    import logging
    import time

    with ND2File(nd2_file) as nd2_data:
        dimensions = nd2_data.sizes
        logging.info(f"ND2 Dimensions: {dimensions}")
        num_channels = dimensions.get('C', 1)
        logging.info(f"Number of channels: {num_channels}")
        # Determine if sample is 2D (no Z) or 3D
        if dimensions.get("Z", 1) == 1:
            z_index = 0
        else:
            z_index = dimensions['Z'] // 2
        time_indices = [0, dimensions['T'] - 1]
        num_positions = dimensions.get('P', 1)
        logging.info(f"Extracting from Middle Z: {z_index}, Time Points: {time_indices}, Positions: {num_positions}")
        
        # Loop through each position
        for position_idx in range(num_positions):
            position_folder_name = f"{date}{initials}_p{position_idx + 1:04d}"
            position_dir = os.path.join(save_dir, position_folder_name)
            os.makedirs(position_dir, exist_ok=True)
            
            for channel in range(num_channels):
                logging.info(f"Processing Channel: {channel} at Position: {position_idx}")
                for t_idx in time_indices:
                    index = calculate_frame_index(dimensions, position_idx, t_idx, z_index)
                    # Check if index is valid
                    if index >= dimensions['T'] * dimensions['P'] * dimensions.get("Z", 1):
                        logging.warning(f"Frame index {index} is out of range for Position={position_idx}, Time={t_idx}, Z={z_index}")
                        continue
                    
                    logging.debug(f"Position {position_idx}, Channel {channel}, Time {t_idx}: Calculated frame index = {index}")
                    t0 = time.time()
                    frame = nd2_data._get_frame(index)
                    t1 = time.time()
                    logging.debug(f"Retrieved frame in {t1 - t0:.2f} seconds")
                    
                    if frame is None or frame.size == 0:
                        logging.warning(f"Empty or invalid image at Position={position_idx}, Time={t_idx}, Z={z_index}")
                        continue
                    # For multi-channel 3D data, select the correct channel
                    if frame.ndim == 3 and frame.shape[0] > 1:
                        frame = frame[channel]
                    filename = f"channel_{channel}_time_{t_idx}_z_{z_index}.tiff"
                    save_path = os.path.join(position_dir, filename)
                    imwrite(save_path, frame.astype(np.uint16))
                    logging.info(f"Saved image to {save_path}")


