import os
import json
import numpy as np
from tifffile import imwrite
from concurrent.futures import ProcessPoolExecutor, as_completed
from nd2 import ND2File
import logging

logging.basicConfig(level=logging.INFO)

def load_black_white_points(json_path):
    """Load black-and-white point adjustments from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

def adjust_image_to_black_white(image, vmin, vmax):
    """Adjust the image to new black and white points and convert to 8-bit."""
    adjusted = np.clip(image, vmin, vmax)
    adjusted_8bit = ((adjusted - vmin) / max(1, vmax - vmin) * 255).astype(np.uint8)
    return adjusted_8bit

def calculate_frame_index3D(dimensions, position_idx, t_idx, z_idx):
    return z_idx + position_idx * dimensions['Z'] + t_idx * dimensions['P'] * dimensions['Z']

def calculate_frame_index2D(dimensions, position_idx, t_idx, z_idx):
    return position_idx + t_idx * dimensions['P']

def process_single_frame(nd2_file_path, position_idx, t_idx, z_idx, channel, vmin, vmax, save_path, compress_level):
    """Process a single frame: read, adjust, and save as an 8-bit PNG with specified compression."""
    try:
        with ND2File(nd2_file_path) as nd2_data:
            dimensions = nd2_data.sizes
            # Choose frame index calculation based on whether the file is 2D or 3D.
            if dimensions.get("Z", 1) == 1:
                frame_index = calculate_frame_index2D(dimensions, position_idx, t_idx, z_idx)
            else:
                frame_index = calculate_frame_index3D(dimensions, position_idx, t_idx, z_idx)
            frame = nd2_data._get_frame(frame_index)
            if frame is None or frame.size == 0:
                logging.warning(f"Empty frame at P={position_idx}, T={t_idx}, Z={z_idx}")
                return f"Empty frame at P={position_idx}, T={t_idx}, Z={z_idx}"
            if frame.ndim == 3 and frame.shape[0] > 1:
                frame = frame[channel]
            adjusted_image = adjust_image_to_black_white(frame, vmin, vmax)
            # Save using the provided compression level (0 = no compression, 9 = maximum)
            imwrite(save_path, adjusted_image, compress=compress_level)
        return f"Processed and saved: {save_path}"
    except Exception as e:
        return f"Error processing frame P={position_idx}, T={t_idx}, Z={z_idx}, C={channel}: {e}"

def process_nd2_images_multithreaded(nd2_file_path, output_dir, black_white_points_path, date, initials, compression_percent=100, progress_callback=None):
    """Process images from an ND2 file using multithreading with adjustable PNG compression."""
    black_white_points = load_black_white_points(black_white_points_path)
    results = []

    # Map compression_percent (0-100) to PNG compression level (0-9)
    # Here, 100% means no compression (compress level 0) and 0% means max compression (compress level 9).
    compress_level = round((100 - compression_percent) / 100 * 9)
    logging.info(f"Using PNG compression level: {compress_level} (from {compression_percent}%)")
    
    with ND2File(nd2_file_path) as nd2_data:
        dimensions = nd2_data.sizes
        logging.info(f"ND2 Dimensions: {dimensions}")
        num_positions = dimensions.get("P", 1)
        num_timepoints = dimensions.get("T", 1)
        num_stacks = dimensions.get("Z", 1)
        num_channels = dimensions.get("C", 1)
        
        tasks = []
        for position_idx in range(num_positions):
            position_folder = os.path.join(output_dir, f"{date}{initials}_p{position_idx + 1:04d}")
            os.makedirs(position_folder, exist_ok=True)
            for t_idx in range(num_timepoints):
                for z_idx in range(num_stacks):
                    for channel in range(num_channels):
                        filename = f"{date}{initials}_p{position_idx + 1:04d}_t{t_idx + 1:05d}_z{z_idx + 1:03d}_w{channel:02d}.png"
                        save_path = os.path.join(position_folder, filename)
                        vmin = black_white_points.get(f"Channel_{channel}", {}).get("Min", 0)
                        vmax = black_white_points.get(f"Channel_{channel}", {}).get("Max", 65535)
                        tasks.append((position_idx, t_idx, z_idx, channel, vmin, vmax, save_path, compress_level))
        
        total_tasks = len(tasks)
        logging.info(f"Total tasks to process: {total_tasks}")
        num_workers = os.cpu_count()
        logging.info(f"Using {num_workers} parallel workers for processing.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_frame, nd2_file_path, *task)
                       for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                if progress_callback:
                    progress_callback(f"Converting image {i} of {total_tasks}", i, total_tasks)
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"Error: {e}")
    return results
