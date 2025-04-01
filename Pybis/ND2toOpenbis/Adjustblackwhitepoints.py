import os
import cv2
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def adjust_black_white_cv2(image_dir):
    """
    Interactive black-and-white point adjustment for all PNG images in a directory.
    """
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
    if not image_paths:
        raise ValueError(f"No PNG files found in directory: {image_dir}")
    
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_paths]
    black_white_points = {}

    for i, img in enumerate(images):
        if img is None:
            raise ValueError(f"Could not load image: {image_paths[i]}")
        if img.dtype != np.uint16:
            raise ValueError(f"Image {image_paths[i]} must be 16-bit grayscale.")
    
    def create_update_callback(image_index, image_name):
        def update_display(val):
            vmin = cv2.getTrackbarPos(f"Min_{image_name}", image_name)
            vmax = cv2.getTrackbarPos(f"Max_{image_name}", image_name)
            if vmin >= vmax:
                return  # Ignore invalid ranges
            black_white_points[image_name] = {"Min": vmin, "Max": vmax}
            display_image = np.clip(images[image_index], vmin, vmax)
            display_image = ((display_image - vmin) / max(1, vmax - vmin) * 65535).astype(np.uint16)
            cv2.imshow(image_name, (display_image / 256).astype(np.uint8))
        return update_display
    
    for idx, path in enumerate(image_paths):
        image_name = os.path.basename(path)
        cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
        callback = create_update_callback(idx, image_name)
        cv2.createTrackbar(f"Min_{image_name}", image_name, 0, 65535, callback)
        cv2.createTrackbar(f"Max_{image_name}", image_name, 65535, 65535, callback)
        cv2.imshow(image_name, (images[idx] / 256).astype(np.uint8))
    
    logging.info("Adjust black and white points for each channel, then press any key to save and exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return black_white_points

def save_black_white_points(black_white_points, output_path="black_white_points.json"):
    """
    Save the black-and-white points for each channel to a JSON file.
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(black_white_points, f, indent=4)
    logging.info(f"Saved black-and-white points to {output_path}")

if __name__ == "__main__":
    composite_image_dir = "output"  # Adjust this as necessary
    if not os.path.exists(composite_image_dir):
        logging.error(f"Directory does not exist: {composite_image_dir}")
    else:
        points = adjust_black_white_cv2(composite_image_dir)
        save_black_white_points(points)
