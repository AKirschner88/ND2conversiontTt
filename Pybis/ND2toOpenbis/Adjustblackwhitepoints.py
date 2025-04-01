import os
import cv2
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def adjust_black_white_cv2(image_dir):
    """
    Interactive adjustment of black and white points for each PNG image (each channel) in a directory.
    For each image, a window is opened with trackbars to adjust Min and Max values. The window title includes the image name.
    When you close the window, the adjustments are saved.
    
    Returns:
        dict: Dictionary mapping each image name (or channel) to its black/white adjustment values.
    """
    # List all PNG files in the directory
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
    if not image_paths:
        raise ValueError(f"No PNG files found in directory: {image_dir}")
    
    black_white_points = {}
    
    # Process each image sequentially
    for idx, path in enumerate(image_paths):
        image_name = os.path.basename(path)
        logging.info(f"Adjust black and white points for {image_name}.")
        
        # Load image
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logging.warning(f"Could not load image: {image_name}. Skipping.")
            continue
        if img.dtype != np.uint16:
            raise ValueError(f"Image {image_name} must be 16-bit grayscale.")
        
        # Create a window for this image with a unique name
        window_name = f"Adjust Black/White - {image_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Define a dummy callback for the trackbars
        def nothing(x):
            pass
        
        # Create trackbars for Min and Max values
        cv2.createTrackbar("Min", window_name, 0, 65535, nothing)
        cv2.createTrackbar("Max", window_name, 65535, 65535, nothing)
        
        # Convert the 16-bit image to an 8-bit image for display
        display_img = (img / 256).astype(np.uint8)
        cv2.imshow(window_name, display_img)
        
        logging.info("Adjust the black and white points for this channel, then close the window to save adjustments.")
        
        # Continuously update the display while the window is open.
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            min_val = cv2.getTrackbarPos("Min", window_name)
            max_val = cv2.getTrackbarPos("Max", window_name)
            if min_val >= max_val:
                # If the range is invalid, show the original display image.
                updated_img = display_img.copy()
            else:
                # Adjust the image: clip and scale to 0-255 for visualization.
                adjusted = np.clip(img, min_val, max_val)
                updated_img = ((adjusted - min_val) / max(1, max_val - min_val) * 255).astype(np.uint8)
            cv2.imshow(window_name, updated_img)
            cv2.waitKey(50)  # Update every 50ms
        
        # Once the window is closed, retrieve the final trackbar values.
        final_min = cv2.getTrackbarPos("Min", window_name)
        final_max = cv2.getTrackbarPos("Max", window_name)
        black_white_points[image_name] = {"Min": final_min, "Max": final_max}
        cv2.destroyWindow(window_name)
        logging.info(f"Saved adjustments for {image_name}: Min={final_min}, Max={final_max}")
    
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
