import json
import nd2
import os
import pandas as pd
import logging
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from nd2 import ND2File
from TATexp import create_tatexp_xml

logging.basicConfig(level=logging.INFO)

def select_file():
    """Let the user select a file and return its path, dimensions, and a dedicated output directory."""
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    import os
    import nd2
    from nd2 import ND2File
    import logging

    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(
        title="Select ND2 File",
        filetypes=[("Microscope files", "*.nd2;*.lif"), ("All files", "*.*")]
    )
    if not file_path:
        logging.error("No file selected. Exiting.")
        exit()
    try:
        with ND2File(file_path) as nd2_data:
            dimensions = nd2_data.sizes
            logging.info(f"ND2 Dimensions: {dimensions}")
    except Exception as e:
        logging.error(f"Error opening ND2 file: {e}")
        exit()
    
    # Get the base directory of the file and its base name
    base_dir = os.path.dirname(file_path)
    file_base = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "250307AK35_WNTbiosensors_esc001"
    
    # Create an analysis folder named <file_base>_analysis in the same directory as the ND2 file
    analysis_folder = os.path.join(base_dir, f"{file_base}_analysis")
    os.makedirs(analysis_folder, exist_ok=True)
    
    # Extract the desired subfolder name (e.g. "250307AK35") from the ND2 file name.
    # This example assumes that the file name is formatted with an underscore,
    # and you want the first part.
    subfolder_name = file_base.split("_")[0]
    output_dir = os.path.join(analysis_folder, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory created: {output_dir}")
    
    return file_path, dimensions, output_dir

def flatten_metadata(metadata, parent_key='', sep='|'):
    """Flattens a nested dictionary into a single-level dictionary."""
    items = []
    for k, v in metadata.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_metadata(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_laser_metadata(laser_info_raw):
    """Parse the laser configuration info and return a list of laser settings rows."""
    laser_rows = []
    current_laser = None
    detector = "Unknown"
    scanner = "Unknown"
    gain = "Unknown"
    power = "Unknown"
    emission_range = "Unknown"
    line_averaging = "N/A"
    current_zoom = "Unknown"
    
    for line in laser_info_raw.splitlines():
        line = line.strip()
        if line.startswith("Scanner"):
            scanner = line.split(":")[1].strip()
        elif line.startswith("Detector"):
            detector = line.split(":")[1].strip()
        elif line.startswith("Gain"):
            gain = line.split(":")[1].strip()
        elif line.startswith("Line Averaging"):
            line_averaging = line.split(":")[1].strip()
        elif line.startswith("Emission Range"):
            emission_range = line.split(":")[1].strip()
        elif line.startswith("Laser") and "nm" in line:
            current_laser = line.split(":")[0].strip()  # e.g., "Laser 488 nm"
        elif current_laser and "Power:" in line:
            power = line.split(":")[1].strip()
        elif line.startswith("Zoom:"):
            current_zoom = line.split(":")[1].strip()
            if current_laser:
                laser_rows.append([
                    current_laser,
                    detector,
                    scanner,
                    emission_range,
                    gain,
                    power,
                    current_zoom,
                    line_averaging
                ])
                current_laser = None
                detector = "Unknown"
                scanner = "Unknown"
                gain = "Unknown"
                power = "Unknown"
                current_zoom = "Unknown"
                emission_range = "Unknown"
                line_averaging = "N/A"
    return laser_rows

def extract_nd2_metadata():
    """
    Extract metadata from an ND2 file, save it as CSV, and generate an experimental description in HTML.
    All output files are stored in the dedicated output folder.
    """
    file_path, dimensions, output_dir = select_file()
    if not file_path:
        logging.error("No valid file selected.")
        return None, None, None, None
    
    # Generate the metadata CSV file path
    output_csv_path = os.path.join(output_dir, os.path.basename(file_path).replace('.nd2', '_metadata.csv'))
    
    file_info_rows = []  # For general metadata
    laser_power_rows = []  # For laser configuration details
    
    try:
        with ND2File(file_path) as nd2_file:
            file_info_rows.append(["File Dimensions", str(nd2_file.shape)])
            logging.info(f"File Dimensions: {nd2_file.shape}")
            
            # Retrieve unstructured metadata and log it
            all_metadata = nd2_file.unstructured_metadata()
            logging.info(f"Unstructured metadata: {all_metadata}")
            
            flattened_metadata = flatten_metadata(all_metadata)
            logging.info(f"Flattened metadata: {flattened_metadata}")
            
            metadata_df = pd.DataFrame(list(flattened_metadata.items()), columns=["Key", "Value"])
            
            logging.info(f"Attempting to write metadata CSV to: {output_csv_path}")
            metadata_df.to_csv(output_csv_path, index=False)
            if os.path.exists(output_csv_path):
                logging.info(f"Metadata table saved as '{output_csv_path}'.")
            else:
                logging.error(f"Failed to create metadata CSV at '{output_csv_path}'.")
            
        resolution_width = flattened_metadata.get('ImageAttributesLV|SLxImageAttributes|uiWidth', "Unknown")
        resolution_height = flattened_metadata.get('ImageAttributesLV|SLxImageAttributes|uiHeight', "Unknown")
        file_info_rows.append(["Resolution", f"{resolution_width}x{resolution_height}"])
        
        objective_value = flattened_metadata.get('ImageCalibrationLV|0|SLxCalibration|Objective', "Unknown")
        file_info_rows.append(["Objective", objective_value])
        
        macro_command = flattened_metadata.get('ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|wsCommandBeforeCapture', "N/A")
        macro_active = "Yes" if macro_command != "N/A" else "No"
        file_info_rows.append(["Macro Active", macro_active])
        file_info_rows.append(["Macro Command", macro_command])
        
        number_of_positions = flattened_metadata.get('ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|uLoopPars|uiCount', "Unknown")
        file_info_rows.append(["Number of Positions", number_of_positions])
        
        laser_info_raw = flattened_metadata.get('ImageTextInfoLV|SLxImageTextInfo|TextInfoItem_6', "")
        if laser_info_raw:
            laser_power_rows = parse_laser_metadata(laser_info_raw)
    except Exception as e:
        logging.error(f"Error during metadata extraction: {e}")
        return None, None, None, None
    
    def create_html_table(rows, headers):
        html = "<table border='1'>"
        html += "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
        for row in rows:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        html += "</table>"
        return html
    
    general_metadata_table = create_html_table(file_info_rows, ["Key", "Value"])
    laser_table_headers = ["Laser Wavelength", "Detector", "Scanner", "Emission Range", "Gain", "Laser Power", "Zoom", "Line Averaging"]
    laser_power_table = create_html_table(laser_power_rows, laser_table_headers)
    
    experimental_description = f"{general_metadata_table}<br><br>{laser_power_table}<br><br>"
    
    return file_path, experimental_description, output_csv_path, output_dir

def generate_metadata(file_path, dimensions, output_dir):
    """
    Generate metadata from the given ND2 file and save it as a CSV in the output directory.
    Returns a tuple: (experimental_description, metadata_csv_path)
    """
    from nd2 import ND2File  # ensure you import here if needed
    import pandas as pd
    output_csv_path = os.path.join(output_dir, os.path.basename(file_path).replace('.nd2', '_metadata.csv'))
    
    file_info_rows = []
    laser_power_rows = []
    
    try:
        with ND2File(file_path) as nd2_file:
            file_info_rows.append(["File Dimensions", str(nd2_file.shape)])
            all_metadata = nd2_file.unstructured_metadata()
            flattened_metadata = flatten_metadata(all_metadata)
            metadata_df = pd.DataFrame(list(flattened_metadata.items()), columns=["Key", "Value"])
            metadata_df.to_csv(output_csv_path, index=False)
            # Log that the CSV was written
            if os.path.exists(output_csv_path):
                print(f"Metadata CSV saved as '{output_csv_path}'.")
            else:
                print(f"Failed to create metadata CSV at '{output_csv_path}'.")
                
        resolution_width = flattened_metadata.get('ImageAttributesLV|SLxImageAttributes|uiWidth', "Unknown")
        resolution_height = flattened_metadata.get('ImageAttributesLV|SLxImageAttributes|uiHeight', "Unknown")
        file_info_rows.append(["Resolution", f"{resolution_width}x{resolution_height}"])
        objective_value = flattened_metadata.get('ImageCalibrationLV|0|SLxCalibration|Objective', "Unknown")
        file_info_rows.append(["Objective", objective_value])
        macro_command = flattened_metadata.get('ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|wsCommandBeforeCapture', "N/A")
        macro_active = "Yes" if macro_command != "N/A" else "No"
        file_info_rows.append(["Macro Active", macro_active])
        file_info_rows.append(["Macro Command", macro_command])
        number_of_positions = flattened_metadata.get('ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|uLoopPars|uiCount', "Unknown")
        file_info_rows.append(["Number of Positions", number_of_positions])
        laser_info_raw = flattened_metadata.get('ImageTextInfoLV|SLxImageTextInfo|TextInfoItem_6', "")
        if laser_info_raw:
            laser_power_rows = parse_laser_metadata(laser_info_raw)
    except Exception as e:
        print(f"Error during metadata extraction: {e}")
        raise e
    
    file_base = os.path.splitext(os.path.basename(file_path))[0]  # e.g. "250301AK35_WNTbiosensors_esc001"
    date_str = file_base[:6]   # "250301"
    user_str = file_base[6:8]  # "AK"
    setup_number = file_base[8:10]  # "35"

    tatexp_xml_path = create_tatexp_xml(
        flattened_metadata,
        dimensions,
        output_dir,
        date_str=date_str,  # or parse from the ND2 file name
        user_str=user_str,
        setup_number=setup_number
    )


    # Build an HTML description from the metadata
    def create_html_table(rows, headers):
        html = "<table border='1'>"
        html += "<tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr>"
        for row in rows:
            html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        html += "</table>"
        return html

    general_metadata_table = create_html_table(file_info_rows, ["Key", "Value"])
    laser_table_headers = ["Laser Wavelength", "Detector", "Scanner", "Emission Range", "Gain", "Laser Power", "Zoom", "Line Averaging"]
    laser_power_table = create_html_table(laser_power_rows, laser_table_headers)
    experimental_description = f"{general_metadata_table}<br><br>{laser_power_table}<br><br>"
    
    return experimental_description, output_csv_path, tatexp_xml_path