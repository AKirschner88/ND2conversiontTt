import os
import xml.etree.ElementTree as ET
import logging
import re

def create_tatexp_xml(
    flattened_metadata,
    dimensions,
    output_dir,
    date_str="",
    user_str="",
    setup_number=""
):
    """
    Create a TATexp.xml file with position, wavelength, objective, etc. information
    using ND2 flattened metadata and dimension info.
    
    Args:
        flattened_metadata (dict): Flattened ND2 metadata (key -> value).
        dimensions (dict): ND2 dimension info, e.g. {'P': #positions, 'T': #timepoints, ...}.
        output_dir (str): Folder where the XML file should be written.
        date_str (str): Date code, e.g. "250301".
        user_str (str): User code, e.g. "AK".
        setup_number (str): Setup number, e.g. "35".
    Returns:
        str: Full path to the created XML file.
    """
    logging.info("Starting TATexp.xml creation...")
    # Number of positions and channels from ND2 dimensions
    num_positions = dimensions.get('P', 1)
    num_channels = dimensions.get('C', 1)
    
    # Example: get image width/height from metadata or from ND2 dimension keys
    # Adjust these keys or fallback logic as needed:
    width = flattened_metadata.get("ImageAttributesLV|SLxImageAttributes|uiWidth", 512)
    height = flattened_metadata.get("ImageAttributesLV|SLxImageAttributes|uiHeight", 512)
    # ND2 might store objective in "ImageCalibrationLV|0|SLxCalibration|Objective"
    objective = flattened_metadata.get("ImageCalibrationLV|0|SLxCalibration|Objective", "4")
    match = re.search(r"(\d+)", objective)
    if match:
        objective_value = match.group(1)
    else:
        objective_value = "4"


    # Build the XML
    root = ET.Element("TATSettings")
    
    # Version
    version = ET.SubElement(root, "TTTConvertExperimentVersion")
    version.text = "160304"
    
    # Positions
    ET.SubElement(root, "PositionCount", attrib={"count": str(num_positions)})
    position_data = ET.SubElement(root, "PositionData")
    
    for i in range(num_positions):
        # Example: For position i, the flattened metadata keys might be:
        # "ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|uLoopPars|Points|i000000000i|dPosX"
        # The outer i0000000000 is the experiment index (often 0 if there's only one experiment),
        # while the inner iXXXXXXXXXX is the position index, zero-padded to 10 digits.
        key_x = f"ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|uLoopPars|Points|i{i:010d}|dPosX"
        key_y = f"ImageMetadataLV|SLxExperiment|ppNextLevelEx|i0000000000|uLoopPars|Points|i{i:010d}|dPosY"
        pos_x = flattened_metadata.get(key_x, "0")
        pos_y = flattened_metadata.get(key_y, "0")
        
        pos_info = ET.SubElement(position_data, "PositionInformation")
        ET.SubElement(
            pos_info,
            "PosInfoDimension",
            attrib={
                "index": f"{i+1:04d}",  # e.g. 0001, 0002, etc.
                "posX": str(pos_x),
                "posY": str(pos_y),
                "comments": ""
            }
        )
    
    # Wavelengths
    ET.SubElement(root, "WavelengthCount", attrib={"count": str(num_channels)})
    wavelength_data = ET.SubElement(root, "WavelengthData")
    
    for ch in range(num_channels):
        wl_info = ET.SubElement(wavelength_data, "WavelengthInformation")
        ET.SubElement(
            wl_info,
            "WLInfo",
            attrib={
                "ImageType": "png",
                "Name": f"{ch:02d}",  # "00", "01", etc.
                "height": str(height),
                "width": str(width)
            }
        )
    
    # Objective
    ET.SubElement(root, "CurrentObjectiveMagnification", attrib={"value": str(objective_value)})
    # Often 1.0 for the TV adapter
    ET.SubElement(root, "CurrentTVAdapterMagnification", attrib={"value": "1.0"})
    
    # Minimal "CellsAndConditions" block
    cells_conditions = ET.SubElement(root, "CellsAndConditions")
    ET.SubElement(cells_conditions, "NumberOfCellTypes", attrib={"value": "1"})
    cact_cell_types = ET.SubElement(cells_conditions, "CellsAndConditions_CellTypes")
    cts_cell_type = ET.SubElement(cact_cell_types, "CNC_CTs_CellType")
    for tag in ["PrimaryCell", "Name", "Species", "Sex", "Organ", "Age", "Purification", "Comment"]:
        ET.SubElement(cts_cell_type, tag, attrib={"value": ""})
    
    # Build final tree
    tree = ET.ElementTree(root)
    # Example naming: "250301AK35_TATexp.xml"
    xml_filename = f"{date_str}{user_str}{setup_number}_TATexp.xml"
    final_path = os.path.join(output_dir, xml_filename)
    
    # Write with XML declaration
    tree.write(final_path, encoding="utf-8", xml_declaration=True)
    logging.info(f"TATexp.xml created at: {final_path}")
    return final_path
