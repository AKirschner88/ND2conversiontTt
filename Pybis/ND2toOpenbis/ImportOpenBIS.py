import os
import json
import getpass
from datetime import datetime
import logging
from Metadataextractionnd2 import extract_nd2_metadata
from pybis import Openbis

logging.basicConfig(level=logging.INFO)

def authenticate_with_openbis():
    """
    Authenticate with OpenBIS using user-provided credentials.
    """
    openbis_host = input("Enter OpenBIS Host (e.g., https://openbis.example.com): ").strip()
    username = input("Enter your OpenBIS username: ").strip()
    password = getpass.getpass("Enter your OpenBIS password: ").strip()
    if not openbis_host or not username or not password:
        raise ValueError("All fields (host, username, and password) must be provided.")
    openbis_instance = Openbis(openbis_host)
    openbis_instance.login(username, password)
    logging.info("Authentication successful!")
    return openbis_instance

def create_experimental_step_with_dataset(
    openbis_instance, experiment_identifier, step_name, file_info, metadata_csv_path, composite_image_paths, results_html
):
    """
    Create an experimental step, attach metadata as a dataset, and include composite images as attachments.
    """
    experiment = openbis_instance.get_experiment(experiment_identifier)
    if not experiment:
        raise ValueError(f"Experiment with identifier '{experiment_identifier}' does not exist. Aborting.")
    step_code = f"{step_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    logging.info(f"Generated unique step code: {step_code}")
    
    sample = openbis_instance.new_object(
        type="EXPERIMENTAL_STEP",
        experiment=experiment,
        code=step_code
    )
    sample.props['$name'] = step_name
    sample.props['experimental_step.experimental_description'] = file_info
    sample.props['experimental_step.experimental_results'] = results_html
    
    try:
        sample.save()
        logging.info(f"Sample '{step_code}' saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save sample '{step_code}': {e}")
        return
    
    try:
        attachment_files = [metadata_csv_path] + composite_image_paths
        dataset = openbis_instance.new_dataset(
            type="ATTACHMENT",
            sample=sample.identifier,
            files=attachment_files,
            props={
                '$name': 'Metadata and Composite Images',
                'notes': 'Metadata extracted from ND2 file and composite images'
            }
        )
        dataset.save()
        logging.info(f"Dataset created and linked to sample '{step_code}'.")
    except Exception as e:
        logging.error(f"Failed to create dataset for sample '{step_code}': {e}")
        return

def generate_results_html(json_path, composite_image_paths, main_folder_name):
    """
    Generate an HTML table summarizing the results, including black/white points and main folder info.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at path: {json_path}")
    with open(json_path, 'r') as f:
        black_white_points = json.load(f)
    logging.info(f"Loaded black-white points: {black_white_points}")
    
    html = "<h3>Experimental Results</h3><table border='1'><tr><th>Channel</th><th>Black Point</th><th>White Point</th></tr>"
    for channel_name, points in black_white_points.items():
        html += f"<tr><td>{channel_name}</td><td>{points['Min']}</td><td>{points['Max']}</td></tr>"
    html += "</table>"
    html += f"<p><strong>Main Folder:</strong> {main_folder_name}</p>"
    return html

def main():
    openbis_instance = authenticate_with_openbis()
    composite_image_paths = [f"output/composite_channel_ch{i}.png" for i in range(4)]
    json_path = "output/black_white_points.json"
    main_folder_name = "output_images"
    default_experiment_identifier = "/AKIRSCHNER/PYBIS/PYBIS_EXP_1"
    nd2_file_path, experimental_description, output_csv_path = extract_nd2_metadata()
    if not nd2_file_path or not experimental_description or not output_csv_path:
        logging.error("Invalid ND2 file or extraction error. Exiting.")
        return
    step_name = os.path.basename(nd2_file_path).replace('.nd2', '')
    results_html = generate_results_html(json_path, composite_image_paths, main_folder_name)
    create_experimental_step_with_dataset(
        openbis_instance=openbis_instance,
        experiment_identifier=default_experiment_identifier,
        step_name=step_name,
        file_info=experimental_description,
        metadata_csv_path=output_csv_path,
        composite_image_paths=composite_image_paths,
        results_html=results_html
    )
    logging.info("Pipeline completed successfully.")

