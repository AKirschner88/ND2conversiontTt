import os
import sys
import time
from datetime import datetime
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QVBoxLayout,
    QPushButton, QProgressBar, QWidget, QGridLayout, QCheckBox, 
    QMessageBox, QDialog, QPlainTextEdit, QComboBox, QTreeWidget, QTreeWidgetItem,
    QHBoxLayout
)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt

# Import updated modules.
from Metadataextractionnd2 import select_file, generate_metadata
from ND2imageextraction import extract_images_with_nd2_plugin
from ND2filecreatecomposite import create_composite_images_for_all_channels
from Adjustblackwhitepoints import adjust_black_white_cv2, save_black_white_points
from nd2to8bitpng import process_nd2_images_multithreaded
from ImportOpenBIS import (
    authenticate_with_openbis, 
    create_experimental_step_with_dataset, 
    generate_results_html
)
from pybis import Openbis

logging.basicConfig(level=logging.INFO)

# --- Custom Qt Logging Handler ---
class QtHandler(logging.Handler, QObject):
    log_signal = pyqtSignal(str)
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
    
    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

# --- OpenBIS Explorer Dialog ---
class OpenBISExplorerDialog(QDialog):
    """
    Displays a collapsible tree of spaces → projects → experiments.
    Only spaces with a code matching your login (in uppercase) are shown.
    """
    def __init__(self, openbis_instance, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OpenBIS Explorer")
        self.openbis_instance = openbis_instance
        self.selected_experiment_identifier = None
        
        self.layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("OpenBIS Hierarchy")
        self.layout.addWidget(self.tree)
        
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self.accept_selection)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(button_layout)
        
        self.populate_tree()
    
    def populate_tree(self):
        try:
            all_spaces = self.openbis_instance.get_spaces()
            login_name = (self.openbis_instance.username if hasattr(self.openbis_instance, "username") else "").upper()
            filtered_spaces = [space for space in all_spaces if space.code.upper() == login_name]

            logging.info(f"Login name: {login_name}")
            logging.info(f"Filtered spaces: {[s.code for s in filtered_spaces]}")

            if not filtered_spaces:
                logging.warning("No matching spaces found in openBIS.")
                return
            
            # IMPORTANT: Iterate over filtered_spaces, NOT all_spaces
            for space in filtered_spaces:
                space_item = QTreeWidgetItem([space.code])
                space_item.setData(0, Qt.UserRole, space)
                self.tree.addTopLevelItem(space_item)
                
                projects = space.get_projects()
                logging.info(f"  Found {len(projects)} projects in space {space.code}.")
                for proj in projects:
                    proj_item = QTreeWidgetItem([proj.code])
                    proj_item.setData(0, Qt.UserRole, proj)
                    space_item.addChild(proj_item)

                    experiments = proj.get_experiments()
                    logging.info(f"    Found {len(experiments)} experiments in project {proj.code}.")
                    for exp in experiments:
                        exp_item = QTreeWidgetItem([exp.code])
                        exp_item.setData(0, Qt.UserRole, exp)
                        proj_item.addChild(exp_item)

            self.tree.expandAll()
        except Exception as e:
            logging.error(f"Failed to populate openBIS hierarchy: {e}")

    def accept_selection(self):
        current_item = self.tree.currentItem()
        if not current_item:
            logging.warning("No item selected.")
            self.reject()
            return
        data = current_item.data(0, Qt.UserRole)
        if hasattr(data, 'identifier') and data.identifier is not None:
            self.selected_experiment_identifier = data.identifier
            logging.info(f"Selected experiment: {self.selected_experiment_identifier}")
            self.accept()
        else:
            logging.warning("Please select an experiment, not a space or project.")
            self.reject()
    
    def get_selected_experiment_identifier(self):
        return self.selected_experiment_identifier

# --- OpenBIS Login Dialog ---
class OpenBISLoginDialog(QDialog):
    def __init__(self, openbis_host):
        super().__init__()
        self.setWindowTitle("OpenBIS Login")
        self.layout = QVBoxLayout(self)
        self.openbis_host = openbis_host
        self.host_label = QLabel(f"OpenBIS Host: {self.openbis_host}")
        self.layout.addWidget(self.host_label)
        self.username_label = QLabel("Username:")
        self.layout.addWidget(self.username_label)
        self.username_input = QLineEdit(self)
        self.layout.addWidget(self.username_input)
        self.password_label = QLabel("Password:")
        self.layout.addWidget(self.password_label)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        self.layout.addWidget(self.password_input)
        self.login_button = QPushButton("Login", self)
        self.login_button.clicked.connect(self.accept_login)
        self.layout.addWidget(self.login_button)
        self.openbis_instance = None

    def accept_login(self):
        try:
            username = self.username_input.text().strip()
            password = self.password_input.text().strip()
            if not username or not password:
                raise ValueError("Both username and password must be provided.")
            self.openbis_instance = Openbis(self.openbis_host)
            self.openbis_instance.login(username, password)
            # Save the username in the instance for later use
            self.openbis_instance.username = username
            QMessageBox.information(self, "Success", "Logged into OpenBIS successfully!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Login Failed", f"Error: {e}")

# --- Worker Thread ---
class WorkerThread(QThread):
    # Step indices: 0: Generate Metadata, 1: Extract Images, 2: Create Composite,
    # 3: Adjust Black/White, 4: Convert to 8-bit PNGs, 5: Update OpenBIS
    progress = pyqtSignal(str, int, int)
    conversion_progress = pyqtSignal(str, int, int)
    completion = pyqtSignal(int)
    final_update_done = pyqtSignal(bool)

    def __init__(self, file_path, dimensions, channels, openbis_instance, output_dir,
                 selected_project, date_str, user_str, setup_str, experiment_identifier):
        super().__init__()
        self.file_path = file_path
        self.dimensions = dimensions
        self.channels = channels
        self.openbis_instance = openbis_instance
        self.output_dir = output_dir
        self.selected_project = selected_project
        self.date_str = date_str
        self.user_str = user_str
        self.setup_str = setup_str
        self.experiment_identifier = experiment_identifier
        self.stopped = False

    def stop(self):
        self.stopped = True

    def run(self):
        try:
            is_2D = (self.dimensions.get("Z", 1) == 1)
            logging.info("Processing a {} sample.".format("2D" if is_2D else "3D"))
            
            # Step 0: Generate Metadata
            self.progress.emit("Generating metadata...", 0, 0)
            experimental_description, metadata_csv_path, tatexp_xml_path = generate_metadata(
                self.file_path, self.dimensions, self.output_dir
            )
            self.completion.emit(0)
            
            # Step 1: Extract Images
            self.progress.emit("Extracting images...", 0, 1)
            extract_images_with_nd2_plugin(self.file_path, save_dir=self.output_dir)
            self.completion.emit(1)
            
            # Step 2: Create Composite Images
            self.progress.emit("Creating composite images...", 0, 2)
            create_composite_images_for_all_channels(self.output_dir, num_channels=len(self.channels), channel_names=self.channels, dimensions=self.dimensions)
            self.completion.emit(2)
            
            # Step 3: Adjust Black/White Points
            self.progress.emit("Adjusting black and white points...", 0, 3)
            bw_points = adjust_black_white_cv2(self.output_dir)
            save_black_white_points(bw_points, os.path.join(self.output_dir, "black_white_points.json"))
            self.completion.emit(3)
            
            # Step 4: Convert to 8-bit PNGs
            self.progress.emit("Converting to 8-bit PNGs...", 0, 4)
            def my_progress_callback(msg, current, total):
                self.conversion_progress.emit(msg, current, total)
            conversion_results = process_nd2_images_multithreaded(
                self.file_path,
                self.output_dir,
                os.path.join(self.output_dir, "black_white_points.json"),
                datetime.now().strftime("%y%m%d"),
                "AK",  # Adjust as needed
                progress_callback=my_progress_callback
            )
            self.completion.emit(4)
            
            # Step 5: Update OpenBIS
            self.progress.emit("Updating OpenBIS...", 0, 5)
            json_path = os.path.join(self.output_dir, "black_white_points.json")
            main_folder_name = self.output_dir
            step_name = os.path.basename(self.file_path).replace('.nd2', '')
            composite_image_paths = [os.path.join(self.output_dir, f"{name}.png") for name in self.channels]
            results_html = generate_results_html(json_path, composite_image_paths, main_folder_name)
            create_experimental_step_with_dataset(
                openbis_instance=self.openbis_instance,
                experiment_identifier=self.experiment_identifier,
                step_name=step_name,
                file_info=experimental_description,
                metadata_csv_path=metadata_csv_path,
                composite_image_paths=composite_image_paths,
                results_html=results_html,
            )
            self.final_update_done.emit(True)
            self.completion.emit(5)
        except Exception as e:
            self.progress.emit(f"Error: {e}", 0, -1)

# --- Main Application Window ---
class ND2PipelineApp(QMainWindow):
    def __init__(self, openbis_instance=None):
        super().__init__()
        self.setWindowTitle("ND2 Pipeline Interface")
        self.openbis_instance = openbis_instance
        self.file_path = None
        self.dimensions = None
        self.channels = []
        self.output_dir = None
        self.selected_experiment_identifier = None

        # Create central widget and main layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        
        # --- Section 1: File Selection ---
        file_selection_layout = QVBoxLayout()
        self.file_label = QLabel("No file selected")
        self.select_file_btn = QPushButton("Select File")
        self.select_file_btn.clicked.connect(self.select_file)
        self.dimension_label = QLabel("Dimensions: Not loaded")
        file_selection_layout.addWidget(self.file_label)
        file_selection_layout.addWidget(self.select_file_btn)
        file_selection_layout.addWidget(self.dimension_label)
        self.layout.addLayout(file_selection_layout)
        
        # --- Section 2: Experiment Information ---
        experiment_info_layout = QVBoxLayout()
        self.explorer_btn = QPushButton("Browse OpenBIS Experiments")
        self.explorer_btn.clicked.connect(self.show_openbis_explorer)
        experiment_info_layout.addWidget(self.explorer_btn)
        
        self.date_edit = QLineEdit("Data of experiment")
        self.user_edit = QLineEdit("Initials")
        self.setup_edit = QLineEdit("Microscope Setup Number")
        experiment_info_layout.addWidget(QLabel("Date:"))
        experiment_info_layout.addWidget(self.date_edit)
        experiment_info_layout.addWidget(QLabel("User:"))
        experiment_info_layout.addWidget(self.user_edit)
        experiment_info_layout.addWidget(QLabel("Setup:"))
        experiment_info_layout.addWidget(self.setup_edit)
        
        self.project_combo = QComboBox()
        experiment_info_layout.addWidget(QLabel("Select Project:"))
        experiment_info_layout.addWidget(self.project_combo)
        if self.openbis_instance is not None:
            self.populate_projects()
        self.layout.addLayout(experiment_info_layout)
        
        # --- Section 3: Channel Entries ---
        self.channel_layout = QGridLayout()
        self.channel_entries = []
        self.layout.addLayout(self.channel_layout)
        
        # --- Section 4: Control Section (Start/Stop & Step Checkboxes) ---
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Pipeline")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_pipeline)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_pipeline)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        self.layout.addLayout(control_layout)
        
        self.step_checkboxes = []
        for step in [
            "Generate Metadata", "Extract Images", "Create Composite",
            "Adjust Black/White", "Convert to 8-bit PNG", "Update OpenBIS"
        ]:
            checkbox = QCheckBox(step)
            checkbox.setEnabled(False)
            self.step_checkboxes.append(checkbox)
            self.layout.addWidget(checkbox)
        
        # --- Section 5: Progress ---
        self.progress_label = QLabel("Progress: Waiting...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)
        
        # --- Section 6: Logging Output ---
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMaximumHeight(200)
        self.layout.addWidget(self.log_widget)
        
        self.setCentralWidget(self.central_widget)
        self.setup_logging()
    
    def setup_logging(self):
        qt_handler = QtHandler()
        qt_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        qt_handler.setFormatter(formatter)
        qt_handler.log_signal.connect(self.append_log)
        logging.getLogger().addHandler(qt_handler)
    
    def append_log(self, msg):
        self.log_widget.appendPlainText(msg)
    
    def populate_projects(self):
        try:
            all_projects = self.openbis_instance.get_projects()
            # Retrieve the username from the openBIS instance and convert it to uppercase.
            login_name = (self.openbis_instance.username if hasattr(self.openbis_instance, "username") else "").upper()
            logging.info(f"Using login name: {login_name}")
            # Filter projects where the space code matches the login name.
            filtered_projects = [p for p in all_projects if p.space.code.upper() == login_name]
            logging.info(f"Found {len(filtered_projects)} projects in the {login_name} space.")
            for p in filtered_projects:
                proj_identifier = f"{p.space.code}/{p.code}"
                self.project_combo.addItem(proj_identifier)
        except Exception as e:
            logging.error(f"Failed to fetch projects: {e}")
    
    def show_openbis_explorer(self):
        if not self.openbis_instance:
            QMessageBox.critical(self, "Error", "No OpenBIS instance available.")
            return
        explorer = OpenBISExplorerDialog(self.openbis_instance, parent=self)
        if explorer.exec_() == QDialog.Accepted:
            selected_exp = explorer.get_selected_experiment_identifier()
            if selected_exp:
                QMessageBox.information(self, "Selection", f"Selected: {selected_exp}")
                self.selected_experiment_identifier = selected_exp
            else:
                QMessageBox.warning(self, "Warning", "No experiment selected.")
    
    def select_file(self):
        try:
            self.file_path, self.dimensions, self.output_dir = select_file()
            self.file_label.setText(self.file_path)
            self.dimension_label.setText(f"Dimensions: {self.dimensions}")
            num_channels = self.dimensions.get("C", 1)
            self.channels = [f"Channel {i+1}" for i in range(num_channels)]
            for i in range(self.channel_layout.count()):
                self.channel_layout.itemAt(i).widget().deleteLater()
            self.channel_entries.clear()
            for i, channel in enumerate(self.channels):
                label = QLabel(f"Channel {i+1}:")
                entry = QLineEdit()
                entry.setText(channel)
                self.channel_layout.addWidget(label, i, 0)
                self.channel_layout.addWidget(entry, i, 1)
                self.channel_entries.append(entry)
            file_base = os.path.splitext(os.path.basename(self.file_path))[0]
            date_str = file_base[:6]
            user_str = file_base[6:8]
            setup_str = file_base[8:10]
            self.date_edit.setText(date_str)
            self.user_edit.setText(user_str)
            self.setup_edit.setText(setup_str)
            self.start_btn.setEnabled(True)
        except Exception as e:
            self.file_label.setText(f"Error selecting file: {e}")
    
    def start_pipeline(self):
        if not self.openbis_instance:
            QMessageBox.critical(self, "Error", "You must log in to OpenBIS first!")
            return
        self.channels = [entry.text() for entry in self.channel_entries]
        date_str = self.date_edit.text()
        user_str = self.user_edit.text()
        setup_str = self.setup_edit.text()
        selected_project = self.project_combo.currentText()
        if self.selected_experiment_identifier is None:
            file_base = os.path.splitext(os.path.basename(self.file_path))[0]
            experiment_identifier = f"{selected_project}/{file_base}"
        else:
            experiment_identifier = self.selected_experiment_identifier
        logging.info(f"Using experiment identifier: {experiment_identifier}")
        for checkbox in self.step_checkboxes:
            checkbox.setChecked(False)
        self.worker = WorkerThread(
            self.file_path,
            self.dimensions,
            self.channels,
            self.openbis_instance,
            self.output_dir,
            selected_project,
            date_str,
            user_str,
            setup_str,
            experiment_identifier
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.completion.connect(self.mark_step_completed)
        self.worker.conversion_progress.connect(self.update_conversion_progress)
        self.worker.final_update_done.connect(self.final_update_complete)
        self.worker.start()
        self.stop_btn.setEnabled(True)
    
    def stop_pipeline(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.stop_btn.setEnabled(False)
            self.progress_label.setText("Pipeline stopped by user.")
            self.progress_bar.setValue(0)
    
    def final_update_complete(self, success):
        if success:
            self.progress_label.setText("OpenBIS update completed successfully!")
            self.mark_step_completed(5)
        else:
            self.progress_label.setText("OpenBIS update failed. Check logs.")
        self.stop_btn.setEnabled(False)
    
    def update_progress(self, message, progress, step_index):
        self.progress_label.setText(f"Progress: {message}")
    
    def update_conversion_progress(self, message, current, total):
        self.progress_label.setText(message)
        progress_percentage = int((current / total) * 100)
        self.progress_bar.setValue(progress_percentage)
    
    def mark_step_completed(self, step_index):
        if 0 <= step_index < len(self.step_checkboxes):
            self.step_checkboxes[step_index].setChecked(True)
            if all(cb.isChecked() for cb in self.step_checkboxes):
                self.stop_btn.setEnabled(False)

def main():
    app = QApplication(sys.argv)
    openbis_host = "https://openbis-csd.ethz.ch"  # Replace with your actual host URL
    login_dialog = OpenBISLoginDialog(openbis_host)
    login_dialog.exec_()
    if not login_dialog.openbis_instance:
        logging.error("OpenBIS login failed or canceled. Exiting.")
        sys.exit(1)
    window = ND2PipelineApp(openbis_instance=login_dialog.openbis_instance)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
