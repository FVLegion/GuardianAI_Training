from clearml import PipelineDecorator, Dataset, Task
import os
import pathlib
import logging
import shutil # Added for file operations
import sys # Added for sys.path modification

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

@PipelineDecorator.component(return_values=["dataset_path"], cache=True, execution_queue="default")
def download_and_verify_clearml_dataset(
    dataset_name: str,
    dataset_project: str,
    local_target_path: str
) -> str | None:
    """
    Downloads a ClearML dataset to a specified local path, ensuring it's present and up-to-date.
    Uses dataset.get_local_copy() with no arguments and then moves/copies contents.
    """
    local_path_obj = pathlib.Path(local_target_path).resolve()
    comp_logger = logging.getLogger(f"Component.{download_and_verify_clearml_dataset.__name__}")
    comp_logger.info(f"Component execution started.")
    comp_logger.info(f"Requested local target path: '{local_target_path}', Resolved to absolute path: '{local_path_obj}'")

    dataset_files_metadata = []

    try:
        comp_logger.info(f"Ensuring local target directory exists: '{local_path_obj}'")
        local_path_obj.mkdir(parents=True, exist_ok=True)
        comp_logger.info(f"Local target directory '{local_path_obj}' ensured to exist.")

        comp_logger.info(f"Attempting to get ClearML dataset metadata for '{dataset_name}' from project '{dataset_project}'.")
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True
        )

        if not dataset:
            comp_logger.error(f"ClearML dataset '{dataset_name}' not found in project '{dataset_project}' or no completed version available.")
            return None
        comp_logger.info(f"Successfully retrieved ClearML dataset metadata: ID='{dataset.id}', Name='{dataset.name}', Version='{dataset.version}'")
        try: 
            dataset_files_metadata = dataset.list_files()
            comp_logger.info(f"Dataset '{dataset.id}' metadata lists {len(dataset_files_metadata)} files. First few: {dataset_files_metadata[:5]}")
        except Exception as e_list_meta: 
            comp_logger.warning(f"Could not list files from dataset metadata: {e_list_meta}")

        comp_logger.info(f"Attempting to get local copy of dataset '{dataset.id}' using default SDK mechanism (no arguments to get_local_copy).")
        temp_download_path_str = dataset.get_local_copy() 

        if not temp_download_path_str:
            comp_logger.error(f"dataset.get_local_copy() returned None or empty path. Cannot proceed.")
            return None

        temp_download_path = pathlib.Path(temp_download_path_str).resolve()
        comp_logger.info(f"dataset.get_local_copy() returned temporary path: '{temp_download_path}'")

        if not temp_download_path.exists() or not temp_download_path.is_dir():
            comp_logger.error(f"Temporary path '{temp_download_path}' from get_local_copy() does not exist or is not a directory.")
            return None

        comp_logger.info(f"Preparing to move/copy contents from temporary path '{temp_download_path}' to target '{local_path_obj}'")
        
        moved_items_count = 0
        copied_items_count = 0
        try:
            for item_name in os.listdir(temp_download_path):
                source_item_path = temp_download_path / item_name
                destination_item_path = local_path_obj / item_name

                if destination_item_path.exists():
                    comp_logger.warning(f"Destination item '{destination_item_path}' already exists. Removing to overwrite.")
                    if destination_item_path.is_dir():
                        shutil.rmtree(destination_item_path)
                    else:
                        destination_item_path.unlink(missing_ok=True) 
                
                if source_item_path.is_dir():
                    shutil.move(str(source_item_path), str(destination_item_path))
                    moved_items_count += 1
                else: 
                    shutil.copy2(str(source_item_path), str(destination_item_path))
                    copied_items_count += 1
            comp_logger.info(f"Successfully moved {moved_items_count} directories and copied {copied_items_count} files from '{temp_download_path}' to '{local_path_obj}'.")

            comp_logger.info(f"Cleaning up temporary download directory: '{temp_download_path}'")
            shutil.rmtree(temp_download_path)
            comp_logger.info(f"Successfully cleaned up '{temp_download_path}'.")

        except Exception as e_copy:
            comp_logger.error(f"Error during copy/move from '{temp_download_path}' to '{local_path_obj}': {e_copy}", exc_info=True)
            if temp_download_path.exists(): 
                try:
                    shutil.rmtree(temp_download_path)
                    comp_logger.info(f"Cleaned up '{temp_download_path}' after copy error.")
                except Exception as e_cleanup:
                    comp_logger.error(f"Failed to cleanup '{temp_download_path}' after copy error: {e_cleanup}")
            return None

        if local_path_obj.exists() and local_path_obj.is_dir():
            comp_logger.info(f"Dataset '{dataset_name}' is now available at the target local path: '{local_path_obj}'.")
            try:
                contents = list(local_path_obj.iterdir())
                comp_logger.info(f"Contents of '{local_path_obj}' (first 10 items): {[str(p.name) for p in contents[:10]]}")
                if not contents and dataset_files_metadata:
                     comp_logger.warning(f"Warning: Dataset directory '{local_path_obj}' is empty, but metadata indicated files. Check for issues.")
            except Exception as e_list_final:
                comp_logger.warning(f"Could not list contents of final path '{local_path_obj}': {e_list_final}")
            return str(local_path_obj)
        else:
            comp_logger.error(f"Target local path '{local_path_obj}' does not exist or is not a directory after copy/move operation.")
            return None

    except Exception as e:
        comp_logger.error(f"An unexpected error occurred in download_and_verify_clearml_dataset for '{dataset_name}': {e}", exc_info=True)
        return None

@PipelineDecorator.component(return_values=["train_loader", "val_loader", "test_loader", "input_size", "num_classes"])
def prepare_data(dataset_path: str):
    # Component-specific logger
    comp_logger = logging.getLogger(f"Component.{prepare_data.__name__}")
    comp_logger.info(f"Component 'prepare_data' started with dataset_path: {dataset_path}")

    comp_logger.info(f"Current sys.path (at component start): {sys.path}")
    comp_logger.info(f"Current working directory (at component start): {os.getcwd()}")
    
    try:
        comp_logger.info("Attempting to import PoseDataset from training.dataset_utils")
        # This import relies on the project root (GuardianAI_Training) being in sys.path
        from training.dataset_utils import PoseDataset 
        comp_logger.info("Successfully imported PoseDataset from training.dataset_utils")
    except ImportError as e:
        comp_logger.error(f"Failed to import PoseDataset from training.dataset_utils: {e}.", exc_info=True)
        comp_logger.error("Ensure 'dataset_utils.py' is inside a 'training' subdirectory, "
                          "and that 'GuardianAI_Training/training/__init__.py' exists. "
                          "Also check if the project root 'GuardianAI_Training' was correctly added to sys.path by the main script.")
        raise

    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    action_classes = ["Falling", "No Action", "Waving"] 
    comp_logger.info(f"Action classes: {action_classes}")
    comp_logger.info(f"Initializing PoseDataset with data_dir: {dataset_path}")

    dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
    
    if not dataset.data or not dataset.labels: 
        comp_logger.error("No data or labels loaded by PoseDataset. Cannot create DataLoaders. "
                          "Check dataset_path, subdirectories for action_classes, and .json files.")
        return None, None, None, 0, 0 

    comp_logger.info(f"Total samples loaded by PoseDataset: {len(dataset.data)}")

    comp_logger.info("Splitting data into train, validation, and test sets.")
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42, stratify=dataset.labels if len(set(dataset.labels)) > 1 else None
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=0.25, random_state=42, stratify=train_val_labels if len(set(train_val_labels)) > 1 else None 
    )
    comp_logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    def make_torch_dataset_for_loader(split_data, split_labels, original_data_dir, action_classes_list):
        temp_ds = PoseDataset(data_dir=original_data_dir, action_classes=action_classes_list) 
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds

    comp_logger.info("Creating DataLoaders...")
    train_dataset_for_loader = make_torch_dataset_for_loader(train_data, train_labels, dataset_path, action_classes)
    val_dataset_for_loader = make_torch_dataset_for_loader(val_data, val_labels, dataset_path, action_classes)
    test_dataset_for_loader = make_torch_dataset_for_loader(test_data, test_labels, dataset_path, action_classes)

    train_loader = DataLoader(train_dataset_for_loader, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset_for_loader, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset_for_loader, batch_size=32, shuffle=False)
    comp_logger.info("DataLoaders created.")
    
    input_features_per_frame = 34 
    num_classes_val = len(action_classes)

    comp_logger.info(f"Returning: input_size={input_features_per_frame}, num_classes={num_classes_val}")
    return train_loader, val_loader, test_loader, input_features_per_frame, num_classes_val


@PipelineDecorator.pipeline(
    name='Data Ingestion and Preparation Pipeline', 
    project='Guardian_Training',
    version='0.4.2' # Incremented version
)
def data_processing_pipeline(): 
    pipe_logger = logging.getLogger(f"Pipeline.{data_processing_pipeline.__name__}")
    pipe_logger.info("Data processing pipeline function CALLED.")

    dataset_name = "Guardian_Dataset" 
    dataset_project = "Guardian_Training" 
    
    try:
        # This will be the directory where Guardian_pipeline.py is located.
        # This path will be used to construct the local_data_path.
        script_dir_for_data = pathlib.Path(__file__).resolve().parent
    except NameError: 
        # Fallback if __file__ is not defined (e.g., interactive session, though less likely for pipeline script)
        script_dir_for_data = pathlib.Path(".").resolve() 
        pipe_logger.warning(f"__file__ not defined when determining script_dir_for_data, using current working directory: {script_dir_for_data}")

    local_data_root = script_dir_for_data / "data" 
    guardian_dataset_path = local_data_root / dataset_name
    pipe_logger.info(f"Target local path for Guardian_Dataset: {guardian_dataset_path}")

    dataset_path_output = download_and_verify_clearml_dataset(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        local_target_path=str(guardian_dataset_path)
    )

    if not dataset_path_output:
        pipe_logger.error("Pipeline FAILED: download_and_verify_clearml_dataset did not return a valid path.")
        return None, None, None, 0, 0 

    pipe_logger.info(f"Dataset downloaded to: {dataset_path_output}")

    pipe_logger.info(f"Calling prepare_data component with dataset_path: {dataset_path_output}")
    train_loader, val_loader, test_loader, input_size, num_classes = prepare_data(
        dataset_path=dataset_path_output
    )

    if train_loader is None: 
        pipe_logger.error("Pipeline FAILED: prepare_data component failed to produce data loaders.")
        return None, None, None, 0, 0

    pipe_logger.info(f"Data preparation completed. Input size: {input_size}, Num classes: {num_classes}")
    pipe_logger.info(f"Train loader samples: {len(train_loader.dataset) if train_loader and hasattr(train_loader, 'dataset') else 'N/A'}, "
                     f"Val loader samples: {len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else 'N/A'}, "
                     f"Test loader samples: {len(test_loader.dataset) if test_loader and hasattr(test_loader, 'dataset') else 'N/A'}")
    
    pipe_logger.info("Data processing pipeline logic finished.")
    return train_loader, val_loader, test_loader, input_size, num_classes


if __name__ == '__main__':
    # --- Add project root to sys.path ---
    # This assumes your script Guardian_pipeline.py is in the project root (GuardianAI_Training)
    # If it's in a subdirectory, adjust accordingly.
    try:
        # Get the directory where this script (Guardian_pipeline.py) is located.
        # This should be your project root: "GuardianAI_Training"
        project_root_dir = str(pathlib.Path(__file__).resolve().parent)
        if project_root_dir not in sys.path:
            sys.path.insert(0, project_root_dir)
            logging.info(f"Added project root '{project_root_dir}' to sys.path.")
        logging.info(f"Current sys.path (after modification): {sys.path}")
    except NameError:
        # __file__ is not defined (e.g. running in an interactive environment like Jupyter)
        # In this case, assume the current working directory is the project root.
        project_root_dir = str(pathlib.Path(".").resolve())
        if project_root_dir not in sys.path:
            sys.path.insert(0, project_root_dir)
            logging.info(f"__file__ not defined. Added current working directory '{project_root_dir}' to sys.path as project root.")
        logging.info(f"Current sys.path (after modification, __file__ undefined case): {sys.path}")
    # --- End of sys.path modification ---

    try:
        logging.info("Attempting to initialize ClearML Task for the script run...")
        task = Task.init(project_name="Guardian_Training/Debug", task_name="Local Pipeline Script Runner")
        logging.info(f"ClearML Task for script run initialized successfully: ID='{task.id if task else 'Unknown'}'")
    except Exception as e:
        logging.error(f"Failed to initialize ClearML Task for script run: {e}", exc_info=True)

    logging.info("Preparing for local pipeline execution with PipelineDecorator.run_locally().")
    
    try:
        PipelineDecorator.run_locally()
        logging.info("PipelineDecorator.run_locally() setup completed.")

        logging.info("Now calling the pipeline function: data_processing_pipeline()...")
        results = data_processing_pipeline() 
        if results and results[0] is not None: 
             train_loader_res, val_loader_res, test_loader_res, input_size_res, num_classes_res = results
             logging.info(f"Pipeline function call completed.")
             logging.info(f"  Input Size: {input_size_res}, Num Classes: {num_classes_res}")
             if train_loader_res and hasattr(train_loader_res, 'dataset'):
                 logging.info(f"  TrainLoader has {len(train_loader_res.dataset)} samples, {len(train_loader_res)} batches.")
             else:
                 logging.info(f"  TrainLoader not fully available or empty.")
        else:
            logging.error("Pipeline execution did not return valid results or failed.")

    except Exception as e:
        logging.error(f"An error occurred during local pipeline execution: {e}", exc_info=True)

    logging.info("Local pipeline script execution process finished.")
