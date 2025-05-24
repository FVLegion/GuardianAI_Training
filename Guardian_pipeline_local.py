from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
import os
import pathlib
import logging
import shutil
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Import all components from the main pipeline file
from Guardian_pipeline import (
    download_and_verify_clearml_dataset,
    prepare_data,
    train_bilstm,
    bilstm_hyperparam_optimizer,
    evaluate_model
)

@PipelineDecorator.pipeline(
    name='Guardian Training Pipeline (Local)',
    project='Guardian_Training',
    version='1.0.0-local'
)
def guardian_training_pipeline_local():
    """Local version of the Guardian AI training pipeline for testing."""
    pipe_logger = logging.getLogger(f"Pipeline.{guardian_training_pipeline_local.__name__}")
    
    # Setup paths
    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    script_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path(".").resolve()
    dataset_path = script_dir / "data" / dataset_name

    # Step 1: Download and verify dataset
    dataset_path_output = download_and_verify_clearml_dataset(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        local_target_path=str(dataset_path)
    )
    if not dataset_path_output:
        pipe_logger.error("Failed to download dataset")
        return None

    # Step 2: Prepare data
    train_loader, val_loader, test_loader, input_size, num_classes = prepare_data(
        dataset_path=dataset_path_output
    )
    if train_loader is None:
        pipe_logger.error("Failed to prepare data")
        return None

    # Step 3: Train baseline model (reduced epochs for testing)
    base_task_id, base_model_id = train_bilstm(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        num_classes=num_classes,
        epochs=5  # Reduced for local testing
    )

    # Step 4: Skip HPO for local testing, use the baseline model
    pipe_logger.info("Skipping HPO for local testing, using baseline model")
    best_task_id, best_model_id = base_task_id, base_model_id

    # Step 5: Evaluate model
    test_accuracy = evaluate_model(
        best_task_id=best_task_id,
        best_model_id=best_model_id,
        test_loader=test_loader,
        input_size=input_size,
        num_classes=num_classes
    )

    pipe_logger.info(f"Local pipeline completed successfully. Final test accuracy: {test_accuracy:.2f}%")
    return test_accuracy

if __name__ == '__main__':
    # Initialize ClearML task
    task = Task.init(project_name="Guardian_Training", task_name="Pipeline Controller (Local)")
    
    # Run pipeline locally for debugging
    PipelineDecorator.run_locally()
    
    # Execute the pipeline
    result = guardian_training_pipeline_local()
    
    if result is not None:
        logging.info(f"Local pipeline completed successfully. Test accuracy: {result:.2f}%")
    else:
        logging.error("Local pipeline execution failed") 