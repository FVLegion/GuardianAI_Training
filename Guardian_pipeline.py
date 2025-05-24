from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
import os
import pathlib
import logging
import shutil
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

@PipelineDecorator.component(return_values=["dataset_path"], cache=True, execution_queue="default")
def download_and_verify_clearml_dataset(
    dataset_name: str,
    dataset_project: str,
    local_target_path: str
) -> str | None:
    local_path_obj = pathlib.Path(local_target_path).resolve()
    comp_logger = logging.getLogger(f"Component.{download_and_verify_clearml_dataset.__name__}")
    
    try:
        # Create target directory if it doesn't exist
        local_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Get dataset from ClearML
        remote_dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True
        )
        
        if not remote_dataset:
            comp_logger.error(f"ClearML dataset '{dataset_name}' not found in project '{dataset_project}'")
            return None

        # Check if local dataset exists
        if local_path_obj.exists() and local_path_obj.is_dir():
            local_files = set(f.name for f in local_path_obj.rglob('*') if f.is_file())
            remote_files = set(f.split('/')[-1] for f in remote_dataset.list_files())
            
            # If local and remote files match, use local dataset
            if local_files == remote_files:
                comp_logger.info("Local dataset matches remote dataset. Using local files.")
                return str(local_path_obj)
            else:
                comp_logger.info("Local dataset differs from remote. Uploading local dataset...")
                # Create new dataset version with local files
                new_dataset = Dataset.create(
                    dataset_name=dataset_name,
                    dataset_project=dataset_project
                )
                new_dataset.add_files(local_path_obj)
                new_dataset.upload()
                new_dataset.finalize()
                comp_logger.info("Local dataset uploaded as new version.")
                return str(local_path_obj)
        
        # Download dataset if no local copy exists
        comp_logger.info("Downloading dataset from ClearML...")
        temp_download_path_str = remote_dataset.get_local_copy()
        if not temp_download_path_str:
            comp_logger.error("Failed to get local copy of dataset")
            return None
            
        temp_download_path = pathlib.Path(temp_download_path_str).resolve()
        if not temp_download_path.exists() or not temp_download_path.is_dir():
            comp_logger.error("Invalid temporary download path")
            return None

        # Move/copy files to target location
        moved_items_count = 0
        copied_items_count = 0
        for item_name in os.listdir(temp_download_path):
            source_item_path = temp_download_path / item_name
            destination_item_path = local_path_obj / item_name
            
            if destination_item_path.exists():
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

        # Cleanup temporary directory
        shutil.rmtree(temp_download_path)
        comp_logger.info(f"Dataset downloaded successfully with {moved_items_count} directories and {copied_items_count} files")
        
        return str(local_path_obj) if local_path_obj.exists() and local_path_obj.is_dir() else None

    except Exception as e:
        comp_logger.error(f"Error in download_and_verify_clearml_dataset: {e}", exc_info=True)
        return None

@PipelineDecorator.component(return_values=["dataset_path", "input_size", "num_classes"])
def prepare_data(dataset_path: str):
    comp_logger = logging.getLogger(f"Component.{prepare_data.__name__}")
    
    try:
        from torch.utils.data import DataLoader, Dataset
        from sklearn.model_selection import train_test_split
        import numpy as np
        import json
        import os
        import torch

        # Embedded dataset class
        class PoseDataset(Dataset):
            def __init__(self, data_dir, action_classes, max_frames=40):
                self.data_dir = data_dir
                self.action_classes = action_classes
                self.max_frames = max_frames  # Maximum number of frames per clip
                self.data, self.labels = self.load_data()

            def load_data(self):
                data = []
                labels = []
                for i, action in enumerate(self.action_classes):
                    action_dir = os.path.join(self.data_dir, action)
                    if not os.path.exists(action_dir):
                        print(f"Warning: Directory not found: {action_dir}")  # Debugging
                        continue

                    for filename in os.listdir(action_dir):
                        if filename.endswith("_keypoints.json"):
                            filepath = os.path.join(action_dir, filename)
                            try:
                                with open(filepath, 'r') as f:
                                    keypoints_data = json.load(f)
                                    # Extract keypoints and normalize

                                    normalized_keypoints = self.process_keypoints(keypoints_data)
                                    if normalized_keypoints is not None:
                                        data.append(normalized_keypoints)
                                        labels.append(i)  # Use index as label
                            except (json.JSONDecodeError, FileNotFoundError) as e:
                                print(f"Error loading or processing {filepath}: {e}")
                                continue  # Skip to the next file

                return data, labels

            def process_keypoints(self, keypoints_data):
                all_frames_keypoints = []
                previous_frame = None  # For temporal smoothing
                alpha = 0.8  # Smoothing factor for EMA

                for frame_data in keypoints_data:
                    if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                        print(f"Skipping invalid frame data: {frame_data}")  # Debugging
                        continue  # Skip malformed data

                    frame_keypoints = frame_data['keypoints']
                    if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                        print("frame keypoints is not a list or is empty")
                        continue

                    frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)  # First person, (17, 3)
                    if frame_keypoints_np.shape != (17, 3):
                        print(f"Incorrect shape: {frame_keypoints_np.shape}")
                        continue

                    # Filter out keypoints with low confidence
                    valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                    if valid_keypoints.size == 0:
                        continue

                    # Z-Score Normalization
                    mean_x = np.mean(valid_keypoints[:, 0])
                    std_x = np.std(valid_keypoints[:, 0]) + 1e-8  # Avoid division by zero
                    mean_y = np.mean(valid_keypoints[:, 1])
                    std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                    normalized_frame_keypoints = frame_keypoints_np.copy()
                    normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                    normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                    # Temporal Smoothing using EMA
                    if previous_frame is not None:
                        normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                        normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                    previous_frame = normalized_frame_keypoints  # Update for the next iteration

                    # Flatten and remove confidence scores
                    normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
                    all_frames_keypoints.append(normalized_frame_keypoints)

                # Padding (or truncating)
                if not all_frames_keypoints:
                    return None
                padded_keypoints = np.zeros((self.max_frames, all_frames_keypoints[0].shape[0]))
                for i, frame_kps in enumerate(all_frames_keypoints):
                    if i < self.max_frames:
                        padded_keypoints[i, :] = frame_kps

                return padded_keypoints

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

        action_classes = ["Falling", "No Action", "Waving"]
        dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
        
        if not dataset.data or not dataset.labels:
            comp_logger.error("No data or labels loaded by PoseDataset")
            return None, 0, 0

        # Just verify data can be split, but don't return DataLoaders
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            dataset.data, dataset.labels, test_size=0.2, random_state=42,
            stratify=dataset.labels if len(set(dataset.labels)) > 1 else None
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, test_size=0.25, random_state=42,
            stratify=train_val_labels if len(set(train_val_labels)) > 1 else None
        )

        input_features_per_frame = 34
        num_classes_val = len(action_classes)

        comp_logger.info(f"Data preparation completed: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        return dataset_path, input_features_per_frame, num_classes_val

    except Exception as e:
        comp_logger.error(f"Error in prepare_data: {e}", exc_info=True)
        return None, 0, 0

@PipelineDecorator.component(
    name="Train_BiLSTM",
    return_values=["task_id", "model_id"],
    packages=["torch>=1.9", "clearml", "scikit-learn", "numpy"],
    task_type=Task.TaskTypes.training,
    cache=False
)
def train_bilstm(
    dataset_path: str,
    input_size: int = 34,
    num_classes: int = 3,
    base_lr: float = 0.001,
    epochs: int = 50,
    hidden_size: int = 256,
    num_layers: int = 4,
    dropout_rate: float = 0.1
):
    """Train a BiLSTM model and return task ID and model ID."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from clearml import Task, OutputModel
    import numpy as np
    import json
    import os
    
    # Embedded model classes
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)  # hidden_size * 2 for BiLSTM

        def forward(self, lstm_output):
            # lstm_output: (batch_size, seq_length, hidden_size * 2)
            scores = self.attention_weights(lstm_output)  # (batch_size, seq_length, 1)
            attention_weights = torch.softmax(scores, dim=1)  # Softmax over the sequence length
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # Weighted sum of the outputs
            return context_vector, attention_weights.squeeze(-1)

    class ActionRecognitionBiLSTMWithAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
            super(ActionRecognitionBiLSTMWithAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = True
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout_rate, bidirectional=self.bidirectional)
            self.attention = AttentionLayer(hidden_size)
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, hidden_size * 2)
            out = self.dropout(out)

            context_vector, attention_weights = self.attention(out)

            # Decode the context vector from the attention layer
            out = self.fc(context_vector)
            return out, attention_weights

    # Embedded dataset class
    class PoseDataset(Dataset):
        def __init__(self, data_dir, action_classes, max_frames=40):
            self.data_dir = data_dir
            self.action_classes = action_classes
            self.max_frames = max_frames  # Maximum number of frames per clip
            self.data, self.labels = self.load_data()

        def load_data(self):
            data = []
            labels = []
            for i, action in enumerate(self.action_classes):
                action_dir = os.path.join(self.data_dir, action)
                if not os.path.exists(action_dir):
                    print(f"Warning: Directory not found: {action_dir}")  # Debugging
                    continue

                for filename in os.listdir(action_dir):
                    if filename.endswith("_keypoints.json"):
                        filepath = os.path.join(action_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                keypoints_data = json.load(f)
                                # Extract keypoints and normalize

                                normalized_keypoints = self.process_keypoints(keypoints_data)
                                if normalized_keypoints is not None:
                                    data.append(normalized_keypoints)
                                    labels.append(i)  # Use index as label
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"Error loading or processing {filepath}: {e}")
                            continue  # Skip to the next file

            return data, labels

        def process_keypoints(self, keypoints_data):
            all_frames_keypoints = []
            previous_frame = None  # For temporal smoothing
            alpha = 0.8  # Smoothing factor for EMA

            for frame_data in keypoints_data:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    print(f"Skipping invalid frame data: {frame_data}")  # Debugging
                    continue  # Skip malformed data

                frame_keypoints = frame_data['keypoints']
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                    print("frame keypoints is not a list or is empty")
                    continue

                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)  # First person, (17, 3)
                if frame_keypoints_np.shape != (17, 3):
                    print(f"Incorrect shape: {frame_keypoints_np.shape}")
                    continue

                # Filter out keypoints with low confidence
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                if valid_keypoints.size == 0:
                    continue

                # Z-Score Normalization
                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8  # Avoid division by zero
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                normalized_frame_keypoints = frame_keypoints_np.copy()
                normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                # Temporal Smoothing using EMA
                if previous_frame is not None:
                    normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                    normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                previous_frame = normalized_frame_keypoints  # Update for the next iteration

                # Flatten and remove confidence scores
                normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
                all_frames_keypoints.append(normalized_frame_keypoints)

            # Padding (or truncating)
            if not all_frames_keypoints:
                return None
            padded_keypoints = np.zeros((self.max_frames, all_frames_keypoints[0].shape[0]))
            for i, frame_kps in enumerate(all_frames_keypoints):
                if i < self.max_frames:
                    padded_keypoints[i, :] = frame_kps

            return padded_keypoints

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    # Initialize task
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="Guardian_Training",
            task_name="Train_BiLSTM"
        )
    
    logger = task.get_logger()
    
    # Connect hyperparameters
    task.connect({
        'General/base_lr': base_lr,
        'General/epochs': epochs,
        'General/hidden_size': hidden_size,
        'General/num_layers': num_layers,
        'General/dropout_rate': dropout_rate
    })

    # Recreate DataLoaders from dataset path
    action_classes = ["Falling", "No Action", "Waving"]
    dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
    
    if not dataset.data or not dataset.labels:
        raise RuntimeError("No data or labels loaded by PoseDataset")

    # Split data into train, validation, and test sets
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42,
        stratify=dataset.labels if len(set(dataset.labels)) > 1 else None
    )
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=0.25, random_state=42,
        stratify=train_val_labels if len(set(train_val_labels)) > 1 else None
    )

    def make_torch_dataset_for_loader(split_data, split_labels):
        temp_ds = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds

    # Create data loaders
    train_loader = DataLoader(make_torch_dataset_for_loader(train_data, train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(make_torch_dataset_for_loader(val_data, val_labels), batch_size=32, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_path = f"best_model_{task.id}.pt"

    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc = 100 * correct / max(total, 1)
        
        # Log metrics
        logger.report_scalar("Loss", "Train", train_loss / len(train_loader), epoch)
        logger.report_scalar("Loss", "Validation", val_loss / len(val_loader), epoch)
        logger.report_scalar("Accuracy", "Validation", val_acc, epoch)
        
        # Also log with different format for HPO compatibility
        logger.report_scalar("metrics", "Validation_Accuracy", val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Publish model
    output_model = OutputModel(task=task, name="BiLSTM_ActionRecognition", framework="PyTorch")
    output_model.update_weights(weights_filename=best_model_path)
    output_model.publish()
    
    return task.id, output_model.id

@PipelineDecorator.component(
    name="BiLSTM_HPO",
    return_values=["best_task_id", "best_model_id"],
    cache=False,
    packages=["clearml"]
)
def bilstm_hyperparam_optimizer(
    base_task_id: str,
    dataset_path: str,
    input_size: int,
    num_classes: int,
    total_max_trials: int = 5
):
    """Run GridSearch HPO on the Train_BiLSTM component."""
    from clearml import Task
    from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange
    from clearml.automation import GridSearch
    from clearml import Model
    
    # Initialize HPO task
    hpo_task = Task.init(
        project_name="Guardian_Training",
        task_name="BiLSTM_GridSearch_Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    # Define search space with discrete values for GridSearch
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            DiscreteParameterRange('General/base_lr', values=[0.001, 0.003, 0.01]),
            DiscreteParameterRange('General/hidden_size', values=[128, 256]),
            DiscreteParameterRange('General/num_layers', values=[2, 3]),
            DiscreteParameterRange('General/dropout_rate', values=[0.1, 0.3, 0.5]),
            DiscreteParameterRange('General/epochs', values=[20, 30])
        ],
        # Objective metric we want to maximize
        objective_metric_title="Accuracy",
        objective_metric_series="Validation",
        objective_metric_sign="max",
        # Limit concurrent experiments
        max_number_of_concurrent_tasks=1,
        # Use GridSearch instead of Optuna
        optimizer_class=GridSearch,
        # Keep only top performing tasks
        save_top_k_tasks_only=3,
        # Fixed arguments passed to each training task
        base_task_kwargs={
            'dataset_path': dataset_path,
            'input_size': input_size,
            'num_classes': num_classes
        },
        compute_time_limit=None,
        total_max_jobs=total_max_trials,
        min_iteration_per_job=None,
        max_iteration_per_job=None,
    )
    
    print("Starting GridSearch optimization...")
    print(f"Total combinations to try: {3 * 2 * 2 * 3 * 2} = {3 * 2 * 2 * 3 * 2}")
    print("This will be limited by total_max_jobs parameter")
    
    # Start optimization
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()
    
    print("GridSearch optimization completed!")
    
    # Get best experiment
    top_exps = optimizer.get_top_experiments(top_k=1)
    if not top_exps:
        print("No experiments found!")
        raise RuntimeError("No HPO experiments returned by optimizer.")
    
    best_exp = top_exps[0]
    best_exp_id = best_exp.id
    
    # Try to get validation accuracy from different possible metric names
    metrics = best_exp.get_last_scalar_metrics()
    best_acc = None
    
    # Try different metric paths
    metric_paths = [
        ("Accuracy", "Validation"),
        ("metrics", "Validation_Accuracy"),
        ("Loss", "Validation"),  # Fallback to validation loss
    ]
    
    for title, series in metric_paths:
        try:
            if title in metrics and series in metrics[title]:
                best_acc = metrics[title][series].get("last")
                if best_acc is not None:
                    print(f"Found metric {title}/{series}: {best_acc}")
                    break
        except (KeyError, AttributeError, TypeError):
            continue
    
    if best_acc is None:
        print("Warning: Could not retrieve validation accuracy from metrics")
        print(f"Available metrics: {list(metrics.keys()) if metrics else 'None'}")
        best_acc = "Unknown"
    
    print(f"Best experiment ID: {best_exp_id}, Validation Accuracy={best_acc}")
    
    # Get the best model
    best_exp_task = Task.get_task(task_id=best_exp_id)
    
    # Check if task has models
    if (best_exp_task.models and 
        'output' in best_exp_task.models and 
        len(best_exp_task.models['output']) > 0):
        print(f"Found model: {best_exp_task.models['output'][0]}")
        # Get the model object
        model = best_exp_task.models['output'][0]
        model_id = model.id
        print(f"Best model ID: {model_id}")
        return best_exp_id, model_id
    else:
        print("No models found in the best task, searching for published models...")
        # Fallback: search for published models
        models = Model.query_models(
            project_name="Guardian_Training",
            model_name="BiLSTM_ActionRecognition",
            only_published=True,
            max_results=10,
            order_by=['-created']
        )
        
        if models and len(models) > 0:
            best_model = models[0]  # Most recent model
            print(f"Using fallback model ID: {best_model.id}")
            return best_exp_id, best_model.id
        else:
            print("No published models found either, checking all models...")
            # Second fallback: search for any models (published or not)
            all_models = Model.query_models(
                project_name="Guardian_Training",
                model_name="BiLSTM_ActionRecognition",
                only_published=False,
                max_results=10,
                order_by=['-created']
            )
            
            if all_models and len(all_models) > 0:
                best_model = all_models[0]
                print(f"Using any available model ID: {best_model.id}")
                return best_exp_id, best_model.id
            else:
                print("ERROR: No models found anywhere!")
                # Return the task ID with a dummy model ID - evaluation will handle this
                return best_exp_id, "no_model_found"

@PipelineDecorator.component(
    name="Evaluate_Model",
    return_values=["test_accuracy"],
    cache=False,
    packages=["torch", "scikit-learn", "numpy", "clearml"]
)
def evaluate_model(
    best_task_id: str,
    best_model_id: str,
    dataset_path: str,
    input_size: int = 34,
    num_classes: int = 3
):
    """Evaluate the best BiLSTM model on the test set."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from clearml import Task, Model
    import numpy as np
    import json
    import os
    
    # Embedded model classes
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)  # hidden_size * 2 for BiLSTM

        def forward(self, lstm_output):
            # lstm_output: (batch_size, seq_length, hidden_size * 2)
            scores = self.attention_weights(lstm_output)  # (batch_size, seq_length, 1)
            attention_weights = torch.softmax(scores, dim=1)  # Softmax over the sequence length
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # Weighted sum of the outputs
            return context_vector, attention_weights.squeeze(-1)

    class ActionRecognitionBiLSTMWithAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
            super(ActionRecognitionBiLSTMWithAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = True
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout_rate, bidirectional=self.bidirectional)
            self.attention = AttentionLayer(hidden_size)
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 for bidirectional
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))  # (batch_size, seq_length, hidden_size * 2)
            out = self.dropout(out)

            context_vector, attention_weights = self.attention(out)

            # Decode the context vector from the attention layer
            out = self.fc(context_vector)
            return out, attention_weights

    # Embedded dataset class
    class PoseDataset(Dataset):
        def __init__(self, data_dir, action_classes, max_frames=40):
            self.data_dir = data_dir
            self.action_classes = action_classes
            self.max_frames = max_frames  # Maximum number of frames per clip
            self.data, self.labels = self.load_data()

        def load_data(self):
            data = []
            labels = []
            for i, action in enumerate(self.action_classes):
                action_dir = os.path.join(self.data_dir, action)
                if not os.path.exists(action_dir):
                    print(f"Warning: Directory not found: {action_dir}")  # Debugging
                    continue

                for filename in os.listdir(action_dir):
                    if filename.endswith("_keypoints.json"):
                        filepath = os.path.join(action_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                keypoints_data = json.load(f)
                                # Extract keypoints and normalize

                                normalized_keypoints = self.process_keypoints(keypoints_data)
                                if normalized_keypoints is not None:
                                    data.append(normalized_keypoints)
                                    labels.append(i)  # Use index as label
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"Error loading or processing {filepath}: {e}")
                            continue  # Skip to the next file

            return data, labels

        def process_keypoints(self, keypoints_data):
            all_frames_keypoints = []
            previous_frame = None  # For temporal smoothing
            alpha = 0.8  # Smoothing factor for EMA

            for frame_data in keypoints_data:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    print(f"Skipping invalid frame data: {frame_data}")  # Debugging
                    continue  # Skip malformed data

                frame_keypoints = frame_data['keypoints']
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                    print("frame keypoints is not a list or is empty")
                    continue

                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)  # First person, (17, 3)
                if frame_keypoints_np.shape != (17, 3):
                    print(f"Incorrect shape: {frame_keypoints_np.shape}")
                    continue

                # Filter out keypoints with low confidence
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                if valid_keypoints.size == 0:
                    continue

                # Z-Score Normalization
                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8  # Avoid division by zero
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                normalized_frame_keypoints = frame_keypoints_np.copy()
                normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                # Temporal Smoothing using EMA
                if previous_frame is not None:
                    normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                    normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                previous_frame = normalized_frame_keypoints  # Update for the next iteration

                # Flatten and remove confidence scores
                normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
                all_frames_keypoints.append(normalized_frame_keypoints)

            # Padding (or truncating)
            if not all_frames_keypoints:
                return None
            padded_keypoints = np.zeros((self.max_frames, all_frames_keypoints[0].shape[0]))
            for i, frame_kps in enumerate(all_frames_keypoints):
                if i < self.max_frames:
                    padded_keypoints[i, :] = frame_kps

            return padded_keypoints

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    # Initialize task
    task = Task.init(
        project_name="Guardian_Training",
        task_name="Evaluate_BiLSTM"
    )
    
    logger = task.get_logger()
    
    # Recreate test DataLoader from dataset path
    action_classes = ["Falling", "No Action", "Waving"]
    dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
    
    if not dataset.data or not dataset.labels:
        raise RuntimeError("No data or labels loaded by PoseDataset")

    # Split data to get test set (same split as training)
    _, test_data, _, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42,
        stratify=dataset.labels if len(set(dataset.labels)) > 1 else None
    )

    def make_torch_dataset_for_loader(split_data, split_labels):
        temp_ds = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds

    test_loader = DataLoader(make_torch_dataset_for_loader(test_data, test_labels), batch_size=32, shuffle=False)
    
    # Get parameters from best task
    best_task = Task.get_task(task_id=best_task_id)
    params = best_task.get_parameters_as_dict(flatten=True)
    hidden_size = int(params.get('General/hidden_size', 256))
    num_layers = int(params.get('General/num_layers', 4))
    dropout_rate = float(params.get('General/dropout_rate', 0.1))

    # Get model
    if best_model_id == "no_model_found":
        print("No model found from HPO, skipping evaluation")
        return 0.0
    
    try:
        model_obj = Model(model_id=best_model_id)
        model_path = model_obj.get_local_copy()
    except Exception as e:
        print(f"Error retrieving model {best_model_id}: {e}")
        print("Skipping evaluation due to model retrieval failure")
        return 0.0

    # Load model
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Evaluate
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            outputs, _ = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())
    
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Log results
    logger.report_scalar("Accuracy", "Test", test_accuracy, 0)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=action_classes)
    logger.report_text(report, print_console=True)
    
    return test_accuracy

@PipelineDecorator.pipeline(
    name="Guardian_Pipeline",
    project="Guardian_Training"
)
def guardian_training_pipeline():
    """Complete Guardian AI training pipeline with HPO."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Guardian Pipeline started...")
    
    # Setup paths
    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    script_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path(".").resolve()
    dataset_path = script_dir / "data" / dataset_name

    # Step 1: Download and verify dataset
    logging.info("Starting dataset download and verification...")
    dataset_path_output = download_and_verify_clearml_dataset(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        local_target_path=str(dataset_path)
    )
    if not dataset_path_output:
        raise ValueError("Dataset step failed.")
    logging.info(f"Dataset step completed. Using path: {dataset_path_output}")

    # Step 2: Prepare data (just for getting input_size and num_classes)
    logging.info("Starting data preparation step...")
    dataset_path, input_size, num_classes = prepare_data(
        dataset_path=dataset_path_output
    )
    if not dataset_path:
        raise RuntimeError("Data preparation step failed.")
    logging.info("Data preparation step completed.")

    # Step 3: Train baseline model
    logging.info("Starting model training step...")
    base_task_id, _ = train_bilstm(
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes
    )
    if not base_task_id:
        raise RuntimeError("Training step failed.")
    logging.info(f"Training step completed. Base task ID: {base_task_id}")

    # Step 4: Hyperparameter optimization
    logging.info("Starting hyperparameter optimization...")
    best_task_id, best_model_id = bilstm_hyperparam_optimizer(
        base_task_id=base_task_id,
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes,
        total_max_trials=5
    )
    logging.info(f"HPO completed. Best task ID: {best_task_id}, Best model ID: {best_model_id}")

    # Step 5: Evaluate best model
    logging.info("Starting model evaluation step...")
    test_accuracy = evaluate_model(
        best_task_id=best_task_id,
        best_model_id=best_model_id,
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes
    )
    logging.info(f"Evaluation step completed. Test accuracy: {test_accuracy:.2f}%")

    logging.info("Guardian Pipeline finished successfully.")
    return test_accuracy

if __name__ == '__main__':
    logging.info("Running Guardian 🦾 pipeline locally...")
    
    # Initialize ClearML task
    task = Task.init(project_name="Guardian_Training", task_name="Pipeline Controller")
    
    # run_locally() executes the pipeline defined by decorators in the current process
    # This allows it to appear in PIPELINES section while running locally
    PipelineDecorator.run_locally()
    
    # Start the pipeline execution
    result = guardian_training_pipeline()
    
    logging.info("Local pipeline execution complete.")
    if result is not None:
        logging.info(f"Pipeline completed successfully. Test accuracy: {result:.2f}%")
    else:
        logging.error("Pipeline execution failed")
