from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
import os
import pathlib
import logging
import shutil
import sys
import time
import urllib.request
import zipfile
import json
from datetime import datetime
import ssl
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Record start time
start_time = time.time()

# ============================================================================
# COMPONENT 1: DATASET MANAGEMENT FOR GITHUB ACTIONS
# ============================================================================

@PipelineDecorator.component(return_values=["dataset_path"], cache=True, execution_queue="default")
def download_and_setup_dataset(
    dataset_name: str,
    dataset_project: str,
    local_target_path: str
) -> str | None:
    """Download actual dataset from ClearML with proper error handling."""
    import pathlib
    import os
    import shutil
    import logging
    from clearml import Dataset
    
    def download_real_dataset_from_clearml(dataset_name: str, project_name: str, local_path: str) -> bool:
        """
        Check if dataset exists locally and download it if not.
        
        Args:
            dataset_name: Name of the dataset in ClearML
            project_name: Name of the project in ClearML
            local_path: Local path where the dataset should be stored
        
        Returns:
            bool: True if dataset exists or was downloaded successfully, False otherwise
        """
        print(f"\nChecking for dataset: {dataset_name}")
        
        # Check if dataset exists locally with expected structure
        if os.path.exists(local_path):
            # Check for action class directories
            expected_classes = ["Falling", "No Action", "Waving"]
            has_action_structure = any(os.path.exists(os.path.join(local_path, folder)) for folder in expected_classes)
            
            # Also check for train/valid/test structure
            train_test_folders = ["train", "valid", "test"]
            has_train_test_structure = all(os.path.exists(os.path.join(local_path, folder)) for folder in train_test_folders)
            
            if has_action_structure or has_train_test_structure:
                print(f"Dataset {dataset_name} found locally at {local_path}")
                return True
            else:
                print(f"Dataset {dataset_name} exists but missing required structure. Will download from ClearML.")
        
        # Try to get the latest version from ClearML
        try:
            print(f"Attempting to connect to ClearML...")
            dataset = Dataset.get(dataset_name=dataset_name, dataset_project=project_name, only_completed=True)
            if dataset is None:
                print(f"Dataset {dataset_name} not found in ClearML project {project_name}")
                return False
            
            print(f"Downloading dataset {dataset_name} from ClearML...")
            
            # Create the directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)
            
            # Download to a temporary location first
            temp_path = dataset.get_local_copy()
            
            # Move contents from temp location to desired location
            for item in os.listdir(temp_path):
                s = os.path.join(temp_path, item)
                d = os.path.join(local_path, item)
                if os.path.isdir(s):
                    if os.path.exists(d):
                        shutil.rmtree(d)
                    shutil.move(s, d)
                else:
                    shutil.copy2(s, d)
            
            # Clean up temporary directory
            shutil.rmtree(temp_path)
            
            print(f"Dataset downloaded successfully to {local_path}")
            return True
        except Exception as e:
            print(f"Error downloading dataset from ClearML: {str(e)}")
            print(f"This could be due to:")
            print(f"  1. Dataset '{dataset_name}' doesn't exist in project '{project_name}'")
            print(f"  2. ClearML credentials not properly configured")
            print(f"  3. Network connectivity issues")
            return False
    
    def validate_and_fix_dataset_structure(dataset_path: pathlib.Path, logger):
        """Validate and fix the dataset structure to match expected format."""
        logger.info("Validating dataset structure...")
        logger.info(f"Dataset path: {dataset_path}")
        
        # First, let's see what's actually in the dataset
        if dataset_path.exists():
            all_items = list(dataset_path.iterdir())
            logger.info(f"Items in dataset directory: {[item.name for item in all_items]}")
            
            # Show directory structure
            dirs = [item for item in all_items if item.is_dir()]
            files = [item for item in all_items if item.is_file()]
            logger.info(f"Directories: {[d.name for d in dirs]}")
            logger.info(f"Files: {[f.name for f in files]}")
            
            # Look deeper into subdirectories
            for dir_item in dirs:
                sub_items = list(dir_item.iterdir())
                logger.info(f"Contents of {dir_item.name}: {[item.name for item in sub_items]}")
        else:
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        expected_classes = ["Falling", "No Action", "Waving"]
        
        # Check if expected directories exist
        missing_dirs = []
        for class_name in expected_classes:
            class_dir = dataset_path / class_name
            if not class_dir.exists():
                missing_dirs.append(class_name)
        
        if missing_dirs:
            logger.warning(f"Missing expected directories: {missing_dirs}")
            
            # Try to find alternative directory names and map them
            existing_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
            logger.info(f"Existing directories: {existing_dirs}")
            
            # Enhanced mapping for common variations
            class_mappings = {
                "falling": "Falling",
                "fall": "Falling", 
                "falls": "Falling",
                "no_action": "No Action",
                "noaction": "No Action",
                "no-action": "No Action",
                "no action": "No Action",
                "normal": "No Action",
                "idle": "No Action",
                "standing": "No Action",
                "waving": "Waving",
                "wave": "Waving",
                "waves": "Waving",
                "hand_wave": "Waving",
                "handwave": "Waving",
                "hand-wave": "Waving"
            }
            
            # Try to rename directories to match expected format
            renamed_count = 0
            for existing_dir in existing_dirs:
                normalized_name = existing_dir.lower().replace(" ", "_").replace("-", "_")
                if normalized_name in class_mappings:
                    old_path = dataset_path / existing_dir
                    new_path = dataset_path / class_mappings[normalized_name]
                    if not new_path.exists():
                        old_path.rename(new_path)
                        logger.info(f"Renamed '{existing_dir}' to '{class_mappings[normalized_name]}'")
                        renamed_count += 1
            
            logger.info(f"Renamed {renamed_count} directories")
        
        # Final check - log what we found after renaming
        total_keypoint_files = 0
        for class_name in expected_classes:
            class_dir = dataset_path / class_name
            if class_dir.exists():
                keypoint_files = list(class_dir.glob("*keypoints.json"))
                json_files = list(class_dir.glob("*.json"))
                logger.info(f"Found {len(keypoint_files)} keypoints files and {len(json_files)} total JSON files in {class_name}")
                total_keypoint_files += len(keypoint_files)
                
                # Show a few example files
                if json_files:
                    logger.info(f"Example files in {class_name}: {[f.name for f in json_files[:3]]}")
            else:
                logger.warning(f"Directory {class_name} still missing after validation")
        
        logger.info(f"Total keypoints files found: {total_keypoint_files}")
        
        # If we still don't have the expected structure, check for nested structure
        if total_keypoint_files == 0:
            logger.info("No keypoints files found in expected locations. Checking for nested structures...")
            for item in dataset_path.iterdir():
                if item.is_dir():
                    nested_files = list(item.rglob("*.json"))
                    if nested_files:
                        logger.info(f"Found {len(nested_files)} JSON files in nested structure under {item.name}")
                        logger.info(f"Example nested files: {[str(f.relative_to(dataset_path)) for f in nested_files[:3]]}")

    try:
        comp_logger = logging.getLogger(f"Component.{download_and_setup_dataset.__name__}")
        comp_logger.info(f"Setting up dataset: {dataset_name} from project: {dataset_project}")
        
        # Create Path objects
        local_path_obj = pathlib.Path(local_target_path)
        
        # Use the robust download function
        success = download_real_dataset_from_clearml(
            dataset_name=dataset_name,
            project_name=dataset_project,
            local_path=str(local_path_obj)
        )
        
        if not success:
            comp_logger.error(f"Failed to download dataset '{dataset_name}' from ClearML")
            return None
        
        # Validate and fix dataset structure
        validate_and_fix_dataset_structure(local_path_obj, comp_logger)
        
        # Final check - ensure we have some data
        total_files = 0
        total_json_files = 0
        
        # Check expected structure first
        for class_name in ["Falling", "No Action", "Waving"]:
            class_dir = local_path_obj / class_name
            if class_dir.exists():
                keypoint_files = list(class_dir.glob("*keypoints.json"))
                json_files = list(class_dir.glob("*.json"))
                total_files += len(keypoint_files)
                total_json_files += len(json_files)
                comp_logger.info(f"Found {len(keypoint_files)} keypoint files and {len(json_files)} JSON files in {class_name}")
        
        # If no files in expected structure, check for any JSON files in the dataset
        if total_files == 0 and total_json_files == 0:
            comp_logger.info("No files in expected structure. Checking entire dataset for JSON files...")
            all_json_files = list(local_path_obj.rglob("*.json"))
            total_json_files = len(all_json_files)
            comp_logger.info(f"Found {total_json_files} JSON files total in dataset")
            
            if total_json_files > 0:
                comp_logger.info("Dataset contains JSON files but not in expected structure. This may still be usable.")
                # Show some example files
                example_files = [f.relative_to(local_path_obj) for f in all_json_files[:5]]
                comp_logger.info(f"Example files: {example_files}")
        
        if total_files == 0 and total_json_files == 0:
            comp_logger.error("No dataset files found")
            return None
        
        if total_files > 0:
            comp_logger.info(f"Dataset validation complete. Found {total_files} keypoints files in expected structure")
        else:
            comp_logger.info(f"Dataset validation complete. Found {total_json_files} JSON files (structure may need adjustment)")
        
        return str(local_path_obj) if local_path_obj.exists() and local_path_obj.is_dir() else None

    except Exception as e:
        comp_logger.error(f"Error in download_and_setup_dataset: {e}", exc_info=True)
        return None

# ============================================================================
# COMPONENT 2: DATA PREPARATION (SAME AS ORIGINAL)
# ============================================================================

@PipelineDecorator.component(return_values=["dataset_path", "input_size", "num_classes"])
def prepare_data(dataset_path: str):
    """Prepare data and return metadata for training."""
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
                self.max_frames = max_frames
                self.data, self.labels = self.load_data()

            def load_data(self):
                data = []
                labels = []
                for i, action in enumerate(self.action_classes):
                    action_dir = os.path.join(self.data_dir, action)
                    if not os.path.exists(action_dir):
                        comp_logger.warning(f"Directory not found: {action_dir}")
                        continue

                    keypoint_files = [f for f in os.listdir(action_dir) if f.endswith("_keypoints.json")]
                    comp_logger.info(f"Found {len(keypoint_files)} keypoint files in {action}")
                    
                    for filename in keypoint_files:
                        filepath = os.path.join(action_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                keypoints_data = json.load(f)
                                normalized_keypoints = self.process_keypoints(keypoints_data)
                                if normalized_keypoints is not None:
                                    data.append(normalized_keypoints)
                                    labels.append(i)
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            comp_logger.error(f"Error loading {filepath}: {e}")
                            continue
                
                comp_logger.info(f"Loaded {len(data)} samples total")
                return data, labels

            def process_keypoints(self, keypoints_data):
                all_frames_keypoints = []
                previous_frame = None
                alpha = 0.8

                for frame_data in keypoints_data:
                    if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                        continue

                    frame_keypoints = frame_data['keypoints']
                    if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                        continue

                    frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                    if frame_keypoints_np.shape != (17, 3):
                        continue

                    # Filter out keypoints with low confidence
                    valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                    if valid_keypoints.size == 0:
                        continue

                    # Z-Score Normalization
                    mean_x = np.mean(valid_keypoints[:, 0])
                    std_x = np.std(valid_keypoints[:, 0]) + 1e-8
                    mean_y = np.mean(valid_keypoints[:, 1])
                    std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                    normalized_frame_keypoints = frame_keypoints_np.copy()
                    normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                    normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                    # Temporal Smoothing using EMA
                    if previous_frame is not None:
                        normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                        normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                    previous_frame = normalized_frame_keypoints

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

# ============================================================================
# COMPONENT 3: ENHANCED TRAINING FOR GITHUB ACTIONS
# ============================================================================

@PipelineDecorator.component(
    name="Train_BiLSTM_GitHub",
    return_values=["task_id", "model_id"],
    packages=["torch>=1.9", "clearml", "scikit-learn", "numpy", "matplotlib"],
    task_type=Task.TaskTypes.training,
    cache=False
)
def train_bilstm_github(
    dataset_path: str,
    input_size: int = 34,
    num_classes: int = 3,
    base_lr: float = 0.001,
    epochs: int = 50,  # Full training epochs
    hidden_size: int = 256,  # Full model size
    num_layers: int = 4,  # Full model depth
    dropout_rate: float = 0.1,
    batch_size: int = 32,  # Standard batch size
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    grad_clip_norm: float = 1.0,
    noise_factor: float = 0.0,
    use_layer_norm: bool = False,
    attention_dropout: float = 0.1
):
    """Enhanced BiLSTM training with full feature set for GitHub Actions."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from clearml import Task, OutputModel
    import numpy as np
    import json
    import os
    import matplotlib.pyplot as plt

    # Enhanced model classes with full features
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size, dropout_rate=0.1):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, lstm_output):
            scores = self.attention_weights(self.dropout(lstm_output))
            attention_weights = torch.softmax(scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            return context_vector, attention_weights.squeeze(-1)

    class ActionRecognitionBiLSTMWithAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                     dropout_rate=0.5, use_layer_norm=False, attention_dropout=0.1):
            super(ActionRecognitionBiLSTMWithAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_layer_norm = use_layer_norm
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, 
                                bidirectional=True)
            
            if use_layer_norm:
                self.layer_norm = nn.LayerNorm(hidden_size * 2)
            
            self.attention = AttentionLayer(hidden_size, attention_dropout)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            
            if self.use_layer_norm:
                out = self.layer_norm(out)
                
            out = self.dropout(out)
            context_vector, attention_weights = self.attention(out)
            out = self.fc(context_vector)
            return out, attention_weights

    class PoseDataset(Dataset):
        def __init__(self, data_dir, action_classes, max_frames=40, noise_factor=0.0):
            self.data_dir = data_dir
            self.action_classes = action_classes
            self.max_frames = max_frames
            self.noise_factor = noise_factor
            self.data, self.labels = self.load_data()

        def load_data(self):
            data = []
            labels = []
            for i, action in enumerate(self.action_classes):
                action_dir = os.path.join(self.data_dir, action)
                if not os.path.exists(action_dir):
                    continue

                for filename in os.listdir(action_dir):
                    if filename.endswith("_keypoints.json"):
                        filepath = os.path.join(action_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                keypoints_data = json.load(f)
                                normalized_keypoints = self.process_keypoints(keypoints_data)
                                if normalized_keypoints is not None:
                                    data.append(normalized_keypoints)
                                    labels.append(i)
                        except Exception as e:
                            continue
                
            return data, labels

        def process_keypoints(self, keypoints_data):
            all_frames_keypoints = []
            previous_frame = None
            alpha = 0.8

            for frame_data in keypoints_data:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    continue

                frame_keypoints = frame_data['keypoints']
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                    continue

                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    continue

                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                if valid_keypoints.size == 0:
                    continue

                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                normalized_frame_keypoints = frame_keypoints_np.copy()
                normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                if previous_frame is not None:
                    normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                    normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                previous_frame = normalized_frame_keypoints
                normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
                all_frames_keypoints.append(normalized_frame_keypoints)

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
            data = self.data[idx].copy()
            
            # Add noise augmentation if specified
            if self.noise_factor > 0:
                noise = np.random.normal(0, self.noise_factor, data.shape)
                data = data + noise
            
            return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
    
    # Initialize task
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="Guardian_Training",
            task_name="Train_BiLSTM_GitHub"
        )
    
    logger = task.get_logger()
    
    # Connect hyperparameters
    hyperparams = {
        'General/base_lr': base_lr,
        'General/epochs': epochs,
        'General/hidden_size': hidden_size,
        'General/num_layers': num_layers,
        'General/dropout_rate': dropout_rate,
        'General/input_size': input_size,
        'General/num_classes': num_classes,
        'General/batch_size': batch_size,
        'General/weight_decay': weight_decay,
        'General/scheduler_patience': scheduler_patience,
        'General/scheduler_factor': scheduler_factor,
        'General/grad_clip_norm': grad_clip_norm,
        'General/noise_factor': noise_factor,
        'General/use_layer_norm': use_layer_norm,
        'General/attention_dropout': attention_dropout
    }
    task.connect(hyperparams)

    # Recreate DataLoaders from dataset path
    action_classes = ["Falling", "No Action", "Waving"]
    dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes, noise_factor=noise_factor)
    
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

    def make_torch_dataset_for_loader(split_data, split_labels, use_noise=False):
        temp_ds = PoseDataset(data_dir=dataset_path, action_classes=action_classes, 
                             noise_factor=noise_factor if use_noise else 0.0)
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds

    # Create data loaders (only apply noise to training data)
    train_loader = DataLoader(make_torch_dataset_for_loader(train_data, train_labels, use_noise=True), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_torch_dataset_for_loader(val_data, val_labels, use_noise=False), 
                           batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
        attention_dropout=attention_dropout
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, 
        patience=scheduler_patience
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0.0
    best_model_path = "best_bilstm_github.pth"
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += y.size(0)
            correct_train += predicted.eq(y).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train if total_train > 0 else 0.0

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                loss = criterion(outputs, y)

                total_val_loss += loss.item()
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().tolist())
                all_val_labels.extend(y.cpu().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        # Log metrics
        logger.report_scalar("Loss", "Train", avg_train_loss, epoch)
        logger.report_scalar("Loss", "Validation", avg_val_loss, epoch)
        logger.report_scalar("Accuracy", "Train", train_acc, epoch)
        logger.report_scalar("Accuracy", "Validation", val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
    
    # Generate training plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_metrics_github.png', dpi=150, bbox_inches='tight')
    logger.report_matplotlib_figure("Training Metrics", "GitHub", plt.gcf(), 0)
    plt.close()
    
    # Save hyperparameters as JSON file
    hyperparams_filename = f"hyperparams_{task.id}.json"
    standard_hyperparams_filename = "train_hyperparams.json"
    hyperparams_filepath = os.path.join(os.getcwd(), hyperparams_filename)
    standard_hyperparams_filepath = os.path.join(os.getcwd(), standard_hyperparams_filename)
    
    hyperparams_data = {
        "model_id": task.id,
        "hyperparameters": {
            "base_lr": base_lr,
            "epochs": epochs,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "input_size": input_size,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "scheduler_patience": scheduler_patience,
            "scheduler_factor": scheduler_factor,
            "grad_clip_norm": grad_clip_norm,
            "noise_factor": noise_factor,
            "use_layer_norm": use_layer_norm,
            "attention_dropout": attention_dropout
        },
        "training_results": {
            "best_validation_accuracy": best_acc,
            "training_epochs": epochs,
            "final_train_losses": train_losses[-5:] if len(train_losses) >= 5 else train_losses,
            "final_val_losses": val_losses[-5:] if len(val_losses) >= 5 else val_losses,
            "final_val_accuracies": val_accuracies[-5:] if len(val_accuracies) >= 5 else val_accuracies
        },
        "model_info": {
            "architecture": "BiLSTM with Enhanced Attention",
            "framework": "PyTorch",
            "model_type": "BiLSTM_ActionRecognition"
        }
    }
    
    try:
        # Save with task ID (for ClearML tracking)
        with open(hyperparams_filepath, 'w') as f:
            json.dump(hyperparams_data, f, indent=2)
        print(f"💾 Hyperparameters saved to {hyperparams_filepath}")
        
        # Also save with a standard name for easy reference
        with open(standard_hyperparams_filepath, 'w') as f:
            json.dump(hyperparams_data, f, indent=2)
        print(f"💾 Hyperparameters also saved to {standard_hyperparams_filepath}")
        
        # Upload hyperparameters as artifact
        task.upload_artifact(
            name="training_hyperparameters",
            artifact_object=hyperparams_filepath,
            metadata={
                "description": "Complete training hyperparameters and results",
                "best_accuracy": best_acc,
                "epochs": epochs
            }
        )
        print(f"📤 Hyperparameters uploaded as ClearML artifact")
        
        # Clean up any other hyperparameter files (optional during training)
        try:
            import glob
            for hp_file in glob.glob("hyperparams_*.json"):
                # Skip the current hyperparams file
                if hp_file == hyperparams_filename:
                    continue
                os.remove(hp_file)
                print(f"🧹 Removed old hyperparameter file: {hp_file}")
        except Exception as cleanup_error:
            print(f"⚠️ Error during hyperparameter cleanup: {cleanup_error}")
        
    except Exception as hyperparams_error:
        print(f"⚠️ Error saving hyperparameters: {hyperparams_error}")
    
    # Publish model
    output_model = OutputModel(task=task, name="BiLSTM_ActionRecognition", framework="PyTorch")
    output_model.update_weights(weights_filename=best_model_path)
    
    output_model.update_design(config_dict={
        "architecture": "BiLSTM with Enhanced Attention",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "grad_clip_norm": grad_clip_norm,
        "noise_factor": noise_factor,
        "use_layer_norm": use_layer_norm,
        "attention_dropout": attention_dropout,
        "best_validation_accuracy": best_acc,
        "framework": "PyTorch",
        "training_epochs": epochs,
        "hyperparams_saved": True
    })
    
    print(f"Model published with ID: {output_model.id}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    return task.id, output_model.id

# ============================================================================
# COMPONENT 4: HYPERPARAMETER OPTIMIZATION
# ============================================================================

@PipelineDecorator.component(
    name="BiLSTM_HPO_GitHub",
    return_values=["best_task_id", "best_model_id"],
    cache=False,
    packages=["clearml"]
)
def bilstm_hyperparam_optimizer_github(
    base_task_id: str,
    dataset_path: str,
    input_size: int,
    num_classes: int,
    total_max_trials: int = 90
):
    """
    Hyperparameter optimization for GitHub Actions with 30 trials.
    """
    from clearml.automation import HyperParameterOptimizer, RandomSearch
    from clearml.automation import DiscreteParameterRange, UniformParameterRange
    from clearml import Task, Model
    import json
    import os
    
    print(f"Starting hyperparameter optimization with {total_max_trials} trials...")
    
    # Initialize HPO task
    hpo_task = Task.init(
        project_name="Guardian_Training",
        task_name="BiLSTM_HPO_GitHub_Controller",
        task_type=Task.TaskTypes.optimizer,
        reuse_last_task_id=False
    )

    # Define search space
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task_id,
        hyper_parameters=[
            # Learning rate
            UniformParameterRange('General/base_lr', min_value=0.0001, max_value=0.01),
            
            # Hidden size
            DiscreteParameterRange('General/hidden_size', values=[128, 192, 256, 320, 384, 512]),
            
            # Number of layers
            DiscreteParameterRange('General/num_layers', values=[2, 3, 4, 5]),
            
            # Dropout rate
            UniformParameterRange('General/dropout_rate', min_value=0.05, max_value=0.5),
            
            # Epochs
            DiscreteParameterRange('General/epochs', values=[30, 40, 50, 60]),
            
            # Batch size
            DiscreteParameterRange('General/batch_size', values=[16, 24, 32, 48, 64]),
            
            # Weight decay
            UniformParameterRange('General/weight_decay', min_value=1e-6, max_value=1e-3),
            
            # Scheduler patience
            DiscreteParameterRange('General/scheduler_patience', values=[3, 5, 7, 10]),
            
            # Scheduler factor
            UniformParameterRange('General/scheduler_factor', min_value=0.2, max_value=0.8),
            
            # Gradient clipping
            UniformParameterRange('General/grad_clip_norm', min_value=0.5, max_value=3.0),
            
            # Noise factor
            UniformParameterRange('General/noise_factor', min_value=0.0, max_value=0.05),
            
            # Layer normalization
            DiscreteParameterRange('General/use_layer_norm', values=[True, False]),
            
            # Attention dropout
            UniformParameterRange('General/attention_dropout', min_value=0.0, max_value=0.2),
        ],
        
        # Objective metric
        objective_metric_title="Accuracy",
        objective_metric_series="Validation",
        objective_metric_sign="max",
        
        # Execution settings
        max_number_of_concurrent_tasks=2,  # Reduced for GitHub Actions
        optimizer_class=RandomSearch,
        save_top_k_tasks_only=5,
        
        # Fixed arguments
        base_task_kwargs={
            'dataset_path': dataset_path,
            'input_size': input_size,
            'num_classes': num_classes
        },
        
        total_max_jobs=total_max_trials,
    )
    
    print(f"Search space configured for {total_max_trials} trials")
    print("Starting optimization...")
    
    # Start optimization
    optimizer.start_locally()
    optimizer.wait()
    optimizer.stop()
    
    print("Hyperparameter optimization completed!")
    
    # Get best experiment
    top_exps = optimizer.get_top_experiments(top_k=1)
    if not top_exps:
        raise RuntimeError("No HPO experiments returned by optimizer.")
    
    best_exp = top_exps[0]
    best_exp_id = best_exp.id
    
    # Get validation accuracy
    metrics = best_exp.get_last_scalar_metrics()
    best_acc = None
    
    metric_paths = [
        ("Accuracy", "Validation"),
        ("metrics", "Validation_Accuracy"),
    ]
    
    for title, series in metric_paths:
        try:
            if title in metrics and series in metrics[title]:
                best_acc = metrics[title][series].get("last")
                if best_acc is not None:
                    break
        except (KeyError, AttributeError, TypeError):
            continue
    
    print(f"Best experiment ID: {best_exp_id}, Validation Accuracy: {best_acc}")
    
    # Get the best model
    best_exp_task = Task.get_task(task_id=best_exp_id)
    
    # Get the actual model from the task
    model_id = None
    if (best_exp_task.models and 
        'output' in best_exp_task.models and 
        len(best_exp_task.models['output']) > 0):
        model = best_exp_task.models['output'][0]
        model_id = model.id
        print(f"Found best model ID: {model_id}")
        
        # Extract architecture from the actual model weights
        try:
            import torch
            import os
            
            # Download the model weights
            model_path = model.get_local_copy()
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract architecture parameters directly from the checkpoint tensors
            lstm_hidden_size = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
            num_lstm_layers = 0
            while f'lstm.weight_ih_l{num_lstm_layers}' in checkpoint:
                num_lstm_layers += 1
            
            use_layer_norm = 'layer_norm.weight' in checkpoint
            
            print(f"📐 Extracted architecture from best model weights:")
            print(f"   - hidden_size: {lstm_hidden_size}")
            print(f"   - num_layers: {num_lstm_layers}")
            print(f"   - use_layer_norm: {use_layer_norm}")
            
            # Save best model hyperparameters as JSON file with actual architecture
            best_hyperparams = best_exp_task.get_parameters()
            
            # Create a clean version of the hyperparameters with the actual architecture
            architecture_aligned_hyperparams = {}
            
            # First add the General/ hyperparameters with default values
            default_hyperparams = {
                'General/base_lr': 0.001,
                'General/epochs': 50,
                'General/hidden_size': lstm_hidden_size,  # Use actual architecture 
                'General/num_layers': num_lstm_layers,    # Use actual architecture
                'General/dropout_rate': 0.1,
                'General/input_size': 34,
                'General/num_classes': 3,
                'General/batch_size': 32,
                'General/weight_decay': 1e-5,
                'General/scheduler_patience': 5,
                'General/scheduler_factor': 0.5,
                'General/grad_clip_norm': 1.0,
                'General/noise_factor': 0.0,
                'General/use_layer_norm': use_layer_norm,  # Use actual architecture
                'General/attention_dropout': 0.1
            }
            
            # Add defaults first
            for key, default_value in default_hyperparams.items():
                architecture_aligned_hyperparams[key] = default_value
                
            # Override with actual hyperparameters from the best experiment (except architecture)
            for key, value in best_hyperparams.items():
                if key.startswith('General/'):
                    # Skip architecture parameters that we've extracted from weights
                    if key in ['General/hidden_size', 'General/num_layers', 'General/use_layer_norm']:
                        continue
                    try:
                        if isinstance(value, dict) and 'value' in value:
                            architecture_aligned_hyperparams[key] = value['value']
                        else:
                            architecture_aligned_hyperparams[key] = value
                    except Exception as e:
                        print(f"⚠️ Error processing parameter {key}: {e}")
            
            # Create files with aligned hyperparameters
            best_hyperparams_filename = "best_hyperparams.json"
            best_hyperparams_filepath = os.path.join(os.getcwd(), best_hyperparams_filename)
            
            with open(best_hyperparams_filepath, 'w') as f:
                json.dump(architecture_aligned_hyperparams, f, indent=2, default=str)
            print(f"💾 Architecture-aligned hyperparameters saved to {best_hyperparams_filepath}")
            
            # Upload as artifact
            hpo_task.upload_artifact(
                name="architecture_aligned_hyperparameters",
                artifact_object=best_hyperparams_filepath,
                metadata={
                    "description": "Hyperparameters aligned with actual model architecture",
                    "best_experiment_id": best_exp_id,
                    "model_id": model_id,
                    "hidden_size": lstm_hidden_size,
                    "num_layers": num_lstm_layers,
                    "use_layer_norm": use_layer_norm
                }
            )
            print(f"📤 Architecture-aligned hyperparameters uploaded as ClearML artifact")
            
        except Exception as arch_error:
            print(f"⚠️ Could not extract architecture from model weights: {arch_error}")
            # Fall back to saving original hyperparameters
            best_hyperparams_filename = f"best_hyperparams_{best_exp_id}.json"
            best_hyperparams_filepath = os.path.join(os.getcwd(), best_hyperparams_filename)
            
            try:
                with open(best_hyperparams_filepath, 'w') as f:
                    json.dump(best_hyperparams, f, indent=2, default=str)
                print(f"💾 Best model hyperparameters saved to {best_hyperparams_filepath}")
                
                # Upload hyperparameters as artifact to the HPO task
                hpo_task.upload_artifact(
                    name="best_model_hyperparameters",
                    artifact_object=best_hyperparams_filepath,
                    metadata={
                        "description": "Hyperparameters of the best model from HPO",
                        "best_experiment_id": best_exp_id,
                    }
                )
                print(f"📤 Best hyperparameters uploaded as ClearML artifact")
            except Exception as hyperparams_error:
                print(f"⚠️ Error saving best hyperparameters: {hyperparams_error}")
    else:
        # No model found, fall back to original hyperparameters
        best_hyperparams_filename = f"best_hyperparams_{best_exp_id}.json"
        best_hyperparams_filepath = os.path.join(os.getcwd(), best_hyperparams_filename)
        
        try:
            with open(best_hyperparams_filepath, 'w') as f:
                json.dump(best_hyperparams, f, indent=2, default=str)
            print(f"💾 Best model hyperparameters saved to {best_hyperparams_filepath}")
            
            # Upload hyperparameters as artifact to the HPO task
            hpo_task.upload_artifact(
                name="best_model_hyperparameters",
                artifact_object=best_hyperparams_filepath,
                metadata={
                    "description": "Hyperparameters of the best model from HPO",
                    "best_experiment_id": best_exp_id,
                }
            )
            print(f"📤 Best hyperparameters uploaded as ClearML artifact")
            
            # Clean up any other hyperparameter files to keep only the best one
            try:
                # Find and remove other hyperparameter files in the current directory
                import glob
                for hp_file in glob.glob("hyperparams_*.json"):
                    # Skip the best hyperparams file
                    if hp_file == best_hyperparams_filename:
                        continue
                    try:
                        os.remove(hp_file)
                        print(f"🧹 Removed unnecessary hyperparameter file: {hp_file}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {hp_file}: {e}")
            except Exception as cleanup_error:
                print(f"⚠️ Error during hyperparameter cleanup: {cleanup_error}")
                
        except Exception as hyperparams_error:
            print(f"⚠️ Error saving best hyperparameters: {hyperparams_error}")
    
    # Return best experiment and model ids
    if model_id:
        return best_exp_id, model_id
    elif (best_exp_task.models and 
        'output' in best_exp_task.models and 
        len(best_exp_task.models['output']) > 0):
        model = best_exp_task.models['output'][0]
        model_id = model.id
        print(f"Best model ID: {model_id}")
        return best_exp_id, model_id
    else:
        # Fallback: search for published models
        models = Model.query_models(
            project_name="Guardian_Training",
            model_name="BiLSTM_ActionRecognition",
            only_published=True,
            max_results=5,
            order_by=['-created']
        )
        
        if models and len(models) > 0:
            best_model = models[0]
            print(f"Using fallback model ID: {best_model.id}")
            return best_exp_id, best_model.id
        else:
            raise RuntimeError("No models found for the best experiment.")

# ============================================================================
# COMPONENT 5: MODEL EVALUATION
# ============================================================================

@PipelineDecorator.component(
    name="Evaluate_Model_GitHub",
    return_values=["test_accuracy"],
    cache=False,
    packages=["torch", "scikit-learn", "numpy", "clearml", "matplotlib", "seaborn"]
)
def evaluate_model_github(
    best_task_id: str,
    best_model_id: str,
    dataset_path: str,
    input_size: int = 34,
    num_classes: int = 3
):
    """Evaluate the best model from HPO."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from clearml import Task, Model
    import numpy as np
    import json
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Model classes (same as training)
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size, dropout_rate=0.1):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, lstm_output):
            scores = self.attention_weights(self.dropout(lstm_output))
            attention_weights = torch.softmax(scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            return context_vector, attention_weights.squeeze(-1)

    class ActionRecognitionBiLSTMWithAttention(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                     dropout_rate=0.5, use_layer_norm=False, attention_dropout=0.1):
            super(ActionRecognitionBiLSTMWithAttention, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.use_layer_norm = use_layer_norm
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout_rate if num_layers > 1 else 0, 
                                bidirectional=True)
            
            if use_layer_norm:
                self.layer_norm = nn.LayerNorm(hidden_size * 2)
            
            self.attention = AttentionLayer(hidden_size, attention_dropout)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
            
            if self.use_layer_norm:
                out = self.layer_norm(out)
                
            out = self.dropout(out)
            context_vector, attention_weights = self.attention(out)
            out = self.fc(context_vector)
            return out, attention_weights

    class PoseDataset(Dataset):
        def __init__(self, data_dir, action_classes, max_frames=40):
            self.data_dir = data_dir
            self.action_classes = action_classes
            self.max_frames = max_frames
            self.data, self.labels = self.load_data()

        def load_data(self):
            data = []
            labels = []
            for i, action in enumerate(self.action_classes):
                action_dir = os.path.join(self.data_dir, action)
                if not os.path.exists(action_dir):
                    continue

                for filename in os.listdir(action_dir):
                    if filename.endswith("_keypoints.json"):
                        filepath = os.path.join(action_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                keypoints_data = json.load(f)
                                normalized_keypoints = self.process_keypoints(keypoints_data)
                                if normalized_keypoints is not None:
                                    data.append(normalized_keypoints)
                                    labels.append(i)
                        except Exception as e:
                            continue
                
            return data, labels

        def process_keypoints(self, keypoints_data):
            all_frames_keypoints = []
            previous_frame = None
            alpha = 0.8

            for frame_data in keypoints_data:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    continue

                frame_keypoints = frame_data['keypoints']
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                    continue

                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    continue

                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
                if valid_keypoints.size == 0:
                    continue

                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                normalized_frame_keypoints = frame_keypoints_np.copy()
                normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                if previous_frame is not None:
                    normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                    normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

                previous_frame = normalized_frame_keypoints
                normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
                all_frames_keypoints.append(normalized_frame_keypoints)

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

    # Initialize evaluation task
    task = Task.init(
        project_name="Guardian_Training",
        task_name="Evaluate_Best_Model_GitHub",
        task_type=Task.TaskTypes.testing
    )
    
    logger = task.get_logger()
    
    # Get best task and model details
    best_task = Task.get_task(task_id=best_task_id)
    best_model = Model(model_id=best_model_id)
    
    print(f"🔍 Analyzing best model architecture...")
    
    # Get model weights first to dynamically detect architecture
    model_path = best_model.get_local_copy()
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Dynamically infer all architecture parameters from checkpoint
    def infer_architecture_from_checkpoint(checkpoint):
        """Dynamically infer model architecture from saved weights."""
        config = {}
        
        try:
            # Infer hidden_size from LSTM weights
            lstm_weight_shape = checkpoint['lstm.weight_ih_l0'].shape
            config['hidden_size'] = lstm_weight_shape[0] // 4  # LSTM has 4 gates
            
            # Infer num_layers by counting LSTM layer weights
            config['num_layers'] = 0
            layer_idx = 0
            while f'lstm.weight_ih_l{layer_idx}' in checkpoint:
                config['num_layers'] += 1
                layer_idx += 1
            
            # Check if layer normalization is used
            config['use_layer_norm'] = 'layer_norm.weight' in checkpoint
            
            # Infer input_size from first LSTM layer
            config['input_size'] = checkpoint['lstm.weight_ih_l0'].shape[1]
            
            # Infer num_classes from final layer
            config['num_classes'] = checkpoint['fc.weight'].shape[0]
            
            # Check if attention dropout layer exists (harder to detect, use default)
            config['attention_dropout'] = 0.1  # Default, hard to infer from weights
            
            # Dropout rate is hard to infer from weights, use default
            config['dropout_rate'] = 0.1  # Default, hard to infer from weights
            
            # Print all checkpoint keys for better debugging
            print(f"📋 Checkpoint keys: {sorted(list(checkpoint.keys()))}")
            
            # Print key tensor shapes for debugging
            print(f"📏 Key tensor shapes:")
            print(f"   - lstm.weight_ih_l0: {checkpoint['lstm.weight_ih_l0'].shape}")
            print(f"   - lstm.weight_hh_l0: {checkpoint['lstm.weight_hh_l0'].shape}")
            if 'fc.weight' in checkpoint:
                print(f"   - fc.weight: {checkpoint['fc.weight'].shape}")
            if 'attention.attention_weights.weight' in checkpoint:
                print(f"   - attention.attention_weights.weight: {checkpoint['attention.attention_weights.weight'].shape}")
            if 'layer_norm.weight' in checkpoint:
                print(f"   - layer_norm.weight: {checkpoint['layer_norm.weight'].shape}")
            
            print(f"✅ Inferred architecture:")
            print(f"   - hidden_size: {config['hidden_size']}")
            print(f"   - num_layers: {config['num_layers']}")
            print(f"   - input_size: {config['input_size']}")
            print(f"   - num_classes: {config['num_classes']}")
            print(f"   - use_layer_norm: {config['use_layer_norm']}")
            print(f"   - dropout_rate: {config['dropout_rate']} (default)")
            print(f"   - attention_dropout: {config['attention_dropout']} (default)")
            
            return config
            
        except Exception as e:
            print(f"❌ Error inferring architecture: {e}")
            print(f"Available checkpoint keys: {sorted(list(checkpoint.keys())) if checkpoint else 'No keys'}")
            return None
    
    # Try to infer architecture from checkpoint
    inferred_config = infer_architecture_from_checkpoint(checkpoint)
    
    if inferred_config:
        # Use inferred configuration
        hidden_size = inferred_config['hidden_size']
        num_layers = inferred_config['num_layers']
        dropout_rate = inferred_config['dropout_rate']
        use_layer_norm = inferred_config['use_layer_norm']
        attention_dropout = inferred_config['attention_dropout']
        input_size = inferred_config['input_size']  # Override with actual
        num_classes = inferred_config['num_classes']  # Override with actual
        batch_size = 32  # Default for evaluation
        
    else:
        print("⚠️  Could not infer architecture, using fallback methods...")
        
        # Fallback 1: Try model config
        try:
            # Try to get model configuration from the model object
            model_config = model.config_dict if hasattr(model, 'config_dict') else {}
            if not model_config and hasattr(model, 'get_model_config'):
                model_config = model.get_model_config() or {}
        except Exception as config_error:
            print(f"⚠️ Could not retrieve model config: {config_error}")
            model_config = {}
            
        if model_config:
            print(f"📋 Using model config: {model_config}")
            hidden_size = int(model_config.get('hidden_size', 256))
            num_layers = int(model_config.get('num_layers', 4))
            dropout_rate = float(model_config.get('dropout_rate', 0.1))
            use_layer_norm = bool(model_config.get('use_layer_norm', False))
            attention_dropout = float(model_config.get('attention_dropout', 0.1))
            batch_size = int(model_config.get('batch_size', 32))
        else:
            print("📋 Using task parameters...")
            # Fallback 2: Task parameters
            best_params = best_task.get_parameters()
            
            def safe_get_param(params, key, default, param_type):
                try:
                    value = params.get(key, default)
                    if isinstance(value, dict) and 'value' in value:
                        value = value['value']
                    return param_type(value)
                except (ValueError, TypeError):
                    return param_type(default)
            
            hidden_size = safe_get_param(best_params, 'General/hidden_size', 256, int)
            num_layers = safe_get_param(best_params, 'General/num_layers', 4, int)
            dropout_rate = safe_get_param(best_params, 'General/dropout_rate', 0.1, float)
            use_layer_norm = safe_get_param(best_params, 'General/use_layer_norm', False, bool)
            attention_dropout = safe_get_param(best_params, 'General/attention_dropout', 0.1, float)
            batch_size = safe_get_param(best_params, 'General/batch_size', 32, int)
    
    print(f"🏗️  Final model configuration:")
    print(f"   - input_size: {input_size}")
    print(f"   - hidden_size: {hidden_size}")
    print(f"   - num_layers: {num_layers}")
    print(f"   - num_classes: {num_classes}")
    print(f"   - dropout_rate: {dropout_rate}")
    print(f"   - use_layer_norm: {use_layer_norm}")
    print(f"   - attention_dropout: {attention_dropout}")
    print(f"   - batch_size: {batch_size}")
    
    # Load dataset
    action_classes = ["Falling", "No Action", "Waving"]
    dataset = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
    
    # Split data (same as training)
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42,
        stratify=dataset.labels if len(set(dataset.labels)) > 1 else None
    )
    
    def make_torch_dataset_for_loader(split_data, split_labels):
        temp_ds = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds
    
    test_loader = DataLoader(make_torch_dataset_for_loader(test_data, test_labels), 
                            batch_size=batch_size, shuffle=False)
    
    # Initialize model with dynamically detected architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Creating model with detected architecture...")
    
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        use_layer_norm=use_layer_norm,
        attention_dropout=attention_dropout
    ).to(device)
    
    # Load model weights (checkpoint already loaded above)
    try:
        model.load_state_dict(checkpoint)
        print("✅ Model weights loaded successfully!")
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print(f"Model architecture: {model}")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        print(f"⚠️ Architecture mismatch detected. Attempting fallback approaches...")
        
        # Fallback 1: Try with strict=False to skip missing keys
        try:
            print(f"Trying load_state_dict with strict=False")
            model.load_state_dict(checkpoint, strict=False)
            print(f"✓ Model loaded with strict=False (some parameters may not be loaded correctly)")
        except Exception as strict_false_error:
            print(f"❌ Failed to load with strict=False: {strict_false_error}")
            
            # Fallback 2: Create a new model with exactly the architecture detected from the checkpoint
            print(f"Creating new model with architecture exactly matching checkpoint...")
            
            # Extract architecture directly from checkpoint tensors
            try:
                lstm_hidden_size = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
                num_lstm_layers = 0
                while f'lstm.weight_ih_l{num_lstm_layers}' in checkpoint:
                    num_lstm_layers += 1
                
                use_layer_norm = 'layer_norm.weight' in checkpoint
                checkpoint_input_size = checkpoint['lstm.weight_ih_l0'].shape[1]
                checkpoint_num_classes = checkpoint['fc.weight'].shape[0]
                
                print(f"Exact checkpoint architecture:")
                print(f" - hidden_size: {lstm_hidden_size}")
                print(f" - num_layers: {num_lstm_layers}")
                print(f" - use_layer_norm: {use_layer_norm}")
                print(f" - input_size: {checkpoint_input_size}")
                print(f" - num_classes: {checkpoint_num_classes}")
                
                # Create model with exact checkpoint architecture
                model = ActionRecognitionBiLSTMWithAttention(
                    input_size=checkpoint_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=num_lstm_layers,
                    num_classes=checkpoint_num_classes,
                    dropout_rate=0.1,  # Use default as we can't infer from weights
                    use_layer_norm=use_layer_norm,
                    attention_dropout=0.1  # Use default as we can't infer from weights
                ).to(device)
                
                # Try loading again with the exact architecture
                model.load_state_dict(checkpoint)
                print(f"✅ Success! Model loaded with exact checkpoint architecture")
            except Exception as exact_arch_error:
                print(f"❌ Failed even with exact architecture: {exact_arch_error}")
                raise RuntimeError("Could not load model weights with any approach")
    
    model.eval()
    
    # Evaluate
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs, _ = model(x)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Log metrics
    logger.report_scalar("Accuracy", "Test", test_accuracy, 0)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=action_classes, yticklabels=action_classes)
    plt.title('Confusion Matrix - Test Set', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_github.png', dpi=150, bbox_inches='tight')
    logger.report_matplotlib_figure("Confusion Matrix", "Test Set", plt.gcf(), 0)
    plt.close()
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=action_classes, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=action_classes)
    
    # Log per-class metrics
    for class_name in action_classes:
        if class_name in report:
            logger.report_scalar("Precision", class_name, report[class_name]['precision'], 0)
            logger.report_scalar("Recall", class_name, report[class_name]['recall'], 0)
            logger.report_scalar("F1-Score", class_name, report[class_name]['f1-score'], 0)
    
    logger.report_text(f"Classification Report:\n{report_str}", print_console=True)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    return float(test_accuracy)

# ============================================================================
# COMPONENT 6: MODEL DEPLOYMENT
# ============================================================================

@PipelineDecorator.component(
    name="Deploy_Model_GitHub",
    return_values=["deployment_status"],
    cache=False,
    packages=["clearml", "pymongo", "torch", "gridfs"]
)
def deploy_model_github(
    best_model_id: str,
    best_model_path: str,
    test_accuracy: float,
    min_accuracy_threshold: float = 85.0,
    mongo_uri: str = None
):
    """Deploy the best model if it meets accuracy threshold and save to MongoDB."""
    from clearml import Model, Task
    import os
    import torch
    import json
    import logging
    import shutil
    import sys
    
    # Add the current directory to the path for importing local modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Try to import the mongodb_model_distribution module
    try:
        from mongodb_model_distribution import GuardianModelDistribution
        has_model_distribution = True
        print("✅ Found mongodb_model_distribution module")
    except ImportError:
        has_model_distribution = False
        print("⚠️ mongodb_model_distribution module not found. Will use basic MongoDB storage.")
    
    task = Task.init(
        project_name="Guardian_Training",
        task_name="Deploy_Best_Model_GitHub",
        task_type=Task.TaskTypes.service
    )
    
    logger = task.get_logger()
    
    def safe_extract_hyperparameter_value(param_value, default_value):
        """Safely extract hyperparameter value from various formats."""
        if param_value is None:
            return default_value
        
        # If it's already a basic type (int, float, bool, str), return it
        if isinstance(param_value, (int, float, bool, str)):
            return param_value
        
        # If it's a dict with 'value' key (ClearML format)
        if isinstance(param_value, dict):
            if 'value' in param_value:
                return param_value['value']
            # If it's just a regular dict, try to extract the value
            elif len(param_value) == 1:
                return list(param_value.values())[0]
            # Empty dict should return default value
            elif len(param_value) == 0:
                return default_value
            else:
                return param_value
        
        # If it's a list, take the first element
        if isinstance(param_value, list) and len(param_value) > 0:
            return param_value[0]
        
        # Default fallback
        return default_value
    
    def process_hyperparameters(raw_hyperparams):
        """Process hyperparameters from ClearML to a clean dictionary."""
        processed = {}
        
        if not raw_hyperparams:
            return processed
        
        # Define the expected hyperparameters with defaults
        expected_params = {
            'General/base_lr': 0.001,
            'General/epochs': 50,
            'General/hidden_size': 256,
            'General/num_layers': 4,
            'General/dropout_rate': 0.1,
            'General/input_size': 34,
            'General/num_classes': 3,
            'General/batch_size': 32,
            'General/weight_decay': 1e-5,
            'General/scheduler_patience': 5,
            'General/scheduler_factor': 0.5,
            'General/grad_clip_norm': 1.0,
            'General/noise_factor': 0.0,
            'General/use_layer_norm': False,
            'General/attention_dropout': 0.1
        }
        
        # Process each expected parameter
        for param_key, default_value in expected_params.items():
            raw_value = raw_hyperparams.get(param_key, default_value)
            processed[param_key] = safe_extract_hyperparameter_value(raw_value, default_value)
        
        # Also include any additional parameters that might be present
        for key, value in raw_hyperparams.items():
            if key not in processed:
                processed[key] = safe_extract_hyperparameter_value(value, value)
        
        return processed
    
    print(f"Deploying model {best_model_id} with test accuracy: {test_accuracy:.2f}%")
    
    if test_accuracy >= min_accuracy_threshold:
        try:
            # Get the model
            model = Model(model_id=best_model_id)
            
            # Publish the model for deployment
            model.publish()
            print(f"📤 Model published successfully")
            
            # Add deployment tags using the correct API
            try:
                # Try the correct method for adding tags
                current_tags = model.tags or []
                new_tags = list(set(current_tags + ["deployed", "production", "github-actions"]))
                model.edit(tags=new_tags)
                print(f"🏷️  Added deployment tags: {new_tags}")
            except Exception as tag_error:
                print(f"⚠️  Could not add tags: {tag_error}")
                # Continue anyway - tags are not critical
            
            # Get the best task to retrieve hyperparameters
            try:
                best_task = Task.get_task(task_id=model.task)
            except (AttributeError, Exception) as e:
                print(f"⚠️  Could not access model.task attribute: {e}")
                best_task = None
                
            if not best_task:
                print("⚠️  Could not retrieve task for model hyperparameters")
                raw_hyperparams = {}
            else:
                raw_hyperparams = best_task.get_parameters() or {}
                print(f"📋 Retrieved hyperparameters from task {best_task.id}")
            
            # Process hyperparameters safely
            hyperparams = process_hyperparameters(raw_hyperparams)
            print(f"✅ Processed {len(hyperparams)} hyperparameters")
            
            # Create hyperparameters config file
            config_filename = f"model_config_{best_model_id[:8]}.json"
            config_filepath = os.path.join(os.getcwd(), config_filename)
            
            # Prepare comprehensive model configuration
            model_config = {
                "model_metadata": {
                    "model_id": best_model_id,
                    "test_accuracy": float(test_accuracy),
                    "training_task_id": str(best_task.id) if best_task else "unknown",
                    "framework": "PyTorch",
                    "model_type": "BiLSTM_ActionRecognition",
                    "architecture": "BiLSTM with Enhanced Attention",
                    "description": "Guardian AI Action Recognition Model",
                    "deployment_threshold": min_accuracy_threshold,
                    "deployed_by": "GitHub Actions",
                    "deployment_date": str(datetime.now())
                },
                "hyperparameters": hyperparams,
                "model_architecture": {
                    "input_size": safe_extract_hyperparameter_value(hyperparams.get("General/input_size"), 34),
                    "hidden_size": safe_extract_hyperparameter_value(hyperparams.get("General/hidden_size"), 256),
                    "num_layers": safe_extract_hyperparameter_value(hyperparams.get("General/num_layers"), 4),
                    "num_classes": safe_extract_hyperparameter_value(hyperparams.get("General/num_classes"), 3),
                    "dropout_rate": safe_extract_hyperparameter_value(hyperparams.get("General/dropout_rate"), 0.1),
                    "use_layer_norm": safe_extract_hyperparameter_value(hyperparams.get("General/use_layer_norm"), False),
                    "attention_dropout": safe_extract_hyperparameter_value(hyperparams.get("General/attention_dropout"), 0.1)
                },
                "training_config": {
                    "base_lr": safe_extract_hyperparameter_value(hyperparams.get("General/base_lr"), 0.001),
                    "epochs": safe_extract_hyperparameter_value(hyperparams.get("General/epochs"), 50),
                    "batch_size": safe_extract_hyperparameter_value(hyperparams.get("General/batch_size"), 32),
                    "weight_decay": safe_extract_hyperparameter_value(hyperparams.get("General/weight_decay"), 1e-5),
                    "scheduler_patience": safe_extract_hyperparameter_value(hyperparams.get("General/scheduler_patience"), 5),
                    "scheduler_factor": safe_extract_hyperparameter_value(hyperparams.get("General/scheduler_factor"), 0.5),
                    "grad_clip_norm": safe_extract_hyperparameter_value(hyperparams.get("General/grad_clip_norm"), 1.0),
                    "noise_factor": safe_extract_hyperparameter_value(hyperparams.get("General/noise_factor"), 0.0)
                }
            }
            
            # Save configuration as JSON file
            try:
                with open(config_filepath, 'w') as config_file:
                    json.dump(model_config, config_file, indent=2, default=str)
                print(f"💾 Model configuration saved to {config_filepath}")
                
                # Upload configuration as artifact to ClearML task
                task.upload_artifact(
                    name="model_configuration",
                    artifact_object=config_filepath,
                    metadata={
                        "description": "Complete model configuration including hyperparameters",
                        "model_id": best_model_id,
                        "accuracy": test_accuracy
                    }
                )
                print(f"📤 Configuration uploaded as ClearML artifact")
                
            except Exception as config_error:
                print(f"⚠️  Error saving model configuration: {config_error}")
                config_filepath = None
            
            # Update model metadata
            try:
                model.update_design(config_dict={
                    "deployment_status": "deployed",
                    "test_accuracy": test_accuracy,
                    "deployment_threshold": min_accuracy_threshold,
                    "deployed_by": "GitHub Actions",
                    "mongodb_stored": False,  # Will update if MongoDB storage succeeds
                    "config_saved": config_filepath is not None
                })
                print(f"📋 Updated model metadata")
            except Exception as metadata_error:
                print(f"⚠️  Could not update metadata: {metadata_error}")
                # Continue anyway - metadata is not critical
            
            # MongoDB integration - Store model weights and hyperparameters
            if mongo_uri:
                try:
                    print(f"🔄 Connecting to MongoDB for model storage...")
                    
                    # Ensure the model path exists
                    if not os.path.exists(best_model_path):
                        model_path = model.get_local_copy()
                        print(f"📥 Model weights downloaded to {model_path}")
                    else:
                        model_path = best_model_path
                        print(f"📄 Using existing model weights at {model_path}")
                    
                    # Load model to extract architecture
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        print(f"✅ Model weights loaded successfully!")
                    except Exception as e:
                        print(f"⚠️ Error loading model weights: {e}")
                        checkpoint = {}
                    
                    # Create model name with timestamp and accuracy
                    model_name = f"guardian_model_{best_model_id[:8]}_{int(test_accuracy)}"
                    
                    # Prepare model metadata for distribution (using processed hyperparams)
                    model_metadata = {
                        "model_id": best_model_id,
                        "test_accuracy": float(test_accuracy),
                        "training_task_id": str(best_task.id) if best_task else "unknown",
                        "architecture": model.config_dict if hasattr(model, 'config_dict') else {},
                        "hyperparameters": hyperparams,  # Use processed hyperparams
                        "checkpoint_keys": list(checkpoint.keys()) if checkpoint else [],
                        "input_size": safe_extract_hyperparameter_value(hyperparams.get("General/input_size"), 34),
                        "hidden_size": safe_extract_hyperparameter_value(hyperparams.get("General/hidden_size"), 256),
                        "num_layers": safe_extract_hyperparameter_value(hyperparams.get("General/num_layers"), 4),
                        "num_classes": safe_extract_hyperparameter_value(hyperparams.get("General/num_classes"), 3),
                        "framework": "PyTorch",
                        "model_type": "BiLSTM_ActionRecognition",
                        "description": "Guardian AI Action Recognition Model"
                    }
                    
                    # Use the GuardianModelDistribution class if available
                    if has_model_distribution:
                        print("🔄 Using GuardianModelDistribution for model storage...")
                        distributor = GuardianModelDistribution(uri=mongo_uri)
                        
                        if distributor.connect():
                            # Upload model using the distribution system
                            result = distributor.upload_model(
                                model_path=model_path,
                                model_metadata=model_metadata,
                                model_name=model_name
                            )
                            
                            if result:
                                print(f"🗃️ Model uploaded to distribution system:")
                                print(f"   Model Name: {result['model_name']}")
                                print(f"   Document ID: {result['document_id']}")
                                print(f"   Download Command: {result['download_command']}")
                                
                                # Also upload the config file if it exists
                                if config_filepath and os.path.exists(config_filepath):
                                    try:
                                        config_result = distributor.upload_model(
                                            model_path=config_filepath,
                                            model_metadata={
                                                **model_metadata,
                                                "file_type": "configuration",
                                                "description": "Model configuration and hyperparameters"
                                            },
                                            model_name=f"{model_name}_config"
                                        )
                                        if config_result:
                                            print(f"🗃️ Configuration uploaded:")
                                            print(f"   Config Name: {config_result['model_name']}")
                                    except Exception as config_upload_error:
                                        print(f"⚠️  Could not upload config to distribution: {config_upload_error}")
                                
                                # Update model metadata to reflect MongoDB storage
                                model.update_design(config_dict={"mongodb_stored": True})
                            else:
                                print("❌ Failed to upload model to distribution system")
                        else:
                            print("❌ Failed to connect to MongoDB distribution system")
                    else:
                        # Fallback to basic MongoDB storage
                        from pymongo import MongoClient
                        import gridfs
                        
                        # Connect to MongoDB with SSL options to fix TLSv1 alert issues
                        client = MongoClient(
                            mongo_uri,
                            ssl=True,
                            ssl_cert_reqs=ssl.CERT_NONE,  # Skip certificate validation
                            tlsAllowInvalidCertificates=True,  # Allow invalid certificates
                            tlsInsecure=True  # Skip hostname validation
                        )
                        db = client.guardian_models
                        fs = gridfs.GridFS(db)
                        
                        # Store the model weights
                        with open(model_path, 'rb') as f:
                            weights_file_id = fs.put(
                                f, 
                                filename=f"{model_name}.pth",
                                metadata={
                                    "model_id": best_model_id,
                                    "accuracy": float(test_accuracy),
                                    "deployment_date": str(datetime.now()),
                                    "file_type": "model_weights"
                                }
                            )
                        
                        # Store the configuration file if it exists
                        config_file_id = None
                        if config_filepath and os.path.exists(config_filepath):
                            try:
                                with open(config_filepath, 'rb') as f:
                                    config_file_id = fs.put(
                                        f,
                                        filename=f"{model_name}_config.json",
                                        metadata={
                                            "model_id": best_model_id,
                                            "accuracy": float(test_accuracy),
                                            "deployment_date": str(datetime.now()),
                                            "file_type": "configuration"
                                        }
                                    )
                                print(f"📁 Configuration file stored in MongoDB")
                            except Exception as config_store_error:
                                print(f"⚠️  Error storing config file: {config_store_error}")
                        
                        model_info = {
                            "model_name": model_name,
                            "model_id": best_model_id,
                            "test_accuracy": float(test_accuracy),
                            "weights_file_id": weights_file_id,
                            "config_file_id": config_file_id,
                            "hyperparameters": hyperparams,  # Use processed hyperparams
                            "deployment_status": "deployed",
                            "architecture": model.config_dict if hasattr(model, 'config_dict') else {},
                            "checkpoint_keys": list(checkpoint.keys()) if checkpoint else [],
                            "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
                            "status": "available",
                            "download_count": 0,
                            "uploaded_at": str(datetime.now()),
                            "file_id": weights_file_id
                        }
                        
                        # Store model metadata
                        db.model_metadata.insert_one(model_info)
                        
                        print(f"🗃️ Model weights and metadata saved to MongoDB")
                        print(f"   Model Name: {model_name}")
                        print(f"   File Size: {model_info['file_size_mb']:.2f} MB")
                        if config_file_id:
                            print(f"   Config File: Stored with ID {config_file_id}")
                        
                        # Update model metadata to reflect MongoDB storage
                        model.update_design(config_dict={"mongodb_stored": True})
                    
                except Exception as mongo_error:
                    print(f"❌ MongoDB storage error: {mongo_error}")
                    logger.report_text(f"MongoDB storage failed: {mongo_error}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
            else:
                print("ℹ️ MongoDB URI not provided, skipping database storage")
            
            logger.report_scalar("Deployment", "Status", 1, 0)  # 1 = deployed
            logger.report_scalar("Deployment", "Test_Accuracy", test_accuracy, 0)
            
            print(f"✅ Model deployed successfully!")
            print(f"📊 Test Accuracy: {test_accuracy:.2f}%")
            print(f"🎯 Threshold: {min_accuracy_threshold}%")
            print(f"🏷️  Model ID: {best_model_id}")
            if config_filepath:
                print(f"📄 Config saved: {config_filename}")
            
            return "deployed"
            
        except Exception as e:
            print(f"❌ Error during deployment: {e}")
            logger.report_scalar("Deployment", "Status", 0, 0)  # 0 = failed
            logger.report_scalar("Deployment", "Test_Accuracy", test_accuracy, 0)
            print(f"⚠️  Model met accuracy threshold but deployment failed")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return "deployment_failed"
    else:
        logger.report_scalar("Deployment", "Status", 0, 0)  # 0 = not deployed
        logger.report_scalar("Deployment", "Test_Accuracy", test_accuracy, 0)
        
        print(f"❌ Model not deployed - accuracy {test_accuracy:.2f}% below threshold {min_accuracy_threshold}%")
        
        return "not_deployed"

# ============================================================================
# MAIN PIPELINE FOR GITHUB ACTIONS
# ============================================================================

@PipelineDecorator.pipeline(
    name="Guardian_Pipeline_GitHub",
    project="Guardian_Training"
)
def guardian_github_pipeline():
    """Complete Guardian AI pipeline with HPO and deployment for GitHub Actions."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Guardian GitHub Pipeline started...")
    
    # Setup paths
    dataset_name = "Guardian_Dataset" 
    dataset_project = "Guardian_Training" 
    
    # Get MongoDB URI from environment variable
    mongo_uri = os.environ.get("MONGODB_URI", None)
    if mongo_uri:
        logging.info("MongoDB URI configured for model storage")
    else:
        logging.warning("MongoDB URI not found in environment variables. Models will not be stored in MongoDB.")
    
    # Multiple path options for your self-hosted runner
    possible_paths = [
        # Your absolute dataset path
        pathlib.Path("/home/sagemaker-user/data/Guardian_Dataset"),
        # GitHub Actions workspace path
        pathlib.Path("/home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training/data/Guardian_Dataset"),
        # Current working directory relative path
        pathlib.Path.cwd() / "data" / dataset_name,
        # Script directory relative path
        (pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path(".").resolve()) / "data" / dataset_name
    ]
    
    dataset_path = None
    for path in possible_paths:
        if path.exists():
            dataset_path = path
            logging.info(f"Found dataset at: {dataset_path}")
            break
    
    if not dataset_path:
        # Use the first path as default (will trigger dataset creation)
        dataset_path = possible_paths[0]
        logging.info(f"No existing dataset found. Will use: {dataset_path}")
    
    # Step 1: Download and setup dataset with validation
    logging.info("Starting dataset download and setup...")
    dataset_path_output = download_and_setup_dataset(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        local_target_path=str(dataset_path)
    )
    if not dataset_path_output:
        raise ValueError("Dataset setup failed.")
    logging.info(f"Dataset setup completed. Using path: {dataset_path_output}")

    # Step 2: Prepare data
    logging.info("Starting data preparation...")
    dataset_path, input_size, num_classes = prepare_data(
        dataset_path=dataset_path_output
    )
    if not dataset_path:
        raise RuntimeError("Data preparation failed.")
    logging.info("Data preparation completed.")

    # Step 3: Train baseline model
    logging.info("Starting baseline model training...")
    base_task_id, base_model_id = train_bilstm_github(
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes
    )
    if not base_task_id:
        raise RuntimeError("Baseline training failed.")
    logging.info(f"Baseline training completed. Task ID: {base_task_id}, Model ID: {base_model_id}")

    # Step 4: Hyperparameter optimization (90 trials)
    logging.info("Starting hyperparameter optimization...")
    best_task_id, best_model_id = bilstm_hyperparam_optimizer_github(
        base_task_id=base_task_id,
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes,
        total_max_trials=100
    )
    logging.info(f"HPO completed. Best task ID: {best_task_id}, Best model ID: {best_model_id}")

    # Get the best model path from ClearML
    try:
        from clearml import Model
        logging.info(f"Retrieving best model with ID: {best_model_id}")
        
        # Create a specific path for the best model that includes the model ID
        best_model_filename = f"best_bilstm_github_{best_model_id}.pth"
        best_model_path = os.path.join(os.getcwd(), best_model_filename)
        
        # Check if we already have this specific model
        if os.path.exists(best_model_path):
            logging.info(f"Best model already exists at {best_model_path}")
        else:
            # Download the model from ClearML by ID
            best_model = Model(model_id=best_model_id)
            downloaded_path = best_model.get_local_copy()
            
            # If the downloaded path is different from our desired path, copy it
            if downloaded_path != best_model_path:
                shutil.copy2(downloaded_path, best_model_path)
                logging.info(f"Copied best model from {downloaded_path} to {best_model_path}")
            else:
                logging.info(f"Downloaded best model to {best_model_path}")
        
        # Verify the model file exists and has content
        if not os.path.exists(best_model_path) or os.path.getsize(best_model_path) == 0:
            logging.error(f"Best model file is missing or empty at {best_model_path}")
            raise FileNotFoundError(f"Best model file not found: {best_model_path}")
            
        # Verify model architecture by loading it
        try:
            import torch
            checkpoint = torch.load(best_model_path, map_location='cpu')
            logging.info(f"Successfully verified model file integrity. Model contains {len(checkpoint)} keys.")
        except Exception as e:
            logging.error(f"Failed to load model for verification: {e}")
            best_model = Model(model_id=best_model_id)
            best_model_path = best_model.get_local_copy()
            logging.warning(f"Re-downloaded model to {best_model_path} after verification failure")
    except Exception as e:
        logging.error(f"Failed to retrieve best model: {e}")
        best_model_path = ""  # Empty string if model download fails

    # Step 5: Evaluate best model
    logging.info("Starting model evaluation...")
    test_accuracy = evaluate_model_github(
        best_task_id=best_task_id,
        best_model_id=best_model_id,
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes
    )
    accuracy_value = float(test_accuracy) if hasattr(test_accuracy, '__float__') else test_accuracy
    logging.info(f"Evaluation completed. Test accuracy: {accuracy_value:.2f}%")

    # Clean up hyperparameter files before deployment - keep only the best model's hyperparameters
    logging.info("Cleaning up hyperparameter files...")
    try:
        deleted_count = cleanup_hyperparameter_files(best_task_id)
        logging.info(f"Removed {deleted_count} unnecessary hyperparameter files")
    except Exception as cleanup_error:
        logging.error(f"Error during hyperparameter cleanup: {cleanup_error}")
        logging.info("Continuing with deployment despite cleanup error")

    # Step 6: Deploy model if it meets threshold
    logging.info("Starting model deployment...")
    try:
        deployment_status = deploy_model_github(
            best_model_id=best_model_id,
            best_model_path=best_model_path,
            test_accuracy=accuracy_value,
            min_accuracy_threshold=85.0,  # Deploy if accuracy >= 85%
            mongo_uri=mongo_uri
        )
        logging.info(f"Deployment completed. Status: {deployment_status}")
    except Exception as e:
        logging.error(f"Deployment failed with error: {e}")
        deployment_status = "deployment_error"

    logging.info("Guardian GitHub Pipeline finished successfully.")
    return accuracy_value, deployment_status

# ============================================================================
# TEST FUNCTION FOR HYPERPARAMETER PROCESSING
# ============================================================================

def test_hyperparameter_processing():
    """Test the hyperparameter processing functions to ensure they handle various formats correctly."""
    print("🧪 Testing hyperparameter processing...")
    
    # Create test functions (copying from deploy function)
    def safe_extract_hyperparameter_value(param_value, default_value):
        """Safely extract hyperparameter value from various formats."""
        if param_value is None:
            return default_value
        
        # If it's already a basic type (int, float, bool, str), return it
        if isinstance(param_value, (int, float, bool, str)):
            return param_value
        
        # If it's a dict with 'value' key (ClearML format)
        if isinstance(param_value, dict):
            if 'value' in param_value:
                return param_value['value']
            # If it's just a regular dict, try to extract the value
            elif len(param_value) == 1:
                return list(param_value.values())[0]
            # Empty dict should return default value
            elif len(param_value) == 0:
                return default_value
            else:
                return param_value
        
        # If it's a list, take the first element
        if isinstance(param_value, list) and len(param_value) > 0:
            return param_value[0]
        
        # Default fallback
        return default_value
    
    # Test various input formats
    test_cases = [
        # (input, expected_output, description)
        (256, 256, "Direct integer"),
        ("test_string", "test_string", "Direct string"),
        (True, True, "Direct boolean"),
        (0.001, 0.001, "Direct float"),
        ({"value": 128}, 128, "ClearML dict format"),
        ({"some_key": 512}, 512, "Single-key dict"),
        ([256], 256, "List with single element"),
        (None, 999, "None value with default"),
        ({}, 999, "Empty dict with default"),
        ([], 999, "Empty list with default"),
    ]
    
    all_passed = True
    for input_val, expected, description in test_cases:
        result = safe_extract_hyperparameter_value(input_val, 999)
        if result != expected:
            print(f"❌ FAILED: {description} - Expected {expected}, got {result}")
            all_passed = False
        else:
            print(f"✅ PASSED: {description}")
    
    if all_passed:
        print("🎉 All hyperparameter processing tests passed!")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return all_passed

# ============================================================================
# CLEAN UP UTILITY
# ============================================================================

def cleanup_hyperparameter_files(best_task_id):
    """
    Remove all hyperparameter files except the one corresponding to the best model.
    This ensures the artifacts folder stays clean.
    
    Args:
        best_task_id (str): The ID of the best task/model to keep
    """
    import os
    import glob
    
    # Define patterns for hyperparam files
    patterns = [
        "hyperparams_*.json",
        "best_hyperparams_*.json"
    ]
    
    # The files we want to keep
    best_hyperparams_filename = f"best_hyperparams_{best_task_id}.json"
    hyperparams_filename = f"hyperparams_{best_task_id}.json"
    standard_best_filename = "best_hyperparams.json"  # Always keep this one
    
    # Rename the best hyperparams file to a standard name if it exists
    if os.path.exists(best_hyperparams_filename) and not os.path.exists(standard_best_filename):
        try:
            import shutil
            shutil.copy2(best_hyperparams_filename, standard_best_filename)
            print(f"✅ Copied {best_hyperparams_filename} to {standard_best_filename}")
        except Exception as e:
            print(f"⚠️ Could not copy {best_hyperparams_filename} to {standard_best_filename}: {e}")
    
    # Rename regular hyperparams file if it exists
    if os.path.exists(hyperparams_filename):
        try:
            os.rename(hyperparams_filename, "model_hyperparams.json")
            print(f"✅ Renamed {hyperparams_filename} to model_hyperparams.json")
        except Exception as e:
            print(f"⚠️ Could not rename {hyperparams_filename}: {e}")
    
    # Count deleted files
    deleted_count = 0
    
    # Find and delete all other hyperparameter files
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            # Skip the files we want to keep
            if file_path == best_hyperparams_filename or file_path == hyperparams_filename:
                continue
                
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Could not delete {file_path}: {e}")
    
    print(f"🧹 Cleanup completed: Removed {deleted_count} unnecessary hyperparameter files")
    return deleted_count

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    logging.info("Running Guardian 🦾 pipeline for GitHub Actions...")
    
    # Test hyperparameter processing before starting
    test_passed = test_hyperparameter_processing()
    if not test_passed:
        logging.error("Hyperparameter processing tests failed! Exiting.")
        sys.exit(1)
    
    # Run locally for GitHub Actions
    PipelineDecorator.run_locally()
    
    # Start the pipeline execution
    result = guardian_github_pipeline()
    
    # Calculate execution time
    finish_time = time.time()
    elapsed_time = (finish_time - start_time) / 60

    print(f"\n🎉 Guardian AI GitHub Pipeline Completed! 🎉")
    print(f"⏱️  Total Execution Time: {elapsed_time:.2f} minutes")
    print(f"🎯 Final Test Accuracy: {result[0]:.2f}%")
    print(f"🚀 Deployment Status: {result[1]}")
    
    if result[1] == "deployed":
        print(f"✅ Model successfully deployed to production!")
    else:
        print(f"⚠️  Model not deployed (accuracy below threshold)") 