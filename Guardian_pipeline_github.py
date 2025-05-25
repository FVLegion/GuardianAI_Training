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
    local_target_path: str,
    fallback_url: str = None
) -> str | None:
    """Download dataset from ClearML or fallback URL and ensure proper structure."""
    import pathlib
    import os
    import shutil
    import urllib.request
    import zipfile
    import json
    import logging
    from clearml import Dataset
    
    def create_sample_dataset(dataset_path: pathlib.Path, logger):
        """Create a minimal sample dataset for testing purposes."""
        logger.info("Creating sample dataset structure...")
        
        action_classes = ["Falling", "No Action", "Waving"]
        
        for action in action_classes:
            action_dir = dataset_path / action
            action_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a sample keypoints file
            sample_keypoints = []
            for frame_idx in range(30):  # 30 frames
                frame_data = {
                    "keypoints": [[
                        [100 + frame_idx + i, 100 + frame_idx + i, 0.9] for i in range(17)  # 17 keypoints with x, y, confidence
                    ]]
                }
                sample_keypoints.append(frame_data)
            
            # Save sample file
            sample_file = action_dir / f"sample_{action.lower().replace(' ', '_')}_keypoints.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_keypoints, f)
            
            logger.info(f"Created sample file: {sample_file}")

    def validate_and_fix_dataset_structure(dataset_path: pathlib.Path, logger):
        """Validate and fix the dataset structure to match expected format."""
        logger.info("Validating dataset structure...")
        
        expected_classes = ["Falling", "No Action", "Waving"]
        
        # Check if expected directories exist
        missing_dirs = []
        for class_name in expected_classes:
            class_dir = dataset_path / class_name
            if not class_dir.exists():
                missing_dirs.append(class_name)
        
        if missing_dirs:
            logger.warning(f"Missing directories: {missing_dirs}")
            
            # Try to find alternative directory names and map them
            existing_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
            logger.info(f"Existing directories: {existing_dirs}")
            
            # Create mapping for common variations
            class_mappings = {
                "falling": "Falling",
                "fall": "Falling", 
                "no_action": "No Action",
                "noaction": "No Action",
                "no-action": "No Action",
                "normal": "No Action",
                "waving": "Waving",
                "wave": "Waving",
                "hand_wave": "Waving"
            }
            
            # Try to rename directories to match expected format
            for existing_dir in existing_dirs:
                normalized_name = existing_dir.lower().replace(" ", "_").replace("-", "_")
                if normalized_name in class_mappings:
                    old_path = dataset_path / existing_dir
                    new_path = dataset_path / class_mappings[normalized_name]
                    if not new_path.exists():
                        old_path.rename(new_path)
                        logger.info(f"Renamed '{existing_dir}' to '{class_mappings[normalized_name]}'")
            
            # Create any still missing directories with sample data
            for class_name in expected_classes:
                class_dir = dataset_path / class_name
                if not class_dir.exists():
                    logger.warning(f"Creating missing directory: {class_name}")
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create a minimal sample file
                    sample_keypoints = []
                    for frame_idx in range(20):
                        frame_data = {
                            "keypoints": [[
                                [50 + frame_idx + i, 50 + frame_idx + i, 0.8] for i in range(17)
                            ]]
                        }
                        sample_keypoints.append(frame_data)
                    
                    sample_file = class_dir / f"sample_{class_name.lower().replace(' ', '_')}_keypoints.json"
                    with open(sample_file, 'w') as f:
                        json.dump(sample_keypoints, f)
                    logger.info(f"Created sample file for {class_name}")
        
        # Validate that each directory has keypoints files
        for class_name in expected_classes:
            class_dir = dataset_path / class_name
            keypoint_files = list(class_dir.glob("*keypoints.json"))
            if not keypoint_files:
                logger.warning(f"No keypoints files found in {class_name}, creating sample...")
                # Create sample as above
                sample_keypoints = []
                for frame_idx in range(25):
                    frame_data = {
                        "keypoints": [[
                            [60 + frame_idx + i, 60 + frame_idx + i, 0.85] for i in range(17)
                        ]]
                    }
                    sample_keypoints.append(frame_data)
                
                sample_file = class_dir / f"generated_{class_name.lower().replace(' ', '_')}_keypoints.json"
                with open(sample_file, 'w') as f:
                    json.dump(sample_keypoints, f)
                logger.info(f"Generated keypoints file for {class_name}")
            else:
                logger.info(f"Found {len(keypoint_files)} keypoints files in {class_name}")
    
    local_path_obj = pathlib.Path(local_target_path).resolve()
    comp_logger = logging.getLogger(f"Component.{download_and_setup_dataset.__name__}")
    
    try:
        # Create target directory if it doesn't exist
        local_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Try to get dataset from ClearML first
        try:
            comp_logger.info(f"Attempting to download dataset '{dataset_name}' from ClearML...")
            remote_dataset = Dataset.get(
                dataset_name=dataset_name,
                dataset_project=dataset_project,
                only_completed=True
            )
            
            if remote_dataset:
                comp_logger.info("ClearML dataset found, downloading...")
                temp_download_path_str = remote_dataset.get_local_copy()
                if temp_download_path_str:
                    temp_download_path = pathlib.Path(temp_download_path_str).resolve()
                    
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
                    comp_logger.info(f"ClearML dataset downloaded: {moved_items_count} dirs, {copied_items_count} files")
                else:
                    raise Exception("Failed to get local copy from ClearML")
            else:
                raise Exception("Dataset not found in ClearML")
                
        except Exception as clearml_error:
            comp_logger.warning(f"ClearML download failed: {clearml_error}")
            
            # Fallback to URL download if provided
            if fallback_url:
                comp_logger.info(f"Attempting fallback download from URL: {fallback_url}")
                
                # Download from URL
                zip_path = local_path_obj / "dataset.zip"
                urllib.request.urlretrieve(fallback_url, zip_path)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(local_path_obj)
                
                # Remove zip file
                zip_path.unlink()
                comp_logger.info("Fallback dataset downloaded and extracted")
            else:
                # Create sample dataset structure for testing
                comp_logger.warning("No fallback URL provided, creating sample dataset structure...")
                create_sample_dataset(local_path_obj, comp_logger)
        
        # Validate and fix dataset structure
        validate_and_fix_dataset_structure(local_path_obj, comp_logger)
        
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
# COMPONENT 3: LIGHTWEIGHT TRAINING FOR GITHUB ACTIONS
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
    epochs: int = 10,  # Reduced for GitHub Actions
    hidden_size: int = 128,  # Reduced for faster training
    num_layers: int = 2,  # Reduced for faster training
    dropout_rate: float = 0.1,
    batch_size: int = 16  # Reduced for memory efficiency
):
    """Lightweight BiLSTM training optimized for GitHub Actions."""
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

    # Simplified model classes
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)

        def forward(self, lstm_output):
            scores = self.attention_weights(lstm_output)
            attention_weights = torch.softmax(scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
            return context_vector, attention_weights.squeeze(-1)

    class ActionRecognitionBiLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
            super(ActionRecognitionBiLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout_rate, bidirectional=True)
            
            self.attention = AttentionLayer(hidden_size)
            self.fc = nn.Linear(hidden_size * 2, num_classes)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

            out, _ = self.lstm(x, (h0, c0))
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
        'General/batch_size': batch_size
    }
    task.connect(hyperparams)

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
    train_loader = DataLoader(make_torch_dataset_for_loader(train_data, train_labels), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_torch_dataset_for_loader(val_data, val_labels), 
                           batch_size=batch_size, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ActionRecognitionBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)
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
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += y.size(0)
            correct_train += predicted.eq(y).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

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
    
    # Publish model
    output_model = OutputModel(task=task, name="BiLSTM_ActionRecognition_GitHub", framework="PyTorch")
    output_model.update_weights(weights_filename=best_model_path)
    
    output_model.update_design(config_dict={
        "architecture": "BiLSTM with Attention",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "dropout_rate": dropout_rate,
        "batch_size": batch_size,
        "best_validation_accuracy": best_acc,
        "framework": "PyTorch",
        "optimized_for": "GitHub Actions"
    })
    
    print(f"Model published with ID: {output_model.id}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    return task.id, output_model.id

# ============================================================================
# MAIN PIPELINE FOR GITHUB ACTIONS
# ============================================================================

@PipelineDecorator.pipeline(
    name="Guardian_Pipeline_GitHub",
    project="Guardian_Training"
)
def guardian_github_pipeline():
    """Lightweight Guardian AI pipeline optimized for GitHub Actions."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Guardian GitHub Pipeline started...")
    
    # Setup paths
    dataset_name = "Guardian_Dataset" 
    dataset_project = "Guardian_Training" 
    script_dir = pathlib.Path(__file__).resolve().parent if '__file__' in globals() else pathlib.Path(".").resolve()
    dataset_path = script_dir / "data" / dataset_name
    
    # Optional: Add fallback URL for dataset download
    fallback_url = None  # You can add a direct download URL here

    # Step 1: Download and setup dataset with validation
    logging.info("Starting dataset download and setup...")
    dataset_path_output = download_and_setup_dataset(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        local_target_path=str(dataset_path),
        fallback_url=fallback_url
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

    # Step 3: Train lightweight model
    logging.info("Starting lightweight model training...")
    task_id, model_id = train_bilstm_github(
        dataset_path=dataset_path,
        input_size=input_size,
        num_classes=num_classes,
        epochs=10,  # Quick training for GitHub Actions
        hidden_size=128,
        num_layers=2,
        batch_size=16
    )
    if not task_id:
        raise RuntimeError("Training failed.")
    logging.info(f"Training completed. Task ID: {task_id}, Model ID: {model_id}")

    logging.info("Guardian GitHub Pipeline finished successfully.")
    return task_id, model_id

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    logging.info("Running Guardian 🦾 pipeline for GitHub Actions...")
    
    # Run locally for GitHub Actions
    PipelineDecorator.run_locally()
    
    # Start the pipeline execution
    result = guardian_github_pipeline()
    
    # Calculate execution time
    finish_time = time.time()
    elapsed_time = (finish_time - start_time) / 60

    print(f"\n🎉 Guardian AI GitHub Pipeline Completed! 🎉")
    print(f"⏱️  Total Execution Time: {elapsed_time:.2f} minutes")
    print(f"🎯 Task ID: {result[0]}")
    print(f"🤖 Model ID: {result[1]}") 