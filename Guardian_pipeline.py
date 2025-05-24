from clearml import PipelineDecorator, Dataset, Task, OutputModel, Model
import os
import pathlib
import logging
import shutil
import sys
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Record start time
start_time = time.time()

# ============================================================================
# COMPONENT 1: DATASET MANAGEMENT
# ============================================================================

@PipelineDecorator.component(return_values=["dataset_path"], cache=True, execution_queue="default")
def download_and_verify_clearml_dataset(
    dataset_name: str,
    dataset_project: str,
    local_target_path: str
) -> str | None:
    """Download and verify ClearML dataset with smart caching."""
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

# ============================================================================
# COMPONENT 2: DATA PREPARATION
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
                        print(f"Warning: Directory not found: {action_dir}")
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
                            except (json.JSONDecodeError, FileNotFoundError) as e:
                                print(f"Error loading or processing {filepath}: {e}")
                                continue

                return data, labels

            def process_keypoints(self, keypoints_data):
                all_frames_keypoints = []
                previous_frame = None
                alpha = 0.8

                for frame_data in keypoints_data:
                    if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                        print(f"Skipping invalid frame data: {frame_data}")
                        continue

                    frame_keypoints = frame_data['keypoints']
                    if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                        print("frame keypoints is not a list or is empty")
                        continue

                    frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                    if frame_keypoints_np.shape != (17, 3):
                        print(f"Incorrect shape: {frame_keypoints_np.shape}")
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
# COMPONENT 3: MODEL TRAINING
# ============================================================================

@PipelineDecorator.component(
    name="Train_BiLSTM",
    return_values=["task_id", "model_id"],
    packages=["torch>=1.9", "clearml", "scikit-learn", "numpy", "matplotlib"],
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
    """Train a BiLSTM model with comprehensive logging and return task ID and model ID."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from clearml import Task, OutputModel
    import numpy as np
    import json
    import os
    import matplotlib.pyplot as plt
    
    # Embedded model classes
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)

        def forward(self, lstm_output):
            scores = self.attention_weights(lstm_output)
            attention_weights = torch.softmax(scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
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
                    print(f"Warning: Directory not found: {action_dir}")
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
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"Error loading or processing {filepath}: {e}")
                            continue

            return data, labels

        def process_keypoints(self, keypoints_data):
            all_frames_keypoints = []
            previous_frame = None
            alpha = 0.8

            for frame_data in keypoints_data:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    print(f"Skipping invalid frame data: {frame_data}")
                    continue

                frame_keypoints = frame_data['keypoints']
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                    print("frame keypoints is not a list or is empty")
                    continue

                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    print(f"Incorrect shape: {frame_keypoints_np.shape}")
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
    
    # Initialize task
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="Guardian_Training",
            task_name="Train_BiLSTM_Enhanced"
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
        'General/num_classes': num_classes
    }
    task.connect(hyperparams)

    # Log model architecture details
    total_params = (input_size * hidden_size * 4 + hidden_size * hidden_size * 8 * num_layers + 
                   hidden_size * 2 * num_classes)
    logger.report_single_value("Total Parameters", total_params)
    logger.report_single_value("Hidden Size", hidden_size)
    logger.report_single_value("Number of Layers", num_layers)

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

    # Log dataset statistics
    logger.report_single_value("Total Samples", len(dataset.data))
    logger.report_single_value("Training Samples", len(train_data))
    logger.report_single_value("Validation Samples", len(val_data))
    logger.report_single_value("Test Samples", len(test_data))
    
    # Log class distribution
    unique_labels, counts = np.unique(dataset.labels, return_counts=True)
    for i, (label, count) in enumerate(zip(unique_labels, counts)):
        class_name = action_classes[label].replace(" ", "_")
        logger.report_single_value(f"Class_Count_{class_name}", count)
        logger.report_text(f"Class {action_classes[label]}: {count} samples")

    def make_torch_dataset_for_loader(split_data, split_labels):
        temp_ds = PoseDataset(data_dir=dataset_path, action_classes=action_classes)
        temp_ds.data = split_data
        temp_ds.labels = split_labels
        return temp_ds

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(make_torch_dataset_for_loader(train_data, train_labels), 
                             batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_torch_dataset_for_loader(val_data, val_labels), 
                           batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.report_text(f"Using device: {device}")
    print(f"Using device: {device}")
    
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_acc = 0.0
    best_model_path = f"best_model_{task.id}.pt"
    train_losses, val_losses, val_accuracies = [], [], []
    learning_rates = []

    print(f"Starting training on {device} with {len(train_data)} training samples...")

    # Training loop with enhanced logging
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, attention_weights = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate training accuracy
            preds = outputs.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs, attention_weights = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y.cpu().numpy())
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_val_labels, all_val_preds, average=None, zero_division=0
        )
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        # Log comprehensive metrics
        logger.report_scalar("Loss", "Train", avg_train_loss, epoch)
        logger.report_scalar("Loss", "Validation", avg_val_loss, epoch)
        logger.report_scalar("Accuracy", "Train", train_acc, epoch)
        logger.report_scalar("Accuracy", "Validation", val_acc, epoch)
        logger.report_scalar("Learning Rate", "Current", current_lr, epoch)
        
        # Log per-class metrics
        for i, class_name in enumerate(action_classes):
            if i < len(precision):
                logger.report_scalar("Precision", class_name, precision[i], epoch)
                logger.report_scalar("Recall", class_name, recall[i], epoch)
                logger.report_scalar("F1-Score", class_name, f1[i], epoch)
        
        # Also log with different format for HPO compatibility
        logger.report_scalar("metrics", "Validation_Accuracy", val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Log best metrics
            logger.report_single_value("Best_Validation_Accuracy", val_acc)
            logger.report_single_value("Best_Epoch", epoch + 1)
    
    # Generate training plots
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='orange')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    logger.report_matplotlib_figure("Training Metrics", "Overview", plt.gcf(), 0)
    plt.close()
    
    # Publish model with comprehensive metadata
    output_model = OutputModel(task=task, name="BiLSTM_ActionRecognition_Enhanced", framework="PyTorch")
    output_model.update_weights(weights_filename=best_model_path)
    
    # Add model metadata
    model_metadata = {
        "best_validation_accuracy": float(best_acc),
        "total_epochs": epochs,
        "final_learning_rate": float(current_lr),
        "model_architecture": {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_classes": num_classes,
            "dropout_rate": dropout_rate
        },
        "training_samples": len(train_data),
        "validation_samples": len(val_data)
    }
    
    # Upload training artifacts
    task.upload_artifact("training_metrics", "training_metrics.png")
    task.upload_artifact("model_metadata", artifact_object=model_metadata)
    task.upload_artifact("best_model_weights", best_model_path)
    
    output_model.publish()
    
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model published with ID: {output_model.id}")
    
    return task.id, output_model.id

# ============================================================================
# COMPONENT 4: HYPERPARAMETER OPTIMIZATION
# ============================================================================

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
            DiscreteParameterRange('General/base_lr', values=[0.0008, 0.0009, 0.001, 0.0011, 0.0012]),  # Denser around 0.001
            DiscreteParameterRange('General/hidden_size', values=[120, 128, 136, 144, 152]),  # Denser around 128-140
            DiscreteParameterRange('General/num_layers', values=[1, 2, 3]),  # Added single layer option
            DiscreteParameterRange('General/dropout_rate', values=[0.1, 0.15, 0.2, 0.25, 0.3]),  # Denser between 0.1-0.3
            DiscreteParameterRange('General/epochs', values=[20, 25, 30])  # Added intermediate value
        ],
        # Objective metric we want to maximize
        objective_metric_title="Accuracy",
        objective_metric_series="Validation",
        objective_metric_sign="max",
        # Increase concurrent experiments since we have more combinations
        max_number_of_concurrent_tasks=2,
        # Use GridSearch instead of Optuna
        optimizer_class=GridSearch,
        # Keep more top tasks for analysis
        save_top_k_tasks_only=5,
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

# ============================================================================
# COMPONENT 5: MODEL EVALUATION
# ============================================================================

@PipelineDecorator.component(
    name="Evaluate_Model",
    return_values=["test_accuracy"],
    cache=False,
    packages=["torch", "scikit-learn", "numpy", "clearml", "matplotlib", "seaborn"]
)
def evaluate_model(
    best_task_id: str,
    best_model_id: str,
    dataset_path: str,
    input_size: int = 34,
    num_classes: int = 3
):
    """Evaluate the best BiLSTM model on the test set with comprehensive analysis."""
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
    
    # Embedded model classes
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.attention_weights = nn.Linear(hidden_size * 2, 1)

        def forward(self, lstm_output):
            scores = self.attention_weights(lstm_output)
            attention_weights = torch.softmax(scores, dim=1)
            context_vector = torch.sum(attention_weights * lstm_output, dim=1)
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
                    print(f"Warning: Directory not found: {action_dir}")
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
                        except (json.JSONDecodeError, FileNotFoundError) as e:
                            print(f"Error loading or processing {filepath}: {e}")
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

    def visualize_spatial_saliency(model, input_sequence, predicted_class, frame_indices, skeleton_connections, raw_keypoints, save_dir):
        """Generate spatial saliency maps for pose keypoints."""
        input_sequence = input_sequence.clone().detach().requires_grad_(True)
        
        logits, attn = model(input_sequence)
        score = logits[0, predicted_class]
        
        model.zero_grad()
        score.backward()

        grads = input_sequence.grad.data.abs().squeeze(0)
        grads = grads.view(grads.size(0), 17, 2).mean(dim=2)

        os.makedirs(save_dir, exist_ok=True)

        for t in frame_indices:
            kp_scores = grads[t].cpu().numpy()
            kps = raw_keypoints[t]

            norm = (kp_scores - kp_scores.min()) / (kp_scores.ptp() + 1e-8)

            plt.figure(figsize=(8, 8))
            for idx, (x, y) in enumerate(kps):
                plt.scatter(x, y, s=(norm[idx]*300)+20, c=norm[idx], cmap='hot', alpha=0.8)
                plt.text(x, y, str(idx), fontsize=8, ha='center', va='center')
            
            for i, j in skeleton_connections:
                if i < len(kps) and j < len(kps):
                    x1, y1 = kps[i]
                    x2, y2 = kps[j]
                    plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.6)
            
            plt.title(f"Spatial Saliency - Frame {t}, Class {predicted_class}", fontsize=14)
            plt.colorbar(label='Saliency Score')
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{save_dir}/spatial_saliency_frame_{t}.png", dpi=150, bbox_inches='tight')
            plt.close()

    # Initialize task
    task = Task.init(
        project_name="Guardian_Training",
        task_name="Evaluate_BiLSTM_Enhanced"
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
    
    # Get parameters from best task - Fix the TypeError
    best_task = Task.get_task(task_id=best_task_id)
    try:
        # Try the new API first
        params = best_task.get_parameters_as_dict()
        if 'General' in params:
            params = params['General']
        else:
            # Flatten manually if needed
            flat_params = {}
            for section, section_params in params.items():
                if isinstance(section_params, dict):
                    for key, value in section_params.items():
                        flat_params[f"{section}/{key}"] = value
                else:
                    flat_params[section] = section_params
            params = flat_params
    except Exception as e:
        print(f"Error getting parameters: {e}")
        # Use default parameters
        params = {
            'hidden_size': 256,
            'num_layers': 4,
            'dropout_rate': 0.1
        }

    # Extract hyperparameters, handling both formats
    hidden_size = int(params.get('hidden_size', params.get('General/hidden_size', 256)))
    num_layers = int(params.get('num_layers', params.get('General/num_layers', 4)))
    dropout_rate = float(params.get('dropout_rate', params.get('General/dropout_rate', 0.1)))

    print(f"Using parameters: hidden_size={hidden_size}, num_layers={num_layers}, dropout_rate={dropout_rate}")

    # Get model
    if best_model_id == "no_model_found":
        print("No model found from HPO, skipping evaluation")
        return 0.0
    
    try:
        model_obj = Model(model_id=best_model_id)
        model_path = model_obj.get_local_copy()
        
        # Try to load the model state dict to inspect its architecture
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Infer model architecture from checkpoint
        if 'lstm.weight_ih_l0' in checkpoint:
            # Extract architecture info from the checkpoint
            lstm_input_size = checkpoint['lstm.weight_ih_l0'].shape[1]
            lstm_hidden_size = checkpoint['lstm.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
            
            # Count LSTM layers by checking for layer-specific weights
            inferred_num_layers = 1
            layer_idx = 1
            while f'lstm.weight_ih_l{layer_idx}' in checkpoint:
                inferred_num_layers += 1
                layer_idx += 1
            
            # Get attention layer size to infer hidden size
            if 'attention.attention_weights.weight' in checkpoint:
                attention_input_size = checkpoint['attention.attention_weights.weight'].shape[1]
                inferred_hidden_size = attention_input_size // 2  # BiLSTM outputs hidden_size * 2
            else:
                inferred_hidden_size = lstm_hidden_size
            
            print(f"Inferred from checkpoint: hidden_size={inferred_hidden_size}, num_layers={inferred_num_layers}")
            
            # Use inferred parameters if they differ significantly
            if abs(inferred_hidden_size - hidden_size) > 10 or inferred_num_layers != num_layers:
                print(f"Architecture mismatch detected. Using inferred parameters.")
                hidden_size = inferred_hidden_size
                num_layers = inferred_num_layers
        
    except Exception as e:
        print(f"Error retrieving or inspecting model {best_model_id}: {e}")
        print("Skipping evaluation due to model retrieval failure")
        return 0.0

    # Load model with correct architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    try:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully with inferred architecture")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        print("Attempting to load with strict=False")
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("Model loaded with strict=False (some parameters may be missing)")
        except Exception as e2:
            print(f"Failed to load model even with strict=False: {e2}")
            return 0.0
    
    model.eval()

    # Skeleton connections for visualization
    skeleton_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                           (0, 7), (7, 8), (8, 9), (7, 10), (10, 11), (11, 12),
                           (8, 13), (13, 14), (14, 15)]

    # Evaluate with comprehensive analysis
    all_preds, all_labels = [], []
    attention_weights_all = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            outputs, attention_weights = model(x)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            attention_weights_all.extend(attention_weights.cpu().numpy())
            
            # Generate saliency maps for first few samples
            if sample_count < 5:
                for i in range(min(3, x.size(0))):
                    raw_kps = x[i].detach().cpu().numpy().reshape(-1, 17, 2)
                    top_frame = attention_weights[i].argmax().item()
                    
                    model.train()  # Enable gradients
                    with torch.enable_grad():
                        visualize_spatial_saliency(
                            model, x[i:i+1], preds[i].item(), [top_frame],
                            skeleton_connections, raw_kps, f"evaluation_outputs/saliency_sample_{sample_count}"
                        )
                    model.eval()
                    sample_count += 1
    
    test_accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Log comprehensive metrics
    logger.report_scalar("Accuracy", "Test", test_accuracy, 0)
    
    # Generate and log confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=action_classes, yticklabels=action_classes)
    plt.title('Confusion Matrix - Test Set', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    logger.report_matplotlib_figure("Confusion Matrix", "Test Set", plt.gcf(), 0)
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=action_classes, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=action_classes)
    
    # Log per-class metrics
    for class_name in action_classes:
        if class_name in report:
            logger.report_scalar("Precision", class_name, report[class_name]['precision'], 0)
            logger.report_scalar("Recall", class_name, report[class_name]['recall'], 0)
            logger.report_scalar("F1-Score", class_name, report[class_name]['f1-score'], 0)
    
    # Log overall metrics
    logger.report_scalar("Metrics", "Macro Avg F1", report['macro avg']['f1-score'], 0)
    logger.report_scalar("Metrics", "Weighted Avg F1", report['weighted avg']['f1-score'], 0)
    
    # Attention analysis
    attention_weights_all = np.array(attention_weights_all)
    avg_attention = np.mean(attention_weights_all, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_attention, marker='o', linewidth=2, markersize=6)
    plt.title('Average Attention Weights Across Test Set', fontsize=16)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=150, bbox_inches='tight')
    logger.report_matplotlib_figure("Attention Analysis", "Average Weights", plt.gcf(), 0)
    plt.close()
    
    # Upload artifacts
    task.upload_artifact("confusion_matrix", "confusion_matrix.png")
    task.upload_artifact("attention_analysis", "attention_analysis.png")
    task.upload_artifact("classification_report", artifact_object=report_str)
    
    logger.report_text(f"Classification Report:\n{report_str}", print_console=True)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    # Return as float to avoid LazyEvalWrapper issues
    return float(test_accuracy)

# ============================================================================
# MAIN PIPELINE DEFINITION
# ============================================================================

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
    
    # Convert to float to avoid LazyEvalWrapper formatting issues
    accuracy_value = float(test_accuracy) if hasattr(test_accuracy, '__float__') else test_accuracy
    logging.info(f"Evaluation step completed. Test accuracy: {accuracy_value:.2f}%")

    logging.info("Guardian Pipeline finished successfully.")
    return accuracy_value

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    logging.info("Running Guardian 🦾 pipeline locally...")
    
    # run_locally() executes the pipeline defined by decorators in the current process
    # This allows it to appear in PIPELINES section while running locally
    PipelineDecorator.run_locally()
    
    # Start the pipeline execution
    result = guardian_training_pipeline()
    
    # Calculate execution time
    finish_time = time.time()
    elapsed_time = (finish_time - start_time) / 60  # Convert to minutes

    print(f"\n🎉 Guardian AI Pipeline Completed Successfully! 🎉")
    print(f"⏱️  Total Execution Time: {elapsed_time:.2f} minutes")
    print(f"🎯 Final Test Accuracy: {result:.2f}%")
    
