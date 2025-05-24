import torch
import torch.nn as nn
import os
from clearml import Dataset
import pathlib
import logging

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



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_clearml_dataset_locally_available(dataset_name: str, project_name: str, local_directory: str) -> pathlib.Path | None:
    """
    Ensures a ClearML dataset is available locally. Downloads it if it doesn't exist
    or if the local version doesn't have the expected structure.

    Args:
        dataset_name: Name of the dataset in ClearML.
        project_name: Name of the project containing the dataset in ClearML.
        local_directory: Path to the local directory where the dataset should reside.

    Returns:
        pathlib.Path | None: The path to the local dataset directory if it's available
                             and has the required structure, otherwise None.
    """
    local_path = pathlib.Path(local_directory)
    required_subdirectories = {"train", "valid", "test"}

    logging.info(f"Checking local availability of ClearML dataset '{dataset_name}'...")

    # Check if the base directory exists and contains the required subdirectories
    if local_path.is_dir() and all((local_path / subdir).is_dir() for subdir in required_subdirectories):
        logging.info(f"Dataset '{dataset_name}' found locally with the expected structure at '{local_path}'.")
        return local_path
    else:
        logging.info(f"Local dataset '{dataset_name}' not found or missing required structure. Attempting download from ClearML...")
        try:
            dataset = Dataset.get(
                dataset_name=dataset_name,
                dataset_project=project_name,
                only_completed=True
            )
            if not dataset:
                logging.error(f"ClearML dataset '{dataset_name}' not found in project '{project_name}'.")
                return None

            logging.info(f"Downloading ClearML dataset '{dataset_name}' to '{local_path}'...")
            downloaded_path = dataset.get_local_copy(local_path=str(local_path))

            # Verify the downloaded structure
            downloaded_path_obj = pathlib.Path(downloaded_path)
            if all((downloaded_path_obj / subdir).is_dir() for subdir in required_subdirectories):
                logging.info(f"Successfully downloaded and verified dataset '{dataset_name}' to '{downloaded_path}'.")
                return downloaded_path_obj
            else:
                logging.error(f"Downloaded dataset '{dataset_name}' is missing the required subdirectories: {required_subdirectories}.")
                # Clean up incomplete download
                if downloaded_path_obj.is_dir():
                    shutil.rmtree(downloaded_path)
                return None

        except Exception as e:
            logging.error(f"An error occurred while trying to download dataset '{dataset_name}': {e}")
            return None
