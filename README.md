# Guardian AI Training Pipeline 🦾

A comprehensive machine learning pipeline for human action recognition using pose estimation data, built with ClearML for experiment tracking and hyperparameter optimization.

## 🎯 Overview

Guardian AI is an action recognition system that analyzes human pose keypoints to classify actions into three categories:
- **Falling** - Detecting fall incidents
- **No Action** - Normal standing/idle behavior  
- **Waving** - Hand waving gestures

The system uses a BiLSTM (Bidirectional Long Short-Term Memory) neural network with attention mechanism to process temporal sequences of pose keypoints.

## 🏗️ Architecture

### Model Architecture
- **BiLSTM with Attention**: Processes temporal sequences of pose keypoints
- **Input**: 34 features per frame (17 keypoints × 2 coordinates)
- **Attention Mechanism**: Focuses on important frames in the sequence
- **Output**: 3-class classification (Falling, No Action, Waving)

### Pipeline Components
1. **Dataset Management**: Downloads and verifies ClearML datasets
2. **Data Preparation**: Preprocesses pose keypoints with normalization and temporal smoothing
3. **Model Training**: Trains BiLSTM with comprehensive logging and metrics
4. **Hyperparameter Optimization**: GridSearch-based HPO for optimal parameters
5. **Model Evaluation**: Comprehensive evaluation with visualizations and saliency analysis

## 🚀 Features

### ✨ Advanced Training Features
- **Comprehensive Logging**: Detailed metrics tracking with ClearML
- **Hyperparameter Optimization**: GridSearch with configurable parameter ranges
- **Attention Visualization**: Spatial saliency maps showing important keypoints
- **Model Artifacts**: Automatic model versioning and metadata tracking
- **Training Visualizations**: Loss curves, accuracy plots, and learning rate schedules

### 📊 Evaluation & Analysis
- **Confusion Matrix**: Visual representation of classification performance
- **Per-Class Metrics**: Precision, recall, and F1-score for each action class
- **Attention Analysis**: Temporal attention weight visualization
- **Saliency Maps**: Spatial importance of pose keypoints
- **Classification Reports**: Detailed performance metrics

### 🔧 Technical Features
- **ClearML Integration**: Full experiment tracking and pipeline management
- **GPU Support**: Automatic CUDA detection and utilization
- **Robust Error Handling**: Graceful failure recovery and fallback mechanisms
- **Model Architecture Inference**: Automatic parameter detection from checkpoints
- **Flexible Data Loading**: Support for various dataset formats

## 📋 Requirements

### Dependencies
```
torch>=1.9
clearml
scikit-learn
numpy
matplotlib
seaborn
```

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- ClearML account and configuration

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd GuardianAI_Training
```

2. **Install dependencies**
```bash
pip install torch clearml scikit-learn numpy matplotlib seaborn
```

3. **Configure ClearML**
```bash
clearml-init
```
Follow the prompts to configure your ClearML credentials.

## 🎮 Usage

### Quick Start
```bash
python Guardian_pipeline.py
```

### Pipeline Configuration
The pipeline can be configured by modifying parameters in the main function:

```python
# Dataset configuration
dataset_name = "Guardian_Dataset"
dataset_project = "Guardian_Training"

# Training parameters
epochs = 50
hidden_size = 256
num_layers = 4
dropout_rate = 0.1
base_lr = 0.001

# HPO configuration
total_max_trials = 5
```

### Hyperparameter Search Space
The GridSearch explores the following parameter ranges:
- **Learning Rate**: [0.001, 0.003, 0.01]
- **Hidden Size**: [128, 256]
- **Number of Layers**: [2, 3]
- **Dropout Rate**: [0.1, 0.3, 0.5]
- **Epochs**: [20, 30]

## 📁 Project Structure

```
GuardianAI_Training/
├── Guardian_pipeline.py          # Main pipeline script with local and remote execution support
├── training_utils/               # Utility modules
│   ├── __init__.py
│   ├── dataset_utils.py         # Dataset handling utilities
│   └── model_utils.py           # Model architecture definitions
├── setupSteps.txt               # Setup instructions
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## 🔄 Pipeline Workflow

1. **Dataset Download & Verification**
   - Downloads dataset from ClearML
   - Verifies data integrity
   - Handles local caching

2. **Data Preparation**
   - Loads pose keypoint data
   - Applies normalization and temporal smoothing
   - Creates train/validation/test splits

3. **Baseline Training**
   - Trains initial BiLSTM model
   - Logs comprehensive metrics
   - Saves best model checkpoint

4. **Hyperparameter Optimization**
   - Runs GridSearch over parameter space
   - Tracks all experiments in ClearML
   - Selects best performing configuration

5. **Final Evaluation**
   - Evaluates best model on test set
   - Generates visualizations and reports
   - Creates saliency maps for interpretability

## 📊 Monitoring & Results

### ClearML Dashboard
The pipeline automatically logs to ClearML, providing:
- **Experiment Tracking**: All runs with parameters and metrics
- **Model Registry**: Versioned models with metadata
- **Pipeline Visualization**: DAG view of pipeline execution
- **Artifact Management**: Plots, models, and reports

### Key Metrics
- **Test Accuracy**: Final model performance
- **Per-Class F1-Score**: Balanced performance across actions
- **Training Time**: Pipeline execution duration
- **Model Size**: Parameter count and memory usage

## 🎨 Visualizations

The pipeline generates several types of visualizations:

1. **Training Metrics**: Loss curves and accuracy plots
2. **Confusion Matrix**: Classification performance heatmap
3. **Attention Analysis**: Temporal attention weight distribution
4. **Saliency Maps**: Spatial importance of pose keypoints
5. **Learning Rate Schedule**: Adaptive learning rate progression

## 🔧 Customization

### Adding New Actions
To add new action classes:
1. Update `action_classes` list in the pipeline
2. Ensure dataset contains corresponding labeled data
3. Adjust `num_classes` parameter accordingly

### Modifying Model Architecture
The BiLSTM architecture can be customized by:
- Changing hidden dimensions
- Adding/removing LSTM layers
- Modifying attention mechanism
- Adjusting dropout rates

### Custom Hyperparameter Ranges
Update the HPO component with new parameter ranges:
```python
DiscreteParameterRange('General/hidden_size', values=[64, 128, 256, 512])
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Architecture mismatch between training and evaluation
   - Solution: Pipeline automatically infers architecture from checkpoints

2. **Memory Issues**
   - Large batch sizes or model dimensions
   - Solution: Reduce batch size or use gradient accumulation

3. **ClearML Connection**
   - Network or authentication issues
   - Solution: Reconfigure with `clearml-init`

### Debug Mode
Enable verbose logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ClearML team for the excellent MLOps platform
- PyTorch community for the deep learning framework
- Contributors to the pose estimation datasets

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review ClearML logs for detailed error information
3. Open an issue on GitHub with relevant logs and configuration

---

**Guardian AI** - Protecting through intelligent action recognition 🛡️
