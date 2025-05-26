# Guardian AI Training Pipeline

A comprehensive ClearML-powered training pipeline for human action recognition using BiLSTM with attention mechanism.

## 🎯 Overview

This repository contains the Guardian AI training pipeline that recognizes human actions (Falling, No Action, Waving) from pose keypoint data using:
- **BiLSTM with Attention** for temporal sequence modeling
- **ClearML** for experiment tracking and dataset management
- **Hyperparameter Optimization** for automated model tuning
- **GitHub Actions** with self-hosted GPU runners for CI/CD

## 🚀 Quick Start

### Prerequisites
- Self-hosted GitHub Actions runner with GPU
- ClearML account and API credentials
- Python 3.11+
- CUDA-compatible GPU

### Setup

1. **Configure GitHub Secrets**:
   ```
   CLEARML_API_ACCESS_KEY
   CLEARML_API_SECRET_KEY
   CLEARML_API_HOST
   ```

2. **Set up Self-hosted Runner**:
   - Install GitHub Actions runner on your GPU machine
   - Add labels: `[self-hosted, guardian, gpu]`
   - Ensure GPU drivers and CUDA are installed

3. **Trigger Pipeline**:
   - Push to `main` branch
   - Close a pull request to `main`
   - Manual trigger via GitHub Actions "Run workflow" button

## 📊 Pipeline Components

### Core Pipeline Files
- **`Guardian_pipeline.py`** - Main local training pipeline
- **`Guardian_pipeline_github.py`** - GitHub Actions optimized pipeline

### What the Pipeline Does

1. **Downloads Dataset** from ClearML (Guardian_Dataset)
2. **Preprocesses Data** - pose keypoint sequences with normalization
3. **Trains BiLSTM Model** with attention mechanism
4. **Optimizes Hyperparameters** - 50 trials with ClearML HPO
5. **Evaluates Best Model** with visualizations and metrics
6. **Publishes Model** to ClearML for deployment

## 🔧 Key Features

- **Automated Training**: No manual intervention required
- **GPU Acceleration**: Runs on your powerful local hardware
- **Experiment Tracking**: All results logged to ClearML
- **Hyperparameter Optimization**: Finds best model configuration
- **Rich Visualizations**: Attention maps, confusion matrices, training curves
- **CI/CD Integration**: Automated pipeline execution via GitHub Actions

## 📈 Results & Outputs

- Training metrics and loss curves
- Confusion matrix visualization
- Attention analysis plots
- Model artifacts (.pth files)
- Complete experiment tracking in ClearML
- Published model ID for deployment

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run local pipeline
python Guardian_pipeline.py

# Run GitHub-optimized pipeline locally
python Guardian_pipeline_github.py
```

## 📁 Repository Structure

```
├── Guardian_pipeline.py              # Main local training pipeline
├── Guardian_pipeline_github.py       # GitHub Actions optimized pipeline
├── requirements.txt                  # Python dependencies
├── setup.py                          # ClearML setup helper
├── clearml.conf                      # ClearML configuration
├── .github/
│   └── workflows/
│       └── guardian-pipeline-aws.yml # GitHub Actions workflow
├── actions-runner/                   # Self-hosted runner setup
├── training_utils/                   # Training utility modules
├── data/                             # Local dataset directory
└── README.md                         # This file
```

## 🔄 GitHub Actions Workflow

The repository includes a complete CI/CD pipeline that:
- Automatically triggers on code changes
- Runs on self-hosted GPU runners
- Executes the Guardian pipeline
- Uploads results and artifacts
- Provides detailed logging and monitoring

## 🔗 Monitoring & Results

- **ClearML Dashboard**: https://app.clear.ml/
- **GitHub Actions**: Check the Actions tab for pipeline runs
- **Artifacts**: Download training outputs from GitHub Actions
- **Model Registry**: Published models available in ClearML

## 🧹 Repository Maintenance

To clean up unnecessary files and keep only Guardian pipeline essentials:

```bash
python cleanup_repository.py
```

This will remove temporary model files, test scripts, and other non-essential components while preserving the core Guardian pipeline functionality.

---

**Guardian AI: Protecting through Intelligence** 🛡️
