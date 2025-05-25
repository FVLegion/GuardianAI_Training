# Guardian AI Training Pipeline

A simple, clean ClearML-powered training pipeline for human action recognition using BiLSTM with attention mechanism.

## 🎯 Overview

This pipeline trains a deep learning model to recognize human actions (Falling, No Action, Waving) from pose keypoint data using:
- **BiLSTM with Attention** for temporal sequence modeling
- **ClearML** for experiment tracking and dataset management
- **Hyperparameter Optimization** for automated model tuning
- **Self-hosted GPU runner** for efficient training

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

## 📊 What the Pipeline Does

1. **Downloads Dataset** from ClearML (Guardian_Dataset)
2. **Preprocesses Data** - pose keypoint sequences with normalization
3. **Trains BiLSTM Model** with attention mechanism
4. **Optimizes Hyperparameters** - 50 trials with ClearML HPO
5. **Evaluates Best Model** with visualizations and metrics

## 🔧 Key Features

- **Automated Training**: No manual intervention required
- **GPU Acceleration**: Runs on your powerful local hardware
- **Experiment Tracking**: All results logged to ClearML
- **Hyperparameter Optimization**: Finds best model configuration
- **Rich Visualizations**: Attention maps, confusion matrices, training curves

## 📈 Results & Outputs

- Training metrics and loss curves
- Confusion matrix visualization
- Attention analysis plots
- Model artifacts (.pth files)
- Complete experiment tracking in ClearML

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline locally (optional)
python Guardian_pipeline.py
```

## 📁 Project Structure

```
├── Guardian_pipeline.py              # Main pipeline script
├── requirements.txt                  # Python dependencies
├── upload_dataset.py                 # Dataset upload utility
├── setup.py                          # ClearML setup helper
├── .github/workflows/guardian-pipeline.yml  # GitHub Actions workflow
├── data/                             # Local dataset (if any)
└── evaluation_outputs/               # Generated visualizations
```

## 🔗 Monitoring

- **ClearML Dashboard**: https://app.clear.ml/
- **GitHub Actions**: Check the Actions tab for pipeline runs
- **Artifacts**: Download training outputs from GitHub Actions

---

**Simple. Clean. Effective.** 🎯
