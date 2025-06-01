# Guardian AI Training Pipeline

A comprehensive ClearML-powered training pipeline for human action recognition using BiLSTM with attention mechanism.

## 🎯 Overview

This repository contains the Guardian AI training pipeline that recognizes human actions (Falling, No Action, Waving) from pose keypoint data using:
- **BiLSTM with Attention** for temporal sequence modeling
- **ClearML** for experiment tracking and dataset management
- **Hyperparameter Optimization** for automated model tuning
- **GitHub Actions** with self-hosted GPU runners for CI/CD
- **MongoDB** for model distribution and versioning

## 🚀 Quick Start

### Prerequisites
- Self-hosted GitHub Actions runner with GPU
- ClearML account and API credentials
- Python 3.11+
- CUDA-compatible GPU
- MongoDB Atlas account (for model distribution)

### Setup

1. **Configure GitHub Secrets**:
   ```
   CLEARML_API_ACCESS_KEY
   CLEARML_API_SECRET_KEY
   CLEARML_API_HOST
   MONGODB_URI
   MONGODB_DATABASE
   ```

2. **Set up Self-hosted Runner**:
   - Install GitHub Actions runner on your GPU machine
   - Add labels: `[self-hosted, guardian, gpu]`
   - Ensure GPU drivers and CUDA are installed

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Trigger Pipeline**:
   - Push to `main` branch
   - Close a pull request to `main`
   - Manual trigger via GitHub Actions "Run workflow" button

## 📊 Pipeline Components

### Core Pipeline Files
- **`Guardian_pipeline.py`** - Main local training pipeline
- **`Guardian_pipeline_github.py`** - GitHub Actions optimized pipeline
- **`mongodb_model_distribution.py`** - Model distribution system
- **`download_guardian_model.py`** - CLI for model downloading

### What the Pipeline Does

1. **Downloads Dataset** from ClearML (Guardian_Dataset)
2. **Preprocesses Data** - pose keypoint sequences with normalization
3. **Trains BiLSTM Model** with attention mechanism
4. **Optimizes Hyperparameters** - up to 100 trials with ClearML HPO
5. **Evaluates Best Model** with visualizations and metrics
6. **Publishes Model** to ClearML and MongoDB for deployment

## 🔧 Key Features

- **Automated Training**: No manual intervention required
- **GPU Acceleration**: Runs on your powerful local hardware
- **Experiment Tracking**: All results logged to ClearML
- **Hyperparameter Optimization**: Finds best model configuration
- **Rich Visualizations**: Attention maps, confusion matrices, training curves
- **CI/CD Integration**: Automated pipeline execution via GitHub Actions
- **Model Distribution**: MongoDB-based system for model storage and retrieval
- **Team Collaboration**: Easy model sharing and versioning

## 📈 Results & Outputs

- Training metrics and loss curves
- Confusion matrix visualization
- Attention analysis plots
- Model artifacts (.pth files)
- Complete experiment tracking in ClearML
- Published model ID for deployment
- Model configuration JSON files

## 📱 Model Distribution System

The Guardian AI project includes a comprehensive model distribution system:

- **MongoDB Storage**: Models are stored in MongoDB Atlas for accessibility
- **Version Control**: Track model versions, accuracies, and metadata
- **Easy Downloads**: Simple CLI tool for team members to download models
- **Best Model Access**: Automatically retrieve the highest performing model
- **Configuration Management**: Store and retrieve model hyperparameters

### Using the Model Downloader

```bash
# List all available models
python download_guardian_model.py --list

# Download the best model (highest accuracy)
python download_guardian_model.py --best

# Download a specific model by name
python download_guardian_model.py --model guardian_model_12345678_95

# Interactive mode
python download_guardian_model.py --interactive
```

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run local pipeline
python Guardian_pipeline.py

# Run GitHub-optimized pipeline locally
python Guardian_pipeline_github.py

# Configure model distribution
cp .secrets.example .secrets
# Edit .secrets with your MongoDB credentials
```

## 📁 Repository Structure

```
├── Guardian_pipeline.py              # Main local training pipeline
├── Guardian_pipeline_github.py       # GitHub Actions optimized pipeline
├── mongodb_model_distribution.py     # Model distribution system
├── download_guardian_model.py        # CLI for model downloading
├── requirements.txt                  # Python dependencies
├── setup.py                          # ClearML setup helper
├── clearml.conf                      # ClearML configuration
├── .secrets                          # MongoDB credentials (gitignored)
├── .github/
│   └── workflows/
│       └── guardian-pipeline-aws.yml # GitHub Actions workflow
├── actions-runner/                   # Self-hosted runner setup
├── training_utils/                   # Training utility modules
├── data/
│   └── Guardian_Dataset/             # Local dataset directory
└── README.md                         # This file
```

## 🔄 GitHub Actions Workflow

The repository includes a complete CI/CD pipeline that:
- Automatically triggers on code changes
- Runs on self-hosted GPU runners
- Executes the Guardian pipeline
- Uploads results and artifacts
- Provides detailed logging and monitoring
- Deploys models to MongoDB for distribution

## 🔗 Monitoring & Results

- **ClearML Dashboard**: https://app.clear.ml/
- **GitHub Actions**: Check the Actions tab for pipeline runs
- **Artifacts**: Download training outputs from GitHub Actions
- **Model Registry**: Published models available in ClearML and MongoDB

## 💡 Advanced Configuration

### Model Deployment Threshold

The pipeline automatically deploys models that exceed an accuracy threshold:

```python
# In Guardian_pipeline_github.py
deployment_status = deploy_model_github(
    best_model_id=best_model_id,
    best_model_path=best_model_path,
    test_accuracy=accuracy_value,
    min_accuracy_threshold=85.0,  # Deploy if accuracy >= 85%
    mongo_uri=mongo_uri
)
```

Adjust the `min_accuracy_threshold` to control when models are deployed to production.

### Hyperparameter Optimization

Customize HPO settings in the `bilstm_hyperparam_optimizer_github` function:

```python
# Increase trials for more thorough search
total_max_trials=100

# Modify search space
UniformParameterRange('General/base_lr', min_value=0.0001, max_value=0.01)
DiscreteParameterRange('General/hidden_size', values=[128, 192, 256, 320, 384, 512])
```

## 🔧 Troubleshooting

### MongoDB Connection Issues

If you encounter MongoDB connection issues in the pipeline, check the following:

1. **Environment Variables**: Ensure `MONGODB_URI` and `MONGODB_DATABASE` are properly set in GitHub Secrets.

2. **Connection Timeout**: The system now uses more robust connection parameters with proper timeout settings:

   ```python
   # Recent updates to MongoDB connection code
   self.client = MongoClient(
       self.uri,
       server_api=ServerApi('1'),
       serverSelectionTimeoutMS=5000,  # 5 second timeout
       connectTimeoutMS=10000,         # 10 second connection timeout
       socketTimeoutMS=45000,          # 45 second socket timeout
       ssl=True,
       ssl_cert_reqs=ssl.CERT_NONE,  
       tlsAllowInvalidCertificates=True
   )
   ```

3. **CI/CD Environment**: The system now checks for environment variables first before attempting to load from `.secrets` file, making it more robust in CI/CD environments.

4. **Certificate Issues**: If you encounter SSL/TLS certificate issues, the code now handles this with proper settings.

5. **Debugging**: More detailed error logging has been added to help diagnose connection issues.

### Common Errors and Solutions

| Error | Solution |
|-------|----------|
| "MongoDB URI not provided" | Set MONGODB_URI in GitHub Secrets or environment variables |
| "Could not connect to MongoDB" | Check network connectivity, firewall settings, and MongoDB Atlas IP access list |
| "not authorized" | Verify credentials in the connection string |
| "SSL handshake failed" | The system now disables strict certificate validation for CI/CD |
| "Model file not found" | Check the file paths and ensure model files are saved correctly |

## 🧹 Repository Maintenance

To clean up unnecessary files and keep only Guardian pipeline essentials:

```bash
# Cleans up hyperparameter files from previous runs
python Guardian_pipeline_github.py --cleanup
```

## 🔒 Security

- All credentials are stored as GitHub Secrets
- MongoDB connection strings are never committed to the repository
- The `.secrets` file is included in `.gitignore`
- Use environment variables for local development

---

**Guardian AI: Protecting through Intelligence** 🛡️
