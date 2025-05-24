# Guardian AI GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating the Guardian AI training and deployment pipeline.

## 🚀 Workflows

### 1. `guardian-pipeline.yml` - Training Pipeline
**Trigger**: Pull requests to `main` branch
**Purpose**: Validates code changes by running the complete training pipeline

**What it does**:
- Sets up Python environment
- Installs dependencies
- Validates ClearML connection
- Runs the Guardian training pipeline
- Uploads training artifacts
- Provides pipeline summary

### 2. `guardian-deploy.yml` - Training & Deployment
**Trigger**: Push to `main` branch or manual dispatch
**Purpose**: Trains the model and deploys it if it meets quality thresholds

**What it does**:
- Runs the complete training pipeline (50 trials HPO)
- Extracts best model and performance metrics
- Validates model meets minimum accuracy threshold (75%)
- Deploys model to ClearML serving if approved
- Creates deployment records
- Provides deployment summary

## 🔧 Setup Requirements

### 1. GitHub Secrets
Add these secrets to your GitHub repository:

```
CLEARML_API_ACCESS_KEY=your_access_key
CLEARML_API_SECRET_KEY=your_secret_key  
CLEARML_API_HOST=https://api.clear.ml
```

**How to add secrets**:
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Add each secret with the exact names above

### 2. ClearML Configuration
Ensure your `clearml.conf` file is properly configured and your ClearML account has:
- Access to the Guardian_Training project
- Permissions to create and publish models
- Dataset access for Guardian_Dataset

## 📊 Model Deployment Process

### What is Model Deployment?
In this context, model deployment means:

1. **Model Registration**: The trained model is registered in ClearML's model registry
2. **Performance Validation**: Model accuracy is checked against minimum thresholds
3. **Model Publishing**: Model is marked as "published" and tagged for serving
4. **Metadata Recording**: Deployment information is recorded for tracking
5. **Serving Ready**: Model becomes available for inference through ClearML Serving

### Deployment Environments
- **Staging**: For testing and validation (default)
- **Production**: For live inference (requires manual approval)

### Quality Gates
- Minimum accuracy threshold: 75%
- Model must complete all training phases successfully
- All hyperparameter optimization trials must complete

## 🎯 Usage Examples

### Automatic Training (PR)
```bash
# Create a feature branch
git checkout -b feature/model-improvements

# Make your changes
git add .
git commit -m "Improve model architecture"
git push origin feature/model-improvements

# Create PR to main → triggers training pipeline
```

### Manual Deployment
1. Go to Actions tab in GitHub
2. Select "Deploy Guardian AI Model"
3. Click "Run workflow"
4. Choose environment (staging/production)
5. Click "Run workflow"

### Monitor Progress
- **GitHub Actions**: View real-time logs and progress
- **ClearML Dashboard**: Monitor training metrics and experiments
- **Artifacts**: Download training plots and model files

## 🔍 Troubleshooting

### Common Issues

1. **ClearML Connection Failed**
   - Check if secrets are properly set
   - Verify ClearML API host URL
   - Ensure account has proper permissions

2. **Pipeline Timeout**
   - Training with 50 trials can take 2-4 hours
   - Timeout is set to 4 hours (240 minutes)
   - Consider reducing trials for testing

3. **Model Deployment Blocked**
   - Check if model accuracy meets 75% threshold
   - Review training logs for issues
   - Verify model was properly saved

4. **Artifact Upload Failed**
   - Check if training generated expected files
   - Verify file paths in workflow
   - Review storage permissions

## 📈 Monitoring & Metrics

### Key Metrics Tracked
- **Training Accuracy**: Per-epoch training performance
- **Validation Accuracy**: Model generalization performance  
- **Test Accuracy**: Final model evaluation
- **Hyperparameter Performance**: HPO trial results
- **Training Time**: Pipeline execution duration

### Dashboards
- **GitHub Actions**: Workflow execution status and logs
- **ClearML**: Experiment tracking and model registry
- **Parallel Coordinates**: Hyperparameter optimization visualization

## 🔄 Workflow Customization

### Adjusting Training Parameters
Edit `Guardian_pipeline.py`:
- `total_max_trials`: Number of HPO trials (default: 50)
- `epochs`: Training epochs per trial
- Hyperparameter ranges in the optimizer

### Changing Deployment Thresholds
Edit `guardian-deploy.yml`:
- `MIN_ACCURACY`: Minimum accuracy for deployment (default: 75%)
- `timeout-minutes`: Maximum workflow duration
- Environment options

### Adding Notifications
You can extend the workflows to add:
- Slack notifications
- Email alerts
- Teams messages
- Custom webhooks

## 🏗️ Architecture

```
GitHub Push/PR
       ↓
GitHub Actions Runner
       ↓
Guardian Pipeline
       ↓
ClearML Experiments
       ↓
Model Registry
       ↓
Deployment Validation
       ↓
ClearML Serving
```

This setup provides a complete MLOps pipeline with automated training, validation, and deployment capabilities. 