# Guardian AI GitHub Actions Setup

This guide helps you set up and run the Guardian AI training pipeline using GitHub Actions with a self-hosted runner.

## 🚀 Quick Start

### 1. Prerequisites
- Self-hosted GitHub Actions runner with:
  - Linux OS
  - Python 3.11
  - GPU support (optional, will fallback to CPU)
  - Labels: `[self-hosted, Linux, X64, guardian, gpu]`

### 2. Configure GitHub Secrets
Go to your repository → Settings → Secrets and variables → Actions

Add these secrets:
- `CLEARML_API_ACCESS_KEY`: Your ClearML access key
- `CLEARML_API_SECRET_KEY`: Your ClearML secret key
- `CLEARML_API_HOST`: Your ClearML server URL (e.g., `https://app.clear.ml`)

### 3. Upload Dataset (Optional)
If you have a real dataset:
```bash
python upload_dataset_simple.py
```

If you don't have a dataset, the pipeline will automatically create mock data for testing.

### 4. Trigger Pipeline
Push to the `main` branch or manually trigger the workflow.

## 📁 File Structure

```
your-repo/
├── .github/workflows/
│   └── guardian-pipeline-aws.yml    # GitHub Actions workflow
├── Guardian_pipeline_github.py      # Main pipeline script
├── upload_dataset_simple.py         # Dataset upload utility
├── requirements.txt                 # Python dependencies
├── TROUBLESHOOTING.md               # Troubleshooting guide
└── README_GITHUB_ACTIONS.md        # This file
```

## 🔧 Pipeline Features

### Robust Dataset Handling
- **Primary**: Downloads dataset from ClearML
- **Fallback**: Creates mock dataset if ClearML fails
- **Validation**: Automatically fixes common dataset structure issues

### Lightweight Training
- Optimized for GitHub Actions (reduced epochs, batch size)
- GPU support with CPU fallback
- Comprehensive logging and monitoring

### Error Recovery
- Graceful handling of missing datasets
- Automatic mock data generation
- Detailed error reporting

## 📊 Expected Output

### Successful Run
```
🔍 Checking ClearML configuration...
✅ ClearML connection successful
✅ Dataset downloaded successfully
✅ Found 9 keypoints files in expected structure
✅ Training completed. Task ID: xxx, Model ID: xxx
🎉 Guardian AI GitHub Pipeline Completed!
```

### Fallback Mode (Also Valid)
```
🔍 Checking ClearML configuration...
⚠️  ClearML connection failed
✅ Mock dataset created successfully for testing purposes
✅ Training completed. Task ID: xxx, Model ID: xxx
🎉 Guardian AI GitHub Pipeline Completed!
```

## 🛠️ Customization

### Modify Training Parameters
Edit `Guardian_pipeline_github.py`:
```python
train_bilstm_github(
    dataset_path=dataset_path,
    input_size=input_size,
    num_classes=num_classes,
    epochs=20,        # Increase for longer training
    hidden_size=256,  # Increase for larger model
    batch_size=32     # Increase if you have more memory
)
```

### Change Runner Requirements
Edit `.github/workflows/guardian-pipeline-aws.yml`:
```yaml
runs-on: [self-hosted, Linux, X64, your-custom-labels]
```

### Add More Actions
The workflow can be extended with additional steps:
```yaml
- name: Deploy Model
  run: |
    python deploy_model.py
    
- name: Run Tests
  run: |
    python -m pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

1. **"Dataset setup failed"**
   - Check ClearML credentials in GitHub Secrets
   - Run `python upload_dataset_simple.py` to create a test dataset

2. **Runner not picking up jobs**
   - Verify runner labels match workflow requirements
   - Check runner status in GitHub repository settings

3. **Memory issues**
   - Reduce `batch_size` in the training parameters
   - Ensure runner has sufficient RAM

4. **GPU not detected**
   - Pipeline will automatically fallback to CPU
   - Check CUDA installation on runner

### Debug Mode
For detailed debugging, check the GitHub Actions logs for:
- ClearML configuration status
- Dataset download progress
- Training metrics and progress

## 📈 Monitoring

### ClearML Dashboard
If ClearML is configured, you can monitor:
- Training progress and metrics
- Model artifacts and versions
- Experiment comparisons

### GitHub Actions
Monitor pipeline execution:
- Build status and logs
- Execution time and resource usage
- Success/failure notifications

## 🔄 Continuous Integration

The pipeline runs automatically on:
- Push to `main` branch
- Manual workflow dispatch

You can modify the trigger in `.github/workflows/guardian-pipeline-aws.yml`:
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
```

## 📚 Additional Resources

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Detailed troubleshooting guide
- [ClearML Documentation](https://clear.ml/docs) - ClearML setup and usage
- [GitHub Actions Documentation](https://docs.github.com/en/actions) - GitHub Actions reference

## 🆘 Getting Help

If you encounter issues:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common solutions
2. Review GitHub Actions logs for error details
3. Test ClearML connection manually on your runner
4. Try the mock dataset approach for initial testing

The pipeline is designed to be robust and will work even without ClearML configured! 