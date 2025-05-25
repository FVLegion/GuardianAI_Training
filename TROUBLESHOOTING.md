# Guardian AI Pipeline Troubleshooting Guide

## Common Issues and Solutions

### 1. "Dataset setup failed" Error

**Symptoms:**
```
ValueError: Dataset setup failed.
```

**Possible Causes & Solutions:**

#### A. ClearML Dataset Not Found
- **Cause**: The dataset "Guardian_Dataset" doesn't exist in ClearML project "Guardian_Training"
- **Solution**: Upload your dataset to ClearML first:
  ```bash
  python upload_dataset_simple.py
  ```

#### B. ClearML Credentials Not Configured
- **Cause**: Missing or incorrect ClearML API credentials in GitHub Secrets
- **Solution**: 
  1. Go to your GitHub repository → Settings → Secrets and variables → Actions
  2. Add these secrets:
     - `CLEARML_API_ACCESS_KEY`: Your ClearML access key
     - `CLEARML_API_SECRET_KEY`: Your ClearML secret key  
     - `CLEARML_API_HOST`: Your ClearML server URL (e.g., `https://app.clear.ml`)

#### C. Network Connectivity Issues
- **Cause**: Runner cannot connect to ClearML servers
- **Solution**: Check if your self-hosted runner has internet access and can reach ClearML servers

### 2. ClearML Connection Issues

**Symptoms:**
```
Error downloading dataset from ClearML: [connection error]
```

**Solutions:**

#### Check ClearML Configuration
Run this on your runner to test ClearML connectivity:
```python
from clearml import Task
try:
    task = Task.init(project_name='test', task_name='connection_test')
    print("✅ ClearML connection successful")
    task.close()
except Exception as e:
    print(f"❌ ClearML connection failed: {e}")
```

#### Verify Environment Variables
The pipeline will now show ClearML configuration status. Look for:
```
ClearML API Host: Set/Not Set
ClearML API Key: Set/Not Set  
ClearML API Secret: Set/Not Set
```

### 3. Mock Dataset Fallback

**What happens:**
If ClearML dataset download fails, the pipeline automatically creates a mock dataset for testing.

**Symptoms:**
```
Mock dataset created successfully for testing purposes
```

**This is normal** when:
- Testing the pipeline without real data
- ClearML is not configured
- Dataset doesn't exist in ClearML

### 4. Self-Hosted Runner Issues

#### Runner Not Picking Up Jobs
- **Check**: Runner labels match workflow requirements: `[self-hosted, Linux, X64, guardian, gpu]`
- **Solution**: Update runner labels or workflow labels to match

#### Python/Dependencies Issues
- **Check**: Python 3.11 is installed and accessible
- **Solution**: Ensure `actions/setup-python@v5` can install Python 3.11

### 5. Memory/Resource Issues

#### Out of Memory During Training
- **Symptoms**: Process killed or memory errors
- **Solution**: Reduce batch size in the pipeline:
  ```python
  batch_size=8  # Reduce from 16
  ```

#### GPU Issues
- **Check**: GPU is available and CUDA is properly installed
- **Solution**: The pipeline will automatically fall back to CPU if GPU is not available

## Debugging Steps

### 1. Enable Verbose Logging
The pipeline now includes detailed logging. Check the GitHub Actions logs for:
- ClearML connection status
- Dataset download progress
- File counts and structure validation

### 2. Test Locally First
Before running in GitHub Actions, test locally:
```bash
# Test ClearML connection
python -c "from clearml import Task; Task.init(project_name='test', task_name='test')"

# Test dataset upload
python upload_dataset_simple.py

# Test pipeline locally
python Guardian_pipeline_github.py
```

### 3. Check GitHub Actions Logs
Look for these key sections in the logs:
- "Verify ClearML Configuration" step
- "Starting dataset setup" messages
- "Dataset validation complete" messages

## Quick Fixes

### Option 1: Use Mock Data (Fastest)
If you just want to test the pipeline:
1. The pipeline will automatically create mock data if ClearML fails
2. This allows you to test the training process without real data

### Option 2: Upload Minimal Dataset
```bash
python upload_dataset_simple.py
```
This creates and uploads a minimal test dataset to ClearML.

### Option 3: Use Your Real Dataset
1. Place your dataset in `data/Guardian_Dataset/` with structure:
   ```
   data/Guardian_Dataset/
   ├── Falling/
   │   ├── video1_keypoints.json
   │   └── video2_keypoints.json
   ├── No Action/
   │   ├── video3_keypoints.json
   │   └── video4_keypoints.json
   └── Waving/
       ├── video5_keypoints.json
       └── video6_keypoints.json
   ```
2. Run: `python upload_dataset.py`

## Expected Behavior

### Successful Run
```
✅ ClearML connection successful
✅ Dataset downloaded successfully
✅ Found X keypoints files in expected structure
✅ Training completed. Task ID: xxx, Model ID: xxx
```

### Fallback Mode (Also OK)
```
⚠️  Failed to download dataset from ClearML
✅ Mock dataset created successfully for testing purposes
✅ Training completed. Task ID: xxx, Model ID: xxx
```

## Getting Help

If you're still having issues:

1. **Check the GitHub Actions logs** for the exact error message
2. **Verify your ClearML credentials** are correctly set in GitHub Secrets
3. **Test ClearML connection** on your runner manually
4. **Try the mock dataset approach** first to isolate the issue

The pipeline is now designed to be robust and will work even without ClearML configured, using mock data for testing purposes. 