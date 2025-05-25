# Guardian AI Local GPU Execution Guide

This guide explains how to set up and use your local GPU resources to run the Guardian AI training pipeline when GitHub Actions are triggered, instead of running everything in the cloud.

## 🎯 Overview

The local execution system allows you to:
- Trigger pipeline execution on your local GPU machine when GitHub Actions are triggered
- Leverage your powerful local GPU resources for training
- Maintain full control over the training environment
- Avoid cloud compute limitations and costs
- Still benefit from ClearML experiment tracking

## 🏗️ Architecture

```
GitHub Action Trigger → Local Machine Detection → Pull Latest Code → Run Pipeline → Upload Results to ClearML
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Local Listener

**Option A: Using the Windows Batch Script (Recommended for Windows)**
```cmd
start_local_listener.bat
```

**Option B: Using Python Directly**
```bash
# Polling mode (checks GitHub every 60 seconds)
python local_pipeline_listener.py

# Webhook mode (real-time triggers)
python local_pipeline_listener.py --webhook
```

### 3. Trigger from GitHub

- Push to main branch
- Create a pull request
- Manually trigger the workflow from GitHub Actions

## 📋 Execution Modes

### 1. Polling Mode (Default)
- Checks GitHub repository every 60 seconds for new commits
- Automatically pulls changes and runs pipeline
- Simple setup, no network configuration needed
- Slight delay (up to 60 seconds) between trigger and execution

### 2. Webhook Mode
- Real-time triggers via HTTP webhook
- Immediate execution when GitHub Action is triggered
- Requires network configuration for external access
- More responsive but needs port forwarding/firewall setup

## 🔧 Configuration

### Local Listener Configuration

Edit `local_pipeline_listener.py` to customize:

```python
# Configuration
REPO_PATH = Path(__file__).parent
GITHUB_REPO = "your-username/GuardianAI_Training-6"  # Update this!
POLL_INTERVAL = 60  # seconds
WEBHOOK_PORT = 8080
LOG_FILE = "pipeline_listener.log"
```

### GitHub Workflow

The `guardian-local-trigger.yml` workflow:
- Runs quickly in GitHub Actions (< 1 minute)
- Creates trigger information
- Provides execution instructions
- Uploads trigger artifacts

## 🌐 Webhook Setup (Advanced)

### 1. Configure Your Router/Firewall

Forward port 8080 to your local machine:
- Router: Forward external port 8080 → your local IP:8080
- Windows Firewall: Allow inbound connections on port 8080

### 2. Update GitHub Workflow

Uncomment and configure the webhook section in `guardian-local-trigger.yml`:

```yaml
# If you have a local webhook endpoint, uncomment and configure:
curl -X POST http://YOUR_EXTERNAL_IP:8080/webhook \
  -H "Content-Type: application/json" \
  -d @pipeline_trigger.json
```

### 3. Test Webhook

```bash
# Test local webhook
curl -X POST http://localhost:8080/webhook \
  -H "Content-Type: application/json" \
  -d '{"test": "trigger"}'

# Check status
curl http://localhost:8080/status
```

## 📊 Monitoring and Logs

### Log Files
- `pipeline_listener.log`: Local listener activity
- Console output: Real-time status updates

### ClearML Integration
- All experiments are still tracked in ClearML
- Results uploaded automatically after training
- View progress at: https://app.clear.ml/

### GitHub Actions
- Quick trigger confirmation
- Execution instructions in workflow summary
- Trigger artifacts for debugging

## 🔍 Troubleshooting

### Common Issues

**1. Git Authentication**
```bash
# Configure Git credentials
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# For private repos, use personal access token
git config --global credential.helper store
```

**2. Python Environment**
```bash
# Ensure correct Python environment
conda activate your_env_name
# or
source venv/bin/activate
```

**3. GPU Not Detected**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**4. ClearML Configuration**
```bash
# Verify ClearML setup
clearml-init
```

### Debug Mode

Run with debug logging:
```python
# In local_pipeline_listener.py, change:
logging.basicConfig(level=logging.DEBUG, ...)
```

## 🎮 Usage Examples

### Example 1: Development Workflow
1. Make changes to your code
2. Push to GitHub
3. Local listener detects changes
4. Pipeline runs automatically on your GPU
5. Results appear in ClearML

### Example 2: Hyperparameter Tuning
1. Update hyperparameters in code
2. Create pull request
3. GitHub Action triggers local execution
4. Multiple experiments run with different parameters
5. Compare results in ClearML dashboard

### Example 3: Manual Trigger
1. Go to GitHub Actions
2. Click "Run workflow" on guardian-local-trigger
3. Local machine receives trigger
4. Pipeline executes immediately

## 🔒 Security Considerations

### Local Machine Security
- Keep your machine updated
- Use strong passwords
- Consider VPN for remote access

### GitHub Secrets
- Store sensitive data in GitHub Secrets
- Never commit API keys or passwords
- Use environment variables for configuration

### Network Security
- Use HTTPS for webhooks when possible
- Consider IP whitelisting
- Monitor webhook access logs

## 🚀 Performance Tips

### GPU Optimization
- Ensure CUDA drivers are up to date
- Monitor GPU memory usage
- Use appropriate batch sizes for your GPU

### System Resources
- Close unnecessary applications during training
- Monitor CPU and RAM usage
- Ensure adequate cooling for extended training

### Network Optimization
- Use wired connection for stability
- Consider local dataset caching
- Monitor bandwidth usage

## 📈 Advanced Features

### Multiple GPU Support
```python
# In Guardian_pipeline.py, configure for multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Distributed Training
- Configure for multiple machines
- Use shared storage for datasets
- Coordinate through ClearML

### Custom Triggers
- Add custom webhook endpoints
- Integrate with other CI/CD systems
- Create custom notification systems

## 🆘 Support

### Getting Help
1. Check the logs: `pipeline_listener.log`
2. Review GitHub Actions output
3. Check ClearML experiment logs
4. Verify system requirements

### Reporting Issues
Include in your report:
- Operating system and version
- Python version and environment
- GPU specifications
- Error messages and logs
- Steps to reproduce

## 📚 Additional Resources

- [ClearML Documentation](https://clear.ml/docs/)
- [PyTorch GPU Guide](https://pytorch.org/get-started/locally/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Git Configuration Guide](https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup) 