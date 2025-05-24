#!/usr/bin/env python3
"""
Guardian AI Training Pipeline Setup Script
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def setup_clearml():
    """Guide user through ClearML setup"""
    print("\n🚀 Setting up ClearML...")
    print("Please run the following command and follow the prompts:")
    print("clearml-init")
    print("\nYou'll need:")
    print("- ClearML server URL (or use the default)")
    print("- Your API credentials from the ClearML web interface")
    
def check_gpu():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not detected. Training will use CPU (slower)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def main():
    """Main setup function"""
    print("🦾 Guardian AI Training Pipeline Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check GPU availability
    check_gpu()
    
    # Setup ClearML
    setup_clearml()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Configure ClearML with: clearml-init")
    print("2. Ensure your dataset is uploaded to ClearML")
    print("3. Run the pipeline with: python Guardian_pipeline.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 