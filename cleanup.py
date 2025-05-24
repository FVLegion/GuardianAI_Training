#!/usr/bin/env python3
"""
Guardian AI Training Pipeline Cleanup Script
Removes training artifacts and temporary files
"""

import os
import shutil
import glob

def remove_files(pattern, description):
    """Remove files matching a pattern"""
    files = glob.glob(pattern)
    if files:
        print(f"🗑️  Removing {len(files)} {description}...")
        for file in files:
            try:
                os.remove(file)
                print(f"   Removed: {file}")
            except OSError as e:
                print(f"   Error removing {file}: {e}")
    else:
        print(f"✅ No {description} found")

def remove_directories(pattern, description):
    """Remove directories matching a pattern"""
    dirs = glob.glob(pattern)
    if dirs:
        print(f"🗑️  Removing {len(dirs)} {description}...")
        for dir_path in dirs:
            try:
                shutil.rmtree(dir_path)
                print(f"   Removed: {dir_path}")
            except OSError as e:
                print(f"   Error removing {dir_path}: {e}")
    else:
        print(f"✅ No {description} found")

def main():
    """Main cleanup function"""
    print("🧹 Guardian AI Training Pipeline Cleanup")
    print("=" * 50)
    
    # Remove model checkpoints
    remove_files("best_model_*.pt", "model checkpoints")
    remove_files("*.pt", "PyTorch model files")
    remove_files("*.pth", "PyTorch model files")
    
    # Remove training artifacts
    remove_files("training_metrics.png", "training metric plots")
    remove_files("attention_analysis.png", "attention analysis plots")
    remove_files("confusion_matrix.png", "confusion matrix plots")
    remove_files("*.png", "image files")
    remove_files("*.jpg", "image files")
    remove_files("*.jpeg", "image files")
    
    # Remove evaluation outputs
    remove_directories("evaluation_outputs", "evaluation output directories")
    
    # Remove Python cache
    remove_directories("__pycache__", "Python cache directories")
    remove_directories("training_utils/__pycache__", "training utils cache directories")
    remove_files("*.pyc", "Python compiled files")
    
    # Remove logs
    remove_files("*.log", "log files")
    
    # Remove temporary files
    remove_files("*.tmp", "temporary files")
    remove_directories("tmp", "temporary directories")
    remove_directories("temp", "temporary directories")
    
    print("\n🎉 Cleanup complete!")
    print("Repository is now clean and ready for commit.")

if __name__ == "__main__":
    main() 