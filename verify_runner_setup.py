#!/usr/bin/env python3
"""
Comprehensive verification script for your self-hosted runner setup.
Run this on your runner to verify everything is ready.
"""

import pathlib
import os
import subprocess
import sys

def check_runner_environment():
    """Check the runner environment and paths."""
    print("🔍 Checking Runner Environment")
    print("=" * 60)
    
    # Check current working directory
    cwd = pathlib.Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the expected workspace
    expected_workspace = "/home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training"
    if str(cwd) == expected_workspace:
        print("✅ Running in expected GitHub Actions workspace")
    else:
        print(f"⚠️  Not in expected workspace. Expected: {expected_workspace}")
    
    # Check Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    return True

def check_dataset_paths():
    """Check all possible dataset paths."""
    print("\n📁 Checking Dataset Paths")
    print("=" * 60)
    
    paths_to_check = [
        ("/home/sagemaker-user/data/Guardian_Dataset", "Absolute dataset path"),
        ("/home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training/data/Guardian_Dataset", "Workspace dataset path"),
        (str(pathlib.Path.cwd() / "data" / "Guardian_Dataset"), "Current directory relative path")
    ]
    
    found_paths = []
    
    for path_str, description in paths_to_check:
        path = pathlib.Path(path_str)
        print(f"\n{description}:")
        print(f"  Path: {path}")
        
        if path.exists():
            print("  ✅ EXISTS")
            
            # Check if it's a symlink
            if path.is_symlink():
                target = path.readlink()
                print(f"  🔗 Symlink pointing to: {target}")
            
            # Check structure
            expected_classes = ["Falling", "No Action", "Waving"]
            found_classes = []
            total_files = 0
            
            for class_name in expected_classes:
                class_dir = path / class_name
                if class_dir.exists():
                    json_files = list(class_dir.glob("*.json"))
                    keypoint_files = list(class_dir.glob("*_keypoints.json"))
                    found_classes.append(class_name)
                    total_files += len(json_files)
                    print(f"  ✅ {class_name}: {len(keypoint_files)} keypoint files, {len(json_files)} total JSON files")
            
            if found_classes:
                print(f"  📊 Summary: {len(found_classes)}/3 classes, {total_files} total files")
                found_paths.append((path, len(found_classes), total_files))
            else:
                print("  ❌ No valid class directories found")
        else:
            print("  ❌ DOES NOT EXIST")
    
    return found_paths

def test_symlink_creation():
    """Test creating symlink if needed."""
    print("\n🔗 Testing Symlink Creation")
    print("=" * 60)
    
    absolute_path = pathlib.Path("/home/sagemaker-user/data/Guardian_Dataset")
    workspace_path = pathlib.Path.cwd() / "data" / "Guardian_Dataset"
    
    if not absolute_path.exists():
        print("❌ Absolute dataset path doesn't exist. Cannot create symlink.")
        return False
    
    if workspace_path.exists():
        print("✅ Workspace dataset path already exists")
        if workspace_path.is_symlink():
            target = workspace_path.readlink()
            print(f"🔗 Existing symlink points to: {target}")
        return True
    
    # Try to create symlink
    try:
        # Create data directory
        workspace_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create symlink
        workspace_path.symlink_to(absolute_path)
        print(f"✅ Successfully created symlink: {workspace_path} -> {absolute_path}")
        
        # Verify
        if workspace_path.exists() and workspace_path.is_symlink():
            print("✅ Symlink verification successful")
            return True
        else:
            print("❌ Symlink verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create symlink: {e}")
        return False

def test_pipeline_detection():
    """Test if the pipeline would detect the dataset."""
    print("\n🧪 Testing Pipeline Dataset Detection")
    print("=" * 60)
    
    # Simulate the pipeline's path detection logic
    possible_paths = [
        pathlib.Path("/home/sagemaker-user/data/Guardian_Dataset"),
        pathlib.Path("/home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training/data/Guardian_Dataset"),
        pathlib.Path.cwd() / "data" / "Guardian_Dataset",
    ]
    
    print("Pipeline will check paths in this order:")
    for i, path in enumerate(possible_paths, 1):
        exists = path.exists()
        status = "✅ FOUND" if exists else "❌ NOT FOUND"
        print(f"  {i}. {path} - {status}")
        
        if exists:
            print(f"     🎯 Pipeline will use this path!")
            return True
    
    print("❌ Pipeline will not find any dataset and will create mock data")
    return False

def run_verification():
    """Run complete verification."""
    print("🦾 Guardian AI Runner Setup Verification")
    print("=" * 70)
    
    # Step 1: Check environment
    env_ok = check_runner_environment()
    
    # Step 2: Check dataset paths
    found_paths = check_dataset_paths()
    
    # Step 3: Test symlink creation if needed
    symlink_ok = test_symlink_creation()
    
    # Step 4: Test pipeline detection
    pipeline_ok = test_pipeline_detection()
    
    # Summary
    print("\n📋 Verification Summary")
    print("=" * 60)
    print(f"Environment: {'✅ OK' if env_ok else '❌ Issues'}")
    print(f"Dataset paths found: {len(found_paths)}")
    print(f"Symlink setup: {'✅ OK' if symlink_ok else '❌ Issues'}")
    print(f"Pipeline detection: {'✅ OK' if pipeline_ok else '❌ Will use mock data'}")
    
    if pipeline_ok:
        print("\n🎉 SUCCESS! Your runner is ready for the pipeline!")
        print("You can now run the GitHub Actions workflow.")
    else:
        print("\n⚠️  ATTENTION: Pipeline will use mock data instead of your dataset.")
        print("This is still OK for testing, but you may want to fix the dataset setup.")
    
    return pipeline_ok

if __name__ == "__main__":
    success = run_verification()
    
    if success:
        print("\n✅ Verification completed successfully!")
    else:
        print("\n⚠️  Verification completed with warnings.")
    
    print("\nNext steps:")
    print("1. Commit and push your code changes")
    print("2. Trigger the GitHub Actions workflow")
    print("3. Check the workflow logs for dataset detection messages") 