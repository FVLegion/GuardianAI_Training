#!/usr/bin/env python3
"""
Check dataset on your specific runner setup.
Run this on your sagemaker runner to verify dataset.
"""

import pathlib
import os

def check_runner_dataset():
    """Check dataset on sagemaker runner."""
    print("🔍 Checking Dataset on SageMaker Runner")
    print("=" * 60)
    
    # Your absolute dataset path
    absolute_dataset_path = pathlib.Path("/home/sagemaker-user/data/Guardian_Dataset")
    
    # Expected workspace path
    workspace_path = pathlib.Path("/home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training")
    relative_dataset_path = workspace_path / "data" / "Guardian_Dataset"
    
    print(f"Checking absolute path: {absolute_dataset_path}")
    print(f"Checking workspace path: {relative_dataset_path}")
    print(f"Current working directory: {pathlib.Path.cwd()}")
    
    # Check absolute path
    print(f"\n📁 Absolute Dataset Path Check:")
    if absolute_dataset_path.exists():
        print(f"✅ Dataset exists at: {absolute_dataset_path}")
        
        # Check structure
        expected_classes = ["Falling", "No Action", "Waving"]
        found_classes = []
        total_files = 0
        
        for class_name in expected_classes:
            class_dir = absolute_dataset_path / class_name
            if class_dir.exists():
                keypoint_files = list(class_dir.glob("*_keypoints.json"))
                json_files = list(class_dir.glob("*.json"))
                print(f"✅ {class_name}: {len(keypoint_files)} keypoint files, {len(json_files)} total JSON files")
                found_classes.append(class_name)
                total_files += len(json_files)
                
                # Show sample files
                if json_files:
                    print(f"   Sample files: {[f.name for f in json_files[:2]]}")
            else:
                print(f"❌ {class_name}: Directory missing")
        
        print(f"\n📊 Absolute Path Summary:")
        print(f"   - Classes found: {len(found_classes)}/3")
        print(f"   - Total JSON files: {total_files}")
        
        absolute_ok = len(found_classes) > 0
    else:
        print(f"❌ Dataset NOT found at: {absolute_dataset_path}")
        absolute_ok = False
    
    # Check workspace path
    print(f"\n📁 Workspace Dataset Path Check:")
    if relative_dataset_path.exists():
        print(f"✅ Dataset exists at: {relative_dataset_path}")
        
        # Check if it's a symlink
        if relative_dataset_path.is_symlink():
            target = relative_dataset_path.readlink()
            print(f"🔗 This is a symlink pointing to: {target}")
        
        workspace_ok = True
    else:
        print(f"❌ Dataset NOT found at: {relative_dataset_path}")
        workspace_ok = False
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if absolute_ok and workspace_ok:
        print("🎉 PERFECT! Both paths work. Pipeline will use the dataset.")
    elif absolute_ok and not workspace_ok:
        print("🔧 SOLUTION NEEDED: Create symlink or copy dataset to workspace")
        print(f"Run this command:")
        print(f"  mkdir -p {workspace_path}/data")
        print(f"  ln -s {absolute_dataset_path} {workspace_path}/data/Guardian_Dataset")
    elif not absolute_ok and workspace_ok:
        print("✅ Workspace dataset is ready. Pipeline will work.")
    else:
        print("❌ No dataset found in either location. Please check your dataset.")
    
    return absolute_ok or workspace_ok

def show_symlink_commands():
    """Show exact commands to create symlink."""
    print(f"\n🔧 Symlink Creation Commands:")
    print("=" * 50)
    print("# Navigate to workspace")
    print("cd /home/sagemaker-user/actions-runner/_work/GuardianAI_Training/GuardianAI_Training")
    print()
    print("# Create data directory")
    print("mkdir -p data")
    print()
    print("# Create symlink")
    print("ln -s /home/sagemaker-user/data/Guardian_Dataset data/Guardian_Dataset")
    print()
    print("# Verify")
    print("ls -la data/")
    print("python check_runner_dataset.py")

if __name__ == "__main__":
    print("🦾 Guardian AI Runner Dataset Checker")
    print("=" * 60)
    
    success = check_runner_dataset()
    
    if not success:
        show_symlink_commands()
    
    print("\n" + "=" * 60) 