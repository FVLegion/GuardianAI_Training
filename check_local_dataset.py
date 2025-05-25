#!/usr/bin/env python3
"""
Quick script to check if local dataset is properly structured.
Run this on your runner to verify dataset placement.
"""

import pathlib
import os

def check_local_dataset():
    """Check if local dataset is properly structured."""
    print("🔍 Checking Local Dataset Structure")
    print("=" * 50)
    
    # Get current directory (should be runner workspace)
    current_dir = pathlib.Path.cwd()
    dataset_path = current_dir / "data" / "Guardian_Dataset"
    
    print(f"Current directory: {current_dir}")
    print(f"Looking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print("❌ Dataset directory does not exist")
        print(f"Please create: {dataset_path}")
        return False
    
    print("✅ Dataset directory exists")
    
    # Check for expected action directories
    expected_classes = ["Falling", "No Action", "Waving"]
    found_classes = []
    total_files = 0
    
    for class_name in expected_classes:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            keypoint_files = list(class_dir.glob("*_keypoints.json"))
            json_files = list(class_dir.glob("*.json"))
            print(f"✅ {class_name}: {len(keypoint_files)} keypoint files, {len(json_files)} total JSON files")
            found_classes.append(class_name)
            total_files += len(json_files)
            
            # Show sample files
            if json_files:
                print(f"   Sample files: {[f.name for f in json_files[:3]]}")
        else:
            print(f"❌ {class_name}: Directory missing")
    
    print(f"\n📊 Summary:")
    print(f"   - Classes found: {len(found_classes)}/3")
    print(f"   - Total JSON files: {total_files}")
    
    # Test if pipeline would detect this
    has_action_structure = any((dataset_path / folder).exists() for folder in expected_classes)
    
    if has_action_structure:
        print("\n🎉 SUCCESS! Pipeline will detect this dataset!")
        print("The download function will return True and use local data.")
        return True
    else:
        print("\n❌ FAILURE! Pipeline will not detect this dataset.")
        print("Please ensure you have the correct directory structure.")
        return False

if __name__ == "__main__":
    success = check_local_dataset()
    
    if success:
        print("\n✅ Your dataset is ready for the pipeline!")
    else:
        print("\n❌ Please fix the dataset structure before running the pipeline.") 