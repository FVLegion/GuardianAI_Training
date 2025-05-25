#!/usr/bin/env python3
"""
Debug script to test ClearML dataset download locally.
"""

from clearml import Dataset
import pathlib
import tempfile
import os

def debug_dataset_download():
    """Debug the dataset download process."""
    
    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    
    print(f"🔍 Debugging dataset download...")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset project: {dataset_project}")
    print("-" * 50)
    
    try:
        # Get the dataset
        print("📥 Getting dataset from ClearML...")
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True
        )
        
        if not dataset:
            print("❌ Dataset not found")
            return
        
        print(f"✅ Dataset found: {dataset.id}")
        
        # Download to temporary location
        print("⬇️  Downloading dataset...")
        temp_download_path_str = dataset.get_local_copy()
        
        if not temp_download_path_str:
            print("❌ Failed to get local copy")
            return
        
        temp_download_path = pathlib.Path(temp_download_path_str).resolve()
        print(f"📁 Downloaded to: {temp_download_path}")
        
        # Check what we got
        if not temp_download_path.exists():
            print("❌ Download path doesn't exist")
            return
        
        print(f"📊 Download path exists: {temp_download_path.is_dir()}")
        
        # Show contents
        print("\n📋 Contents of download path:")
        items = list(temp_download_path.iterdir())
        for i, item in enumerate(items):
            item_type = "DIR" if item.is_dir() else "FILE"
            print(f"  {i+1}. [{item_type}] {item.name}")
        
        # Test the structure detection logic
        expected_actions = ["Falling", "No Action", "Waving"]
        dataset_root = None
        
        print(f"\n🔍 Testing structure detection...")
        print(f"Expected actions: {expected_actions}")
        
        # Check if the temp path directly contains action directories
        action_dirs_found = []
        for action in expected_actions:
            action_path = temp_download_path / action
            print(f"  Checking: {action_path}")
            if action_path.exists() and action_path.is_dir():
                action_dirs_found.append(action)
                print(f"    ✅ Found: {action}")
            else:
                print(f"    ❌ Not found: {action}")
        
        print(f"Action directories found: {action_dirs_found}")
        
        if len(action_dirs_found) == len(expected_actions):
            dataset_root = temp_download_path
            print(f"✅ Direct structure found, dataset_root = {dataset_root}")
        else:
            print("❌ Direct structure not found, checking nested...")
            
            # Look for nested structure
            for item in temp_download_path.iterdir():
                if item.is_dir():
                    print(f"  Checking nested directory: {item}")
                    nested_actions = []
                    for action in expected_actions:
                        nested_path = item / action
                        if nested_path.exists():
                            nested_actions.append(action)
                            print(f"    ✅ Found nested: {action}")
                        else:
                            print(f"    ❌ Not found nested: {action}")
                    
                    if len(nested_actions) == len(expected_actions):
                        dataset_root = item
                        print(f"✅ Nested structure found, dataset_root = {dataset_root}")
                        break
        
        print(f"\n🎯 Final dataset_root: {dataset_root}")
        print(f"Type of dataset_root: {type(dataset_root)}")
        
        if dataset_root is None:
            print("❌ dataset_root is None - this is the problem!")
            
            # Show detailed structure
            print("\n📂 Detailed directory structure:")
            for root, dirs, files in os.walk(temp_download_path):
                level = root.replace(str(temp_download_path), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:3]:  # Show first 3 files per directory
                    print(f"{subindent}{file}")
                if len(files) > 3:
                    print(f"{subindent}... and {len(files) - 3} more files")
            
            # Look for JSON files
            json_files = list(temp_download_path.rglob("*_keypoints.json"))
            print(f"\n📄 Found {len(json_files)} keypoint JSON files:")
            for i, json_file in enumerate(json_files[:5]):
                rel_path = json_file.relative_to(temp_download_path)
                print(f"  {i+1}. {rel_path}")
        
        else:
            print(f"✅ dataset_root is valid: {dataset_root}")
            
            # Test the copy operation that's failing
            print(f"\n🧪 Testing copy operation...")
            for action in expected_actions:
                source_action_dir = dataset_root / action
                print(f"  Testing: {dataset_root} / {action} = {source_action_dir}")
                print(f"  Exists: {source_action_dir.exists()}")
                if source_action_dir.exists():
                    json_files = list(source_action_dir.glob("*_keypoints.json"))
                    print(f"  JSON files: {len(json_files)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset_download() 