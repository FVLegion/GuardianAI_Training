#!/usr/bin/env python3
"""
Debug script to identify why dataset download is failing.
Run this script to diagnose the exact issue with your dataset setup.
"""

import os
import pathlib
import json
from clearml import Dataset, Task

def check_clearml_connection():
    """Test ClearML connection and credentials."""
    print("🔍 Testing ClearML Connection...")
    print("=" * 50)
    
    # Check environment variables
    clearml_api_host = os.getenv('CLEARML_API_HOST')
    clearml_api_key = os.getenv('CLEARML_API_ACCESS_KEY')
    clearml_api_secret = os.getenv('CLEARML_API_SECRET_KEY')
    
    print(f"CLEARML_API_HOST: {'✅ Set' if clearml_api_host else '❌ Not Set'}")
    print(f"CLEARML_API_ACCESS_KEY: {'✅ Set' if clearml_api_key else '❌ Not Set'}")
    print(f"CLEARML_API_SECRET_KEY: {'✅ Set' if clearml_api_secret else '❌ Not Set'}")
    
    if clearml_api_host:
        print(f"Host URL: {clearml_api_host}")
    
    # Test connection
    try:
        print("\n🧪 Testing ClearML connection...")
        task = Task.init(project_name='debug_test', task_name='connection_test', auto_connect_frameworks=False)
        print("✅ ClearML connection successful!")
        task.close()
        return True
    except Exception as e:
        print(f"❌ ClearML connection failed: {e}")
        return False

def check_dataset_exists():
    """Check if the Guardian_Dataset exists in ClearML."""
    print("\n📊 Checking Dataset Existence...")
    print("=" * 50)
    
    try:
        dataset = Dataset.get(
            dataset_name="Guardian_Dataset",
            dataset_project="Guardian_Training",
            only_completed=True
        )
        
        if dataset:
            print("✅ Dataset 'Guardian_Dataset' found in ClearML!")
            print(f"   - Dataset ID: {dataset.id}")
            print(f"   - Project: Guardian_Training")
            
            # List files in dataset
            files = dataset.list_files()
            print(f"   - Total files: {len(files)}")
            
            if files:
                print("   - Sample files:")
                for file in files[:5]:  # Show first 5 files
                    print(f"     • {file}")
                if len(files) > 5:
                    print(f"     ... and {len(files) - 5} more files")
            
            return True
        else:
            print("❌ Dataset 'Guardian_Dataset' not found in ClearML")
            return False
            
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")
        return False

def check_local_dataset():
    """Check local dataset structure."""
    print("\n📁 Checking Local Dataset...")
    print("=" * 50)
    
    script_dir = pathlib.Path(__file__).resolve().parent
    dataset_path = script_dir / "data" / "Guardian_Dataset"
    
    print(f"Looking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print("❌ Local dataset directory does not exist")
        return False
    
    print("✅ Local dataset directory exists")
    
    # Check for action directories
    expected_classes = ["Falling", "No Action", "Waving"]
    found_classes = []
    
    for class_name in expected_classes:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            keypoint_files = list(class_dir.glob("*_keypoints.json"))
            json_files = list(class_dir.glob("*.json"))
            print(f"✅ {class_name}: {len(keypoint_files)} keypoint files, {len(json_files)} total JSON files")
            found_classes.append(class_name)
        else:
            print(f"❌ {class_name}: Directory not found")
    
    # Check for any other directories
    all_dirs = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    other_dirs = [d for d in all_dirs if d not in expected_classes]
    if other_dirs:
        print(f"📂 Other directories found: {other_dirs}")
    
    # Count total files
    all_json_files = list(dataset_path.rglob("*.json"))
    all_keypoint_files = list(dataset_path.rglob("*_keypoints.json"))
    
    print(f"\n📊 Summary:")
    print(f"   - Expected classes found: {len(found_classes)}/3")
    print(f"   - Total JSON files: {len(all_json_files)}")
    print(f"   - Total keypoint files: {len(all_keypoint_files)}")
    
    return len(found_classes) > 0 or len(all_json_files) > 0

def simulate_download_function():
    """Simulate the download function to see where it fails."""
    print("\n🔄 Simulating Download Function...")
    print("=" * 50)
    
    script_dir = pathlib.Path(__file__).resolve().parent
    dataset_path = script_dir / "data" / "Guardian_Dataset"
    
    # Step 1: Check local dataset
    print("Step 1: Checking local dataset...")
    if dataset_path.exists():
        expected_classes = ["Falling", "No Action", "Waving"]
        has_action_structure = any((dataset_path / folder).exists() for folder in expected_classes)
        
        if has_action_structure:
            print("✅ Local dataset with expected structure found")
            return str(dataset_path)
        else:
            print("⚠️  Local dataset exists but missing expected structure")
    else:
        print("❌ No local dataset found")
    
    # Step 2: Try ClearML download
    print("\nStep 2: Attempting ClearML download...")
    try:
        dataset = Dataset.get(
            dataset_name="Guardian_Dataset",
            dataset_project="Guardian_Training",
            only_completed=True
        )
        
        if dataset is None:
            print("❌ Dataset not found in ClearML")
            print("This is why the download function returns None!")
            return None
        else:
            print("✅ Dataset found in ClearML")
            print("Download would proceed normally")
            return "would_download"
            
    except Exception as e:
        print(f"❌ ClearML error: {e}")
        print("This is why the download function returns None!")
        return None

def suggest_solutions():
    """Suggest solutions based on the diagnosis."""
    print("\n💡 Suggested Solutions...")
    print("=" * 50)
    
    print("Based on the diagnosis above, here are your options:")
    print()
    
    print("🚀 Option 1: Quick Test (Recommended)")
    print("   The pipeline now has mock data fallback.")
    print("   Just run the pipeline - it will create test data automatically.")
    print()
    
    print("📤 Option 2: Upload a Test Dataset")
    print("   Run: python upload_dataset_simple.py")
    print("   This will create and upload a minimal test dataset.")
    print()
    
    print("🔧 Option 3: Fix ClearML Configuration")
    print("   If ClearML connection failed:")
    print("   1. Check your GitHub Secrets are correctly set")
    print("   2. Verify ClearML credentials on your runner")
    print("   3. Test connection manually")
    print()
    
    print("📁 Option 4: Use Your Real Dataset")
    print("   If you have a real dataset:")
    print("   1. Place it in data/Guardian_Dataset/ with proper structure")
    print("   2. Run: python upload_dataset.py")
    print()

def main():
    """Run complete diagnosis."""
    print("🦾 Guardian AI Dataset Download Debugger")
    print("=" * 60)
    
    # Test 1: ClearML Connection
    clearml_ok = check_clearml_connection()
    
    # Test 2: Dataset Existence
    dataset_exists = False
    if clearml_ok:
        dataset_exists = check_dataset_exists()
    
    # Test 3: Local Dataset
    local_dataset_ok = check_local_dataset()
    
    # Test 4: Simulate download function
    download_result = simulate_download_function()
    
    # Summary
    print("\n📋 Diagnosis Summary")
    print("=" * 50)
    print(f"ClearML Connection: {'✅ OK' if clearml_ok else '❌ Failed'}")
    print(f"Dataset in ClearML: {'✅ Found' if dataset_exists else '❌ Not Found'}")
    print(f"Local Dataset: {'✅ Found' if local_dataset_ok else '❌ Not Found'}")
    print(f"Download Function: {'✅ Would Work' if download_result else '❌ Would Fail'}")
    
    # Provide solutions
    suggest_solutions()
    
    print("\n" + "=" * 60)
    print("Run this script to understand exactly what's happening!")

if __name__ == "__main__":
    main() 