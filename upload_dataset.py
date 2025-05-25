#!/usr/bin/env python3
"""
Upload Guardian Dataset to ClearML for CI/CD pipeline access.
Run this script once to upload your local dataset to ClearML.
"""

from clearml import Dataset
import os
import pathlib

def upload_guardian_dataset():
    """Upload the local Guardian dataset to ClearML."""
    
    # Define paths
    script_dir = pathlib.Path(__file__).resolve().parent
    local_dataset_path = script_dir / "data" / "Guardian_Dataset"
    
    print(f"🔍 Checking local dataset at: {local_dataset_path}")
    
    # Verify local dataset exists
    if not local_dataset_path.exists():
        print(f"❌ Local dataset not found at: {local_dataset_path}")
        print("Please ensure your dataset is in the 'data/Guardian_Dataset' directory")
        return False
    
    # Check for expected action directories
    expected_actions = ["Falling", "No Action", "Waving"]
    missing_actions = []
    
    for action in expected_actions:
        action_dir = local_dataset_path / action
        if not action_dir.exists():
            missing_actions.append(action)
        else:
            # Count files in each action directory
            json_files = list(action_dir.glob("*_keypoints.json"))
            print(f"✅ Found {len(json_files)} keypoint files in '{action}' directory")
    
    if missing_actions:
        print(f"❌ Missing action directories: {missing_actions}")
        return False
    
    # Count total files
    total_files = list(local_dataset_path.rglob("*_keypoints.json"))
    total_size = sum(f.stat().st_size for f in local_dataset_path.rglob("*") if f.is_file())
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"📊 Dataset Statistics:")
    print(f"   - Total keypoint files: {len(total_files)}")
    print(f"   - Total size: {total_size_mb:.1f} MB")
    
    # Create ClearML dataset
    print(f"\n🚀 Creating ClearML dataset...")
    
    try:
        # Check if dataset already exists
        try:
            existing_dataset = Dataset.get(
                dataset_name="Guardian_Dataset",
                dataset_project="Guardian_Training",
                only_completed=True
            )
            if existing_dataset:
                print(f"⚠️  Dataset 'Guardian_Dataset' already exists in ClearML")
                response = input("Do you want to create a new version? (y/N): ").strip().lower()
                if response != 'y':
                    print("Upload cancelled.")
                    return True
        except:
            print("✅ No existing dataset found, creating new one...")
        
        # Create new dataset
        dataset = Dataset.create(
            dataset_name="Guardian_Dataset",
            dataset_project="Guardian_Training",
            dataset_tags=["pose-estimation", "action-recognition", "guardian-ai"]
        )
        
        print(f"📁 Adding files from: {local_dataset_path}")
        dataset.add_files(path=str(local_dataset_path))
        
        print(f"⬆️  Uploading dataset to ClearML...")
        dataset.upload()
        
        print(f"✅ Finalizing dataset...")
        dataset.finalize()
        
        print(f"\n🎉 Dataset uploaded successfully!")
        print(f"   - Dataset ID: {dataset.id}")
        print(f"   - Dataset Name: Guardian_Dataset")
        print(f"   - Project: Guardian_Training")
        print(f"   - Files uploaded: {len(total_files)}")
        print(f"   - Size: {total_size_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False

if __name__ == "__main__":
    print("🦾 Guardian AI Dataset Uploader")
    print("=" * 50)
    
    success = upload_guardian_dataset()
    
    if success:
        print("\n✅ Dataset upload completed!")
        print("Your GitHub Actions pipeline can now access the dataset.")
    else:
        print("\n❌ Dataset upload failed!")
        print("Please check the error messages above and try again.") 