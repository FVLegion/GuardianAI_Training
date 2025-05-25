#!/usr/bin/env python3
"""
Simple script to upload Guardian Dataset to ClearML.
This script creates a minimal dataset if no local dataset exists.
"""

from clearml import Dataset
import os
import pathlib
import json

def create_minimal_dataset(dataset_path):
    """Create a minimal dataset with mock data for testing."""
    print("🔧 Creating minimal test dataset...")
    
    action_classes = ["Falling", "No Action", "Waving"]
    
    for action in action_classes:
        action_dir = dataset_path / action
        action_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2 mock files per action
        for i in range(2):
            mock_keypoints = []
            # Create 5 frames of mock data
            for frame in range(5):
                keypoints = []
                for joint in range(17):  # 17 COCO keypoints
                    x = 100 + joint * 10 + frame * 2
                    y = 100 + joint * 5 + frame * 3
                    confidence = 0.8
                    keypoints.extend([x, y, confidence])
                
                mock_keypoints.append({
                    "frame": frame,
                    "keypoints": [keypoints]
                })
            
            # Save mock file
            mock_file = action_dir / f"test_{action.lower().replace(' ', '_')}_{i}_keypoints.json"
            with open(mock_file, 'w') as f:
                json.dump(mock_keypoints, f)
    
    print(f"✅ Created minimal dataset at {dataset_path}")

def upload_dataset():
    """Upload dataset to ClearML."""
    
    # Define paths
    script_dir = pathlib.Path(__file__).resolve().parent
    dataset_path = script_dir / "data" / "Guardian_Dataset"
    
    print("🦾 Guardian AI Dataset Uploader")
    print("=" * 50)
    
    # Check if dataset exists locally
    if not dataset_path.exists():
        print(f"📁 Local dataset not found at: {dataset_path}")
        response = input("Create a minimal test dataset? (y/N): ").strip().lower()
        if response == 'y':
            create_minimal_dataset(dataset_path)
        else:
            print("❌ No dataset to upload. Exiting.")
            return False
    
    # Count files
    json_files = list(dataset_path.rglob("*_keypoints.json"))
    print(f"📊 Found {len(json_files)} keypoint files")
    
    if len(json_files) == 0:
        print("❌ No keypoint files found. Please check your dataset structure.")
        return False
    
    try:
        print("🚀 Creating ClearML dataset...")
        
        # Create dataset
        dataset = Dataset.create(
            dataset_name="Guardian_Dataset",
            dataset_project="Guardian_Training",
            dataset_tags=["pose-estimation", "action-recognition", "guardian-ai", "github-actions"]
        )
        
        print(f"📁 Adding files from: {dataset_path}")
        dataset.add_files(path=str(dataset_path))
        
        print(f"⬆️  Uploading to ClearML...")
        dataset.upload()
        
        print(f"✅ Finalizing dataset...")
        dataset.finalize()
        
        print(f"\n🎉 Dataset uploaded successfully!")
        print(f"   - Dataset ID: {dataset.id}")
        print(f"   - Dataset Name: Guardian_Dataset")
        print(f"   - Project: Guardian_Training")
        print(f"   - Files: {len(json_files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        print("This could be due to:")
        print("  1. ClearML credentials not configured")
        print("  2. Network connectivity issues")
        print("  3. Insufficient permissions")
        return False

if __name__ == "__main__":
    success = upload_dataset()
    
    if success:
        print("\n✅ Dataset upload completed!")
        print("Your GitHub Actions pipeline can now access the dataset.")
    else:
        print("\n❌ Dataset upload failed!")
        print("The pipeline will use mock data instead.") 