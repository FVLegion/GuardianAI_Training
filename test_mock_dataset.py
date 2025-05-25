#!/usr/bin/env python3
"""
Test script to verify mock dataset creation works correctly.
"""

import pathlib
import json
import os

def create_mock_dataset(local_path: str) -> bool:
    """Create a minimal mock dataset for testing when real dataset is unavailable."""
    try:
        local_path_obj = pathlib.Path(local_path)
        local_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Create action class directories
        action_classes = ["Falling", "No Action", "Waving"]
        
        for action in action_classes:
            action_dir = local_path_obj / action
            action_dir.mkdir(exist_ok=True)
            
            # Create a few mock keypoint files for each action
            for i in range(3):  # Create 3 mock files per action
                mock_keypoints = []
                # Create 10 frames of mock data
                for frame in range(10):
                    # Mock keypoints for 17 joints (COCO format)
                    keypoints = []
                    for joint in range(17):
                        x = 100 + joint * 10 + frame * 2  # Mock x coordinate
                        y = 100 + joint * 5 + frame * 3   # Mock y coordinate
                        confidence = 0.8 + (joint % 3) * 0.1  # Mock confidence
                        keypoints.extend([x, y, confidence])
                    
                    mock_keypoints.append({
                        "frame": frame,
                        "keypoints": [keypoints]  # Wrap in list for person detection
                    })
                
                # Save mock keypoints file
                mock_file = action_dir / f"mock_{action.lower().replace(' ', '_')}_{i}_keypoints.json"
                with open(mock_file, 'w') as f:
                    json.dump(mock_keypoints, f)
                print(f"Created: {mock_file}")
        
        print(f"✅ Created mock dataset at {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create mock dataset: {e}")
        return False

def test_mock_dataset():
    """Test the mock dataset creation and validation."""
    print("🧪 Testing Mock Dataset Creation")
    print("=" * 50)
    
    # Create test directory
    test_dir = pathlib.Path("test_mock_dataset")
    
    # Clean up if exists
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    
    # Create mock dataset
    success = create_mock_dataset(str(test_dir))
    
    if not success:
        print("❌ Mock dataset creation failed")
        return False
    
    # Validate the created dataset
    print("\n🔍 Validating created dataset...")
    
    expected_classes = ["Falling", "No Action", "Waving"]
    total_files = 0
    total_json_files = 0
    
    for class_name in expected_classes:
        class_dir = test_dir / class_name
        if class_dir.exists():
            keypoint_files = list(class_dir.glob("*keypoints.json"))
            json_files = list(class_dir.glob("*.json"))
            total_files += len(keypoint_files)
            total_json_files += len(json_files)
            print(f"✅ {class_name}: {len(keypoint_files)} keypoint files, {len(json_files)} JSON files")
        else:
            print(f"❌ {class_name}: Directory not found")
    
    print(f"\n📊 Summary:")
    print(f"   - Total keypoint files: {total_files}")
    print(f"   - Total JSON files: {total_json_files}")
    
    # Test file content
    if total_json_files > 0:
        sample_file = list(test_dir.rglob("*.json"))[0]
        print(f"\n📄 Testing file content: {sample_file.name}")
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            print(f"   - Frames: {len(data)}")
            print(f"   - First frame keys: {list(data[0].keys()) if data else 'No data'}")
            print("✅ File content is valid JSON")
        except Exception as e:
            print(f"❌ File content error: {e}")
    
    # Clean up
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print("\n🧹 Cleaned up test directory")
    
    return total_files > 0 and total_json_files > 0

if __name__ == "__main__":
    success = test_mock_dataset()
    
    if success:
        print("\n✅ Mock dataset creation test PASSED")
        print("The mock dataset creation function works correctly.")
    else:
        print("\n❌ Mock dataset creation test FAILED")
        print("There's an issue with the mock dataset creation function.") 