#!/usr/bin/env python3
"""
Test script to verify ClearML dataset access.
This helps debug dataset access issues in GitHub Actions.
"""

from clearml import Dataset
import sys

def test_clearml_dataset_access():
    """Test ClearML dataset access and list available datasets."""
    
    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    
    print(f"🔍 Testing ClearML dataset access...")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset project: {dataset_project}")
    print("-" * 50)
    
    try:
        # Test 1: List all datasets in the project
        print("📋 Test 1: Listing datasets in project...")
        datasets = Dataset.list_datasets(dataset_project=dataset_project)
        
        if datasets:
            print(f"✅ Found {len(datasets)} datasets:")
            for i, ds in enumerate(datasets):
                try:
                    if hasattr(ds, 'name') and hasattr(ds, 'id'):
                        print(f"  {i+1}. Name: '{ds.name}' | ID: {ds.id}")
                        if hasattr(ds, 'status'):
                            print(f"      Status: {ds.status}")
                    elif isinstance(ds, dict):
                        print(f"  {i+1}. Dict: {ds}")
                    else:
                        print(f"  {i+1}. Object: {type(ds)} - {str(ds)}")
                except Exception as e:
                    print(f"  {i+1}. Error accessing dataset info: {e}")
        else:
            print("❌ No datasets found in project")
            
    except Exception as e:
        print(f"❌ Error listing datasets: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 50)
    
    try:
        # Test 2: Try to get the specific dataset
        print("🎯 Test 2: Getting specific dataset...")
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True
        )
        
        if dataset:
            print(f"✅ Successfully retrieved dataset!")
            print(f"   ID: {dataset.id}")
            print(f"   Name: {dataset.name}")
            if hasattr(dataset, 'status'):
                print(f"   Status: {dataset.status}")
            
            # Test 3: Try to get file list
            print("\n📁 Test 3: Getting file list...")
            try:
                files = dataset.list_files()
                print(f"✅ Dataset contains {len(files)} files")
                
                # Show first few files
                for i, file_path in enumerate(files[:5]):
                    print(f"   {i+1}. {file_path}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
                    
            except Exception as e:
                print(f"❌ Error getting file list: {e}")
                
        else:
            print(f"❌ Dataset.get() returned None")
            print("This could mean:")
            print("  - Dataset doesn't exist")
            print("  - Dataset is not completed/finalized")
            print("  - Wrong dataset name or project")
            
    except Exception as e:
        print(f"❌ Error getting dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 50)
    
    try:
        # Test 4: Try without only_completed flag
        print("🔄 Test 4: Getting dataset without only_completed flag...")
        dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=False
        )
        
        if dataset:
            print(f"✅ Found dataset without only_completed flag!")
            print(f"   ID: {dataset.id}")
            print(f"   Name: {dataset.name}")
            if hasattr(dataset, 'status'):
                print(f"   Status: {dataset.status}")
        else:
            print(f"❌ Still no dataset found")
            
    except Exception as e:
        print(f"❌ Error getting dataset without only_completed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_clearml_dataset_access() 