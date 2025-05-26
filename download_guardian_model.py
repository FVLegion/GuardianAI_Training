#!/usr/bin/env python3
"""
Guardian Model Downloader
Simple tool for team members to download trained Guardian models from anywhere
"""

import os
import sys
import argparse
import logging
from mongodb_model_distribution import GuardianModelDistribution, load_secrets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def list_models():
    """List all available models."""
    
    print("🛡️ Guardian AI - Available Models")
    print("=" * 60)
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Get models
        models = distributor.list_available_models()
        
        if not models:
            print("📊 No models available for download")
            return True
        
        print(f"📋 Found {len(models)} available models:\n")
        
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['model_name']}")
            print(f"     📊 Accuracy: {model.get('test_accuracy', 'N/A')}%")
            print(f"     📦 Size: {model.get('file_size_mb', 'N/A'):.2f} MB")
            print(f"     📥 Downloads: {model.get('download_count', 0)}")
            print(f"     📅 Uploaded: {model.get('uploaded_at', 'N/A')}")
            print(f"     🏷️  Task ID: {model.get('training_task_id', 'N/A')}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        distributor.close()

def download_best():
    """Download the best model."""
    
    print("🏆 Guardian AI - Downloading Best Model")
    print("=" * 50)
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Get best model info
        best_model = distributor.get_best_model_info()
        
        if not best_model:
            print("📊 No models available")
            return False
        
        print(f"🎯 Best model found:")
        print(f"   Name: {best_model['model_name']}")
        print(f"   Accuracy: {best_model.get('test_accuracy', 'N/A')}%")
        print(f"   Size: {best_model.get('file_size_mb', 'N/A'):.2f} MB")
        print()
        
        # Download it
        print(f"🔍 Downloading best model {best_model['model_name']}...")
        download_path = f"./Guardian_Best_Model.pth"
        success = distributor.download_best_model(download_path)
        
        if success:
            print(f"✅ Best model downloaded successfully!")
            print(f"📁 Saved as: {download_path}")
            return True
        else:
            print("❌ Failed to download best model")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        distributor.close()

def download_specific(model_name: str):
    """Download a specific model by name."""
    
    print(f"📥 Guardian AI - Downloading Model: {model_name}")
    print("=" * 60)
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Download the model
        download_path = f"./{model_name}.pth"
        success = distributor.download_model(
            model_name=model_name,
            download_path=download_path
        )
        
        if success:
            print(f"✅ Model downloaded successfully!")
            print(f"📁 Saved as: {download_path}")
            return True
        else:
            print(f"❌ Failed to download model: {model_name}")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        distributor.close()

def interactive_download():
    """Interactive model download."""
    
    print("🛡️ Guardian AI - Interactive Model Download")
    print("=" * 60)
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            print("❌ Failed to connect to MongoDB")
            return False
        
        # Get available models
        models = distributor.list_available_models()
        
        if not models:
            print("📊 No models available for download")
            return False
        
        # Show models
        print("\n📋 Available Models:")
        for i, model in enumerate(models, 1):
            print(f"{i:2d}. {model['model_name']} - {model.get('test_accuracy', 'N/A')}% accuracy")
        
        print(f"{len(models)+1:2d}. Download best model")
        print(f"{len(models)+2:2d}. Exit")
        
        # Get user choice
        try:
            choice = int(input(f"\nSelect model to download (1-{len(models)+2}): "))
        except ValueError:
            print("❌ Invalid choice")
            return False
        
        if choice == len(models) + 2:  # Exit
            print("👋 Goodbye!")
            return True
        elif choice == len(models) + 1:  # Best model
            return download_best()
        elif 1 <= choice <= len(models):  # Specific model
            selected_model = models[choice - 1]
            download_path = f"./{selected_model['model_name']}.pth"
            
            success = distributor.download_model(
                model_name=selected_model['model_name'],
                download_path=download_path
            )
            
            if success:
                print(f"✅ Model downloaded successfully!")
                print(f"📁 Saved as: {download_path}")
                return True
            else:
                print("❌ Download failed")
                return False
        else:
            print("❌ Invalid choice")
            return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    finally:
        distributor.close()

def main():
    """Main function with command line interface."""
    
    parser = argparse.ArgumentParser(description='Download Guardian AI models from MongoDB')
    parser.add_argument('--list', action='store_true', help='List all available models')
    parser.add_argument('--best', action='store_true', help='Download the best model')
    parser.add_argument('--model', type=str, help='Download specific model by name')
    parser.add_argument('--interactive', action='store_true', help='Interactive download mode')
    
    args = parser.parse_args()
    
    # Check if secrets file exists
    if not os.path.exists('.secrets'):
        print("❌ Error: .secrets file not found!")
        print("🔧 Please ensure you have MongoDB credentials in .secrets file")
        sys.exit(1)
    
    # Execute based on arguments
    if args.list:
        success = list_models()
    elif args.best:
        success = download_best()
    elif args.model:
        success = download_specific(args.model)
    elif args.interactive:
        success = interactive_download()
    else:
        # Default to interactive mode
        print("🛡️ Guardian AI Model Downloader")
        print("=" * 40)
        print("Usage examples:")
        print("  python download_guardian_model.py --list              # List all models")
        print("  python download_guardian_model.py --best              # Download best model")
        print("  python download_guardian_model.py --model MODEL_NAME  # Download specific model")
        print("  python download_guardian_model.py --interactive       # Interactive mode")
        print()
        success = interactive_download()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 