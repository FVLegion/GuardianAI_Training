#!/usr/bin/env python3
"""
MongoDB Model Distribution System for Guardian AI
Stores actual model files in MongoDB for team access and downloading
"""

import os
import io
import torch
import gridfs
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, BinaryIO
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def load_secrets(secrets_file: str = '.secrets') -> dict:
    """Load secrets from .secrets file."""
    secrets = {}
    
    if not os.path.exists(secrets_file):
        raise FileNotFoundError(f"Secrets file {secrets_file} not found")
    
    with open(secrets_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                secrets[key.strip()] = value.strip()
    
    return secrets

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GuardianModelDistribution:
    """MongoDB-based model distribution system for Guardian AI."""
    
    def __init__(self, uri: str = None, database: str = "guardian_ai"):
        """Initialize MongoDB model distribution."""
        
        self.uri = uri or os.getenv('MONGODB_URI')
        self.database_name = database
        
        if not self.uri:
            raise ValueError("MongoDB URI not provided. Set MONGODB_URI environment variable.")
        
        self.client = None
        self.database = None
        self.models_collection = None
        self.gridfs = None  # For storing large model files
        
    def connect(self) -> bool:
        """Connect to MongoDB cluster."""
        
        try:
            logger.info("🔗 Connecting to MongoDB for model distribution...")
            
            # Create client
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB!")
            
            # Set up database and collections
            self.database = self.client[self.database_name]
            self.models_collection = self.database['distributed_models']
            
            # GridFS for large file storage
            self.gridfs = gridfs.GridFS(self.database)
            
            logger.info(f"📁 Using database: {self.database_name}")
            logger.info(f"📄 Models collection: distributed_models")
            logger.info(f"🗂️ GridFS initialized for file storage")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    def generate_download_url(self, model_name: str = None, model_id: str = None, expires_hours: int = 24) -> Optional[str]:
        """Generate a MongoDB GridFS download URL for a model."""
        
        if self.models_collection is None:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return None
        
        try:
            # Find model document
            if model_id:
                model_doc = self.models_collection.find_one({"_id": model_id})
            elif model_name:
                model_doc = self.models_collection.find_one({"model_name": model_name})
            else:
                logger.error("❌ Either model_name or model_id must be provided")
                return None
            
            if not model_doc:
                logger.error(f"❌ Model not found: {model_name or model_id}")
                return None
            
            # Get file_id for direct MongoDB access
            file_id = model_doc['file_id']
            
            # Generate MongoDB GridFS download URL
            # Format: mongodb+srv://username:password@cluster/database/gridfs/files/{file_id}
            secrets = load_secrets()
            uri_parts = secrets.get('MONGODB_URI', '').replace('mongodb+srv://', '').split('@')
            if len(uri_parts) >= 2:
                credentials = uri_parts[0]
                cluster_info = uri_parts[1].split('/')[0]
                
                # Create direct GridFS download URL
                download_url = f"https://{cluster_info}/gridfs/{self.database_name}/files/{file_id}"
                
                logger.info(f"🔗 MongoDB GridFS URL generated:")
                logger.info(f"   URL: {download_url}")
                logger.info(f"   File ID: {file_id}")
                
                return download_url
            else:
                # Fallback: provide instructions for manual download
                download_url = f"mongodb-gridfs://{self.database_name}/files/{file_id}"
                logger.info(f"🔗 GridFS reference generated:")
                logger.info(f"   Reference: {download_url}")
                logger.info(f"   Use download_guardian_model.py to access this file")
                
                return download_url
            
        except Exception as e:
            logger.error(f"❌ Failed to generate download URL: {e}")
            return None
    
    def upload_model(self, 
                    model_path: str,
                    model_metadata: Dict[str, Any],
                    model_name: str = None) -> Optional[Dict[str, str]]:
        """Upload a model file to MongoDB for distribution."""
        
        if not self.gridfs:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return None
        
        try:
            logger.info(f"📤 Uploading model: {model_path}")
            
            # Generate model name if not provided
            if not model_name:
                model_name = f"guardian_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Read model file
            with open(model_path, 'rb') as model_file:
                # Store file in GridFS
                file_id = self.gridfs.put(
                    model_file,
                    filename=f"{model_name}.pth",
                    metadata={
                        "uploaded_at": datetime.utcnow(),
                        "original_path": model_path,
                        "model_name": model_name,
                        **model_metadata
                    }
                )
            
            # Store model info in collection
            model_doc = {
                "model_name": model_name,
                "file_id": file_id,
                "uploaded_at": datetime.utcnow(),
                "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
                "status": "available",
                "download_count": 0,
                **model_metadata
            }
            
            result = self.models_collection.insert_one(model_doc)
            doc_id = str(result.inserted_id)
            
            # Note: MongoDB GridFS requires programmatic access
            # Use download_guardian_model.py for downloads
            
            logger.info(f"✅ Model uploaded successfully!")
            logger.info(f"   Model Name: {model_name}")
            logger.info(f"   Document ID: {doc_id}")
            logger.info(f"   File ID: {file_id}")
            logger.info(f"   Size: {model_doc['file_size_mb']:.2f} MB")
            logger.info(f"   Download command: python download_guardian_model.py --model {model_name}")
            
            return {
                "document_id": doc_id,
                "model_name": model_name,
                "file_id": str(file_id),
                "download_command": f"python download_guardian_model.py --model {model_name}"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model: {e}")
            return None
    
    def download_model(self, 
                      model_name: str = None,
                      model_id: str = None,
                      download_path: str = "./downloaded_model.pth") -> bool:
        """Download a model from MongoDB."""
        
        if not self.gridfs:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return False
        
        try:
            # Find model document
            if model_id:
                model_doc = self.models_collection.find_one({"_id": model_id})
            elif model_name:
                model_doc = self.models_collection.find_one({"model_name": model_name})
            else:
                logger.error("❌ Either model_name or model_id must be provided")
                return False
            
            if not model_doc:
                logger.error(f"❌ Model not found: {model_name or model_id}")
                return False
            
            logger.info(f"📥 Downloading model: {model_doc['model_name']}")
            
            # Get file from GridFS
            file_id = model_doc['file_id']
            grid_file = self.gridfs.get(file_id)
            
            # Write to local file
            with open(download_path, 'wb') as output_file:
                output_file.write(grid_file.read())
            
            # Update download count
            self.models_collection.update_one(
                {"_id": model_doc["_id"]},
                {"$inc": {"download_count": 1}}
            )
            
            logger.info(f"✅ Model downloaded successfully!")
            logger.info(f"   Saved to: {download_path}")
            logger.info(f"   Size: {model_doc['file_size_mb']:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download model: {e}")
            return False
    
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the best performing model."""
        
        if self.models_collection is None:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return None
        
        try:
            # Find model with highest test accuracy
            best_model = self.models_collection.find_one(
                {"test_accuracy": {"$exists": True}, "status": "available"},
                sort=[("test_accuracy", -1)]
            )
            
            if best_model:
                best_model["_id"] = str(best_model["_id"])
                logger.info(f"🏆 Best model: {best_model['model_name']} ({best_model.get('test_accuracy', 'N/A')}%)")
                return best_model
            else:
                logger.info("📊 No models found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get best model info: {e}")
            return None
    
    def download_best_model(self, download_path: str = "./best_guardian_model.pth") -> bool:
        """Download the best performing model."""
        
        logger.info("🏆 Finding and downloading best model...")
        
        # Get best model info
        best_model = self.get_best_model_info()
        
        if not best_model:
            logger.error("❌ No best model found")
            return False
        
        # Download it
        return self.download_model(
            model_name=best_model['model_name'],
            download_path=download_path
        )
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models for download."""
        
        if self.models_collection is None:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return []
        
        try:
            models = list(
                self.models_collection.find({"status": "available"})
                .sort("uploaded_at", -1)
            )
            
            # Convert ObjectId to string
            for model in models:
                model["_id"] = str(model["_id"])
            
            logger.info(f"📋 Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from MongoDB."""
        
        if not self.gridfs:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return False
        
        try:
            # Find model document
            model_doc = self.models_collection.find_one({"model_name": model_name})
            
            if not model_doc:
                logger.error(f"❌ Model not found: {model_name}")
                return False
            
            # Delete file from GridFS
            self.gridfs.delete(model_doc['file_id'])
            
            # Delete document
            self.models_collection.delete_one({"_id": model_doc["_id"]})
            
            logger.info(f"🗑️ Model deleted: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model: {e}")
            return False
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("🔒 MongoDB connection closed")

def upload_guardian_model(model_path: str, training_metadata: Dict[str, Any]) -> Dict[str, str]:
    """Upload a Guardian model for team distribution."""
    
    logger.info("📤 Uploading Guardian model for distribution...")
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize distribution system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            return False
        
        # Generate model name
        model_name = f"guardian_bilstm_{training_metadata.get('test_accuracy', 0):.1f}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Upload model
        result = distributor.upload_model(
            model_path=model_path,
            model_metadata=training_metadata,
            model_name=model_name
        )
        
        return result or {"error": "Upload failed"}
        
    finally:
        distributor.close()

def download_best_guardian_model(download_path: str = "./best_guardian_model.pth") -> bool:
    """Download the best Guardian model for your team."""
    
    logger.info("🏆 Downloading best Guardian model...")
    
    # Load secrets
    secrets = load_secrets()
    
    # Initialize distribution system
    distributor = GuardianModelDistribution(
        uri=secrets.get('MONGODB_URI'),
        database=secrets.get('MONGODB_DATABASE', 'guardian_ai')
    )
    
    try:
        if not distributor.connect():
            return False
        
        return distributor.download_best_model(download_path)
        
    finally:
        distributor.close()

def main():
    """Demo of model distribution system."""
    
    print("🛡️ Guardian AI - Model Distribution System")
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
            return
        
        # List available models
        print("\n📋 Available Models:")
        models = distributor.list_available_models()
        
        if models:
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model['model_name']}")
                print(f"      Accuracy: {model.get('test_accuracy', 'N/A')}%")
                print(f"      Size: {model.get('file_size_mb', 'N/A'):.2f} MB")
                print(f"      Downloads: {model.get('download_count', 0)}")
                print()
        else:
            print("   No models available")
        
        # Show best model
        print("🏆 Best Model:")
        best_model = distributor.get_best_model_info()
        if best_model:
            print(f"   Name: {best_model['model_name']}")
            print(f"   Accuracy: {best_model.get('test_accuracy', 'N/A')}%")
            print(f"   Size: {best_model.get('file_size_mb', 'N/A'):.2f} MB")
        else:
            print("   No models found")
        
    finally:
        distributor.close()

if __name__ == "__main__":
    main() 