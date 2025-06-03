#!/usr/bin/env python3
"""
MongoDB Model Distribution System for Guardian AI
-------------------------------------------------

This module provides a complete system for storing and retrieving ML models in MongoDB.
It is designed to work in both local development and CI/CD environments (GitHub Actions).

Key features:
- Store model weights in MongoDB GridFS
- Track model metadata and version history
- Retrieve best performing models based on accuracy
- Compatible with GitHub Actions using repository secrets
- Robust error handling and connection management

Usage:
    # Upload a model (typically called from the pipeline)
    result = upload_guardian_model(model_path="path/to/model.pth", training_metadata={
        "test_accuracy": 95.2, 
        "training_task_id": "abc123"
    })

    # Download best model (for deployment)
    success = download_best_guardian_model(download_path="./best_model.pth")
"""

import os
import io
import ssl
import torch
import gridfs
import logging
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, BinaryIO, Union
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
import certifi # Added import for certifi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default database settings
DEFAULT_DATABASE = 'guardian_ai'
DEFAULT_COLLECTION = 'distributed_models'

def get_mongodb_connection_info() -> Dict[str, str]:
    """
    Get MongoDB connection info directly from environment variables.
    Designed for GitHub Actions where repository secrets are available as env vars.
    
    In GitHub Actions workflow:
    - MONGODB_URI is passed as an environment variable from repository secrets
    
    Returns:
        Dict[str, str]: Dictionary containing URI, database name, and collection name
    """
    # Dictionary to hold connection info
    connection_info = {
        'uri': os.environ.get('MONGODB_URI'),  # Using os.environ to directly match GitHub Actions
        'database': os.environ.get('MONGODB_DATABASE', DEFAULT_DATABASE),
        'collection': os.environ.get('MONGODB_COLLECTION', DEFAULT_COLLECTION)
    }
    
    # Log status
    if connection_info['uri']:
        logger.info(f"✅ MongoDB connection info available from environment variables. Database: {connection_info['database']}")
    else:
        # Only try .secrets file if environment variables aren't available (for local development)
        secrets_file = '.secrets'
        if os.path.exists(secrets_file):
            try:
                with open(secrets_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Map keys to connection_info dictionary
                            if key == 'MONGODB_URI' and not connection_info['uri']:
                                connection_info['uri'] = value
                            elif key == 'MONGODB_DATABASE' and connection_info['database'] == DEFAULT_DATABASE:
                                connection_info['database'] = value
                            elif key == 'MONGODB_COLLECTION' and connection_info['collection'] == DEFAULT_COLLECTION:
                                connection_info['collection'] = value
                
                if connection_info['uri']:
                    logger.info(f"✅ MongoDB connection info loaded from .secrets file. Database: {connection_info['database']}")
            except Exception as e:
                logger.warning(f"Failed to read secrets file: {e}")
                
    # Final check if URI is available
    if not connection_info['uri']:
        logger.warning("⚠️ MongoDB URI not found in environment or secrets file.")
        logger.warning("💡 For GitHub Actions, ensure MONGODB_URI is set in repository secrets")
    
    return connection_info

class GuardianModelDistribution:
    """
    MongoDB-based model distribution system for Guardian AI.
    
    This class provides methods to:
    - Connect to MongoDB
    - Upload models to GridFS
    - Download models from GridFS
    - Track model metadata
    - Find the best performing model
    """
    
    def __init__(self, uri: str = None, database: str = None):
        """
        Initialize MongoDB model distribution.
        
        Args:
            uri (str, optional): MongoDB connection URI. Defaults to environment variable.
            database (str, optional): MongoDB database name. Defaults to environment variable or "guardian_ai".
        """
        if uri and database:
            self.uri = uri
            self.database_name = database
            logger.info(f"GuardianModelDistribution initialized with provided URI and database: {database}")
        else:
            # Only fetch from env/secrets if not provided explicitly
            connection_info = get_mongodb_connection_info()
            self.uri = uri or connection_info['uri']
            self.database_name = database or connection_info['database']
        
        if not self.uri:
            logger.warning("⚠️ MongoDB URI not provided. Set MONGODB_URI environment variable.")
            # Don't raise an exception here, let connect() handle it
        
        self.client = None
        self.database = None
        self.models_collection = None
        self.gridfs = None  # For storing large model files
        
    def connect(self) -> bool:
        """
        Connect to MongoDB cluster with robust error handling.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.uri:
            logger.error("❌ MongoDB URI not provided. Set MONGODB_URI environment variable.")
            return False
        
        try:
            logger.info("🔗 Connecting to MongoDB for model distribution...")
            
            # Create client with more robust connection parameters
            # Using the correct parameters for newer pymongo versions
            self.client = MongoClient(
                self.uri,
                server_api=ServerApi('1'),
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,         # 10 second connection timeout
                socketTimeoutMS=45000,          # 45 second socket timeout
                # TLS/SSL settings for newer pymongo versions
                tls=True,                       # Enable TLS/SSL
                tlsCAFile=certifi.where(),      # Use certifi's CA bundle
                tlsAllowInvalidCertificates=True,  # Allow self-signed certificates (if needed for some envs)
                tlsAllowInvalidHostnames=True   # Allow hostname mismatch (if needed for some envs)
            )
            
            # Test connection with timeout
            self.client.admin.command('ping', serverSelectionTimeoutMS=5000)
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
            
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failure: {e}")
            logger.error("💡 Check your network connection and MongoDB URI")
            return False
        except ServerSelectionTimeoutError as e:
            logger.error(f"❌ MongoDB server selection timeout: {e}")
            logger.error("💡 Check your MongoDB Atlas IP access list and network settings")
            return False
        except OperationFailure as e:
            logger.error(f"❌ MongoDB operation failure: {e}")
            logger.error("💡 Check your MongoDB credentials and permissions")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            # More detailed error reporting
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Clean up resources
            if self.client:
                try:
                    self.client.close()
                except:
                    pass
                self.client = None
            
            self.database = None
            self.models_collection = None
            self.gridfs = None
            
            return False
    
    def generate_download_url(self, model_name: str = None, model_id: str = None, expires_hours: int = 24) -> Optional[str]:
        """
        Generate a MongoDB GridFS download URL for a model.
        
        Args:
            model_name (str, optional): Name of the model to download. Defaults to None.
            model_id (str, optional): ID of the model to download. Defaults to None.
            expires_hours (int, optional): URL expiration in hours. Defaults to 24.
            
        Returns:
            Optional[str]: Download URL or None if failed
        """
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
            # Try to get MongoDB URI from environment directly for CI/CD
            uri = os.environ.get('MONGODB_URI') 
            
            # Generate download reference using best available method
            if uri and 'mongodb+srv://' in uri:
                uri_parts = uri.replace('mongodb+srv://', '').split('@')
                if len(uri_parts) >= 2:
                    cluster_info = uri_parts[1].split('/')[0]
                    
                    # Create direct GridFS download URL
                    download_url = f"https://{cluster_info}/gridfs/{self.database_name}/files/{file_id}"
                    
                    logger.info(f"🔗 MongoDB GridFS URL generated:")
                    logger.info(f"   URL: {download_url}")
                    logger.info(f"   File ID: {file_id}")
                    
                    return download_url
            
            # Fallback: provide instructions for manual download
            download_url = f"mongodb-gridfs://{self.database_name}/files/{file_id}"
            logger.info(f"🔗 GridFS reference generated:")
            logger.info(f"   Reference: {download_url}")
            logger.info(f"   Use download_guardian_model.py to access this file")
            
            return download_url
            
        except Exception as e:
            logger.error(f"❌ Failed to generate download URL: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def upload_model(self, 
                    model_path: str,
                    model_metadata: Dict[str, Any],
                    model_name: str = None) -> Optional[Dict[str, str]]:
        """
        Upload a model file to MongoDB for distribution.
        
        Args:
            model_path (str): Path to the model file
            model_metadata (Dict[str, Any]): Model metadata to store
            model_name (str, optional): Custom name for the model. Defaults to None.
            
        Returns:
            Optional[Dict[str, str]]: Upload result info or None if failed
        """
        if not self.gridfs:
            logger.error("❌ Not connected to MongoDB. Call connect() first.")
            return None
        
        try:
            logger.info(f"📤 Uploading model: {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file does not exist: {model_path}")
                return None
                
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
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def download_model(self, 
                      model_name: str = None,
                      model_id: str = None,
                      download_path: str = "./downloaded_model.pth") -> bool:
        """
        Download a model from MongoDB.
        
        Args:
            model_name (str, optional): Name of the model to download. Defaults to None.
            model_id (str, optional): ID of the model to download. Defaults to None.
            download_path (str, optional): Local path to save the model. Defaults to "./downloaded_model.pth".
            
        Returns:
            bool: True if download successful, False otherwise
        """
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
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(download_path)), exist_ok=True)
            
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
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the best performing model based on test accuracy.
        
        Returns:
            Optional[Dict[str, Any]]: Best model info or None if not found
        """
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
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def download_best_model(self, download_path: str = "./best_guardian_model.pth") -> bool:
        """
        Download the best performing model (highest test accuracy).
        
        Args:
            download_path (str, optional): Local path to save the model. Defaults to "./best_guardian_model.pth".
            
        Returns:
            bool: True if download successful, False otherwise
        """
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
        """
        List all available models for download.
        
        Returns:
            List[Dict[str, Any]]: List of model info dictionaries
        """
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
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from MongoDB.
        
        Args:
            model_name (str): Name of the model to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
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
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def close(self):
        """Close MongoDB connection and release resources."""
        if self.client:
            try:
                self.client.close()
                logger.info("🔒 MongoDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing MongoDB connection: {e}")

def upload_guardian_model(model_path: str, training_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload a Guardian model for team distribution.
    
    This is the main function to call from the pipeline to upload a trained model
    to MongoDB for distribution to other team members.
    
    Args:
        model_path (str): Path to the model file
        training_metadata (Dict[str, Any]): Model metadata, including test_accuracy
        
    Returns:
        Dict[str, Any]: Upload result with success status
    """
    logger.info("📤 Uploading Guardian model for distribution...")
    
    # Get MongoDB connection info directly from environment variables (for GitHub Actions)
    connection_info = get_mongodb_connection_info()
    
    # Check if we have MongoDB URI
    if not connection_info.get('uri'):
        error_msg = "MongoDB URI not found in environment variables or .secrets file"
        logger.error(f"❌ {error_msg}")
        return {"error": error_msg, "success": False}
    
    # Initialize distribution system
    distributor = GuardianModelDistribution(
        uri=connection_info['uri'],
        database=connection_info['database']
    )
    
    try:
        if not distributor.connect():
            error_msg = "Failed to connect to MongoDB"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "success": False}
        
        # Generate model name
        model_name = f"guardian_bilstm_{training_metadata.get('test_accuracy', 0):.1f}pct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at {model_path}"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg, "success": False}
        
        # Upload model
        result = distributor.upload_model(
            model_path=model_path,
            model_metadata=training_metadata,
            model_name=model_name
        )
        
        if result:
            return {**result, "success": True}
        else:
            return {"error": "Upload failed", "success": False}
        
    except Exception as e:
        error_msg = f"Error uploading model: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return {"error": error_msg, "success": False}
    finally:
        distributor.close()

def download_best_guardian_model(download_path: str = "./best_guardian_model.pth") -> bool:
    """
    Download the best Guardian model for your team.
    
    This function finds and downloads the model with the highest test accuracy.
    
    Args:
        download_path (str, optional): Local path to save the model. Defaults to "./best_guardian_model.pth".
        
    Returns:
        bool: True if download successful, False otherwise
    """
    logger.info("🏆 Downloading best Guardian model...")
    
    # Get MongoDB connection info directly from environment variables (for GitHub Actions)
    connection_info = get_mongodb_connection_info()
    
    # Check if we have MongoDB URI
    if not connection_info.get('uri'):
        logger.error("❌ MongoDB URI not found in environment variables or .secrets file")
        return False
    
    # Initialize distribution system
    distributor = GuardianModelDistribution(
        uri=connection_info['uri'],
        database=connection_info['database']
    )
    
    try:
        if not distributor.connect():
            logger.error("❌ Failed to connect to MongoDB")
            return False
        
        return distributor.download_best_model(download_path)
        
    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return False
    finally:
        distributor.close()

def main():
    """Demo of model distribution system."""
    
    print("🛡️ Guardian AI - Model Distribution System")
    print("=" * 60)
    
    # Get MongoDB connection info directly from environment variables (for GitHub Actions)
    connection_info = get_mongodb_connection_info()
    
    # Check if we have MongoDB URI
    if not connection_info.get('uri'):
        print("❌ MongoDB URI not found in environment variables or .secrets file")
        print("💡 For GitHub Actions, set MONGODB_URI in repository secrets")
        return
    
    # Initialize system
    distributor = GuardianModelDistribution(
        uri=connection_info['uri'],
        database=connection_info['database']
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
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
    finally:
        distributor.close()

if __name__ == "__main__":
    main() 