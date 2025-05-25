#!/usr/bin/env python3
"""
Local Pipeline Listener for Guardian AI Training

This script monitors for GitHub Action triggers and automatically runs
the Guardian AI pipeline on your local GPU machine when triggered.

Usage:
    python local_pipeline_listener.py

Features:
- Monitors GitHub repository for new commits/PRs
- Automatically pulls latest changes
- Runs Guardian pipeline with local GPU resources
- Sends results back to ClearML
- Optional webhook server for real-time triggers
"""

import os
import sys
import time
import json
import subprocess
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import requests
    from flask import Flask, request, jsonify
    import git
    from clearml import Task
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install requests flask gitpython clearml")
    sys.exit(1)

# Configuration
REPO_PATH = Path(__file__).parent
GITHUB_REPO = "your-username/GuardianAI_Training-6"  # Update this
POLL_INTERVAL = 60  # seconds
WEBHOOK_PORT = 8080
LOG_FILE = "pipeline_listener.log"

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GitHubMonitor:
    """Monitor GitHub repository for changes"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.repo = git.Repo(repo_path)
        self.last_commit = self.get_latest_commit()
        
    def get_latest_commit(self) -> str:
        """Get the latest commit hash"""
        try:
            return self.repo.head.commit.hexsha
        except Exception as e:
            logger.error(f"Error getting latest commit: {e}")
            return ""
    
    def check_for_updates(self) -> bool:
        """Check if there are new commits"""
        try:
            # Fetch latest changes
            self.repo.remotes.origin.fetch()
            
            # Get current commit
            current_commit = self.get_latest_commit()
            
            # Check if there's a new commit
            if current_commit != self.last_commit:
                logger.info(f"New commit detected: {current_commit}")
                self.last_commit = current_commit
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return False
    
    def pull_latest_changes(self) -> bool:
        """Pull the latest changes from the repository"""
        try:
            logger.info("Pulling latest changes...")
            self.repo.remotes.origin.pull()
            logger.info("SUCCESS: Successfully pulled latest changes")
            return True
        except Exception as e:
            logger.error(f"ERROR: Error pulling changes: {e}")
            return False

class PipelineRunner:
    """Run the Guardian AI pipeline"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.pipeline_script = repo_path / "Guardian_pipeline.py"
        
    def run_pipeline(self, trigger_info: Optional[Dict[str, Any]] = None) -> bool:
        """Run the Guardian AI pipeline"""
        try:
            logger.info("STARTING: Guardian AI pipeline...")
            
            # Change to repository directory
            os.chdir(self.repo_path)
            
            # Set environment variables if needed
            env = os.environ.copy()
            
            # Add trigger information to environment if available
            if trigger_info:
                env['GITHUB_TRIGGER_INFO'] = json.dumps(trigger_info)
                logger.info(f"Trigger info: {trigger_info}")
            
            # Run the pipeline
            start_time = datetime.now()
            result = subprocess.run(
                [sys.executable, str(self.pipeline_script)],
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"SUCCESS: Pipeline completed successfully in {duration}")
                logger.info("Pipeline output:")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"ERROR: Pipeline failed with return code {result.returncode}")
                logger.error("Pipeline stderr:")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("ERROR: Pipeline timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"ERROR: Error running pipeline: {e}")
            return False

class WebhookServer:
    """Simple webhook server to receive GitHub triggers"""
    
    def __init__(self, port: int, pipeline_runner: PipelineRunner, github_monitor: GitHubMonitor):
        self.port = port
        self.pipeline_runner = pipeline_runner
        self.github_monitor = github_monitor
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/webhook', methods=['POST'])
        def webhook():
            try:
                data = request.get_json()
                logger.info(f"Received webhook trigger: {data}")
                
                # Pull latest changes
                if self.github_monitor.pull_latest_changes():
                    # Run pipeline in background thread
                    thread = threading.Thread(
                        target=self.pipeline_runner.run_pipeline,
                        args=(data,)
                    )
                    thread.daemon = True
                    thread.start()
                    
                    return jsonify({
                        "status": "success",
                        "message": "Pipeline execution started",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        "status": "error",
                        "message": "Failed to pull latest changes"
                    }), 500
                    
            except Exception as e:
                logger.error(f"Webhook error: {e}")
                return jsonify({
                    "status": "error",
                    "message": str(e)
                }), 500
        
        @self.app.route('/status', methods=['GET'])
        def status():
            return jsonify({
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "repo_path": str(self.github_monitor.repo_path),
                "latest_commit": self.github_monitor.get_latest_commit()
            })
    
    def run(self):
        """Run the webhook server"""
        logger.info(f"Starting webhook server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False)

def main():
    """Main function"""
    logger.info("Starting Guardian AI Local Pipeline Listener")
    logger.info(f"Repository path: {REPO_PATH}")
    
    # Initialize components
    github_monitor = GitHubMonitor(REPO_PATH)
    pipeline_runner = PipelineRunner(REPO_PATH)
    
    # Check if we should run webhook server
    webhook_mode = "--webhook" in sys.argv
    
    if webhook_mode:
        # Run webhook server
        webhook_server = WebhookServer(WEBHOOK_PORT, pipeline_runner, github_monitor)
        logger.info(f"Webhook URL: http://localhost:{WEBHOOK_PORT}/webhook")
        logger.info(f"Status URL: http://localhost:{WEBHOOK_PORT}/status")
        webhook_server.run()
    else:
        # Run polling mode
        logger.info(f"Starting polling mode (interval: {POLL_INTERVAL}s)")
        logger.info("Use --webhook flag to run webhook server instead")
        
        try:
            while True:
                if github_monitor.check_for_updates():
                    logger.info("New changes detected, pulling and running pipeline...")
                    
                    if github_monitor.pull_latest_changes():
                        pipeline_runner.run_pipeline()
                    else:
                        logger.error("Failed to pull changes, skipping pipeline run")
                
                time.sleep(POLL_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Stopping pipeline listener...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 