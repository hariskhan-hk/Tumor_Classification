import os
import requests
import shutil
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

def ensure_model_dir(dir_path='./model'):
    """Create model directory if it doesn't exist."""
    if not os.path.exists(dir_path):
        print(f"Creating directory: {dir_path}")
        os.makedirs(dir_path)

def download_from_gdrive(gdrive_url, output_path):
    """Download a file from Google Drive."""
    print(f"Downloading model from Google Drive: {gdrive_url}")
    try:
        gdown.download(gdrive_url, output_path, quiet=False)
        if os.path.exists(output_path):
            print(f"Successfully downloaded model to {output_path}")
            return True
        else:
            print("Download failed")
            return False
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def get_model(gdrive_url=None, local_path='./model/model.keras'):
    """Get the model - either download it or use an existing one."""
    # Check if model already exists locally
    if os.path.exists(local_path):
        try:
            model = load_model(local_path)
            print(f"Found and loaded existing model from {local_path}")
            return model
        except Exception as e:
            print(f"Found model file but couldn't load it: {e}")
            print("Will try to download a fresh copy")
            # If model exists but is corrupt, remove it
            os.remove(local_path)
    
    # If we need to download
    if gdrive_url:
        ensure_model_dir(os.path.dirname(local_path))
        success = download_from_gdrive(gdrive_url, local_path)
        if success:
            try:
                model = load_model(local_path)
                print("Successfully loaded downloaded model")
                return model
            except Exception as e:
                print(f"Downloaded model but failed to load it: {e}")
                return None
    
    print("No model available - you need to train the model first or provide a valid download URL")
    return None

if __name__ == "__main__":
    # Replace this with your actual Google Drive URL for the model
    # For example: https://drive.google.com/file/d/1AbCdEfG0123456789/view?usp=sharing
    GDRIVE_URL = "https://drive.google.com/file/d/12VDSpKWK7em7O3-HTx6qWxw5awQUHlIs/view"
    
    # Install gdown if not already installed
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call(["pip", "install", "gdown"])
        import gdown
    
    model = get_model(GDRIVE_URL)
    
    if model:
        print("Model loaded successfully!")
        model.summary()