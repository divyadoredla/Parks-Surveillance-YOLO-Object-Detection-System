"""
Configuration file for Enhanced Surveillance Web App
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model Configuration
MODEL_PATH = str(BASE_DIR / "runs" / "train" / "parks_surveillance_training3" / "weights" / "best.pt")
CONFIDENCE_THRESHOLD = 0.25

# Activity Classification Rules
UNAUTHORIZED_ACTIVITIES = {
    'vandalism': ['Drawing', 'Chalkpiece'],  # Graffiti/vandalism indicators
    'suspicious_items': ['Bag'],  # Unattended items (when no person nearby)
    'vehicle_violations': ['car', 'Bicycle']  # Vehicles in restricted areas
}

AUTHORIZED_ACTIVITIES = ['person', 'Bench', 'Tree', 'Building', 'road', 'Cap']

# Detection Classes (from custom model)
ALL_CLASSES = [
    'Bag', 'Bench', 'Bicycle', 'Building', 'Cap',
    'Chalkpiece', 'Drawing', 'Tree', 'car', 'person', 'road'
]

# Alert Severity Levels
SEVERITY_LEVELS = {
    'HIGH': ['Drawing', 'Chalkpiece'],  # Vandalism
    'MEDIUM': ['Bag'],  # Unattended items
    'LOW': ['car', 'Bicycle']  # Vehicle violations
}

# Proximity threshold for unattended item detection (in pixels)
PROXIMITY_THRESHOLD = 100

# File Upload Settings
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "detection_results"
UNAUTHORIZED_DIR = BASE_DIR / "unauthorized_activities"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
UNAUTHORIZED_DIR.mkdir(exist_ok=True)

# Supported file formats
SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov']

# Authentication Settings
AUTH_COOKIE_NAME = "surveillance_auth"
AUTH_COOKIE_KEY = "surveillance_secret_key_2024"
AUTH_COOKIE_EXPIRY_DAYS = 30

# App Settings
APP_TITLE = "Parks Surveillance - Activity Monitoring System"
APP_ICON = "🎥"
