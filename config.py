"""
config.py

Central configuration file for the AI Object Detection System.

This module stores all the constant values, file paths, and default
settings used across the project. Keeping configuration in one place
makes the project easier to maintain and extend — if a path or a
default value ever needs to change, it only needs to change here.

No classes or logic live in this file on purpose: it is meant to be a
simple, readable "settings sheet" that every other module can import.
"""

import os

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
# BASE_DIR points to the root of the project (the folder this file lives in).
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR: str = os.path.join(BASE_DIR, "models")
IMAGES_DIR: str = os.path.join(BASE_DIR, "images")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
LOGS_DIR: str = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# MobileNet-SSD Model Files
# ---------------------------------------------------------------------------
# The pre-trained MobileNet-SSD model (Caffe format) needs two files:
#   1. The .prototxt file       -> describes the network architecture
#   2. The .caffemodel file     -> contains the pre-trained weights
#
# These files are NOT trained by this project. They are downloaded once
# and placed inside the models/ folder. This is what makes this project
# "Transfer Learning" -- we reuse a model that was already trained on the
# large-scale PASCAL VOC dataset, instead of training our own from scratch.
PROTOTXT_PATH: str = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt")
MODEL_WEIGHTS_PATH: str = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

# ---------------------------------------------------------------------------
# Class Labels
# ---------------------------------------------------------------------------
# MobileNet-SSD (trained on PASCAL VOC) recognizes 20 object classes plus
# a background class at index 0. These labels are fixed by the pre-trained
# model itself, so they are defined here as a constant rather than being
# loaded from an external file.
CLASS_LABELS: list[str] = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

# ---------------------------------------------------------------------------
# Detection Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50
MIN_CONFIDENCE_THRESHOLD: float = 0.0
MAX_CONFIDENCE_THRESHOLD: float = 1.0

# MobileNet-SSD expects a fixed 300x300 input size and specific
# normalization values used during its original training.
INPUT_IMAGE_SIZE: tuple[int, int] = (300, 300)
SCALE_FACTOR: float = 0.007843  # 1 / 127.5
MEAN_SUBTRACTION: float = 127.5

# ---------------------------------------------------------------------------
# Visualization Settings
# ---------------------------------------------------------------------------
BOUNDING_BOX_COLOR: tuple[int, int, int] = (0, 255, 0)   # Green (BGR format)
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)       # White
BOX_THICKNESS: int = 2
FONT_SCALE: float = 0.5
FONT_THICKNESS: int = 1

# ---------------------------------------------------------------------------
# Supported Image Extensions
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------------------------
# Logging & History
# ---------------------------------------------------------------------------
LOG_FILE_PATH: str = os.path.join(LOGS_DIR, "detection_log.txt")
HISTORY_FILE_PATH: str = os.path.join(LOGS_DIR, "detection_history.json")
MAX_HISTORY_ENTRIES: int = 50
