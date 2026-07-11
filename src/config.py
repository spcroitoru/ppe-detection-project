"""
Centralized configuration for the PPE Detection project.
All paths, constants, and shared parameters live here, so they aren't
duplicated/inconsistent across train.py, inference.py, etc.
"""
import os
import yaml

# --- Paths (relative to project root; scripts are expected to run from anywhere,
# the root is computed automatically below) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data_raw")
YAML_PATH = os.path.join(PROJECT_ROOT, "data.yaml")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

SPLITS = ["train", "val", "test"]

# --- Classes (loaded from data.yaml) ---
with open(YAML_PATH) as f:
    _data_config = yaml.safe_load(f)
CLASS_NAMES = _data_config["names"]

# --- Classes that signal missing/violated PPE -> drawn in RED ---
RED_CLASSES = {"NO_helmet", "NO_Vest", "NO_goggles", "No_SafetyShoes", "NO_Gloves", "Slippers"}

# --- Drawing colors (BGR, used by OpenCV) ---
COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 255)

# --- MLflow ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "ppe-detection"

# --- Default training parameters (can be overridden per call) ---
DEFAULT_TRAIN_PARAMS = {
    "imgsz": 416,
    "batch": 8,
    "device": 0,
}