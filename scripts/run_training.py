"""
Entry point for training - smoke test with default parameters.
The actual logic lives in src/train.py; this file just calls it.

Run from the project root:
    python scripts/run_training.py
"""
import sys
import os

# make sure the project root is on the path, so the 'src' package can be found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model

if __name__ == "__main__":
    train_model(
        model_name="yolov8n.pt",
        epochs=3,
        imgsz=416,
        batch=8,
        device=0,
        project="runs_test",
        name="smoke_test_refactored",  # different name, so we don't overwrite the original run
    )