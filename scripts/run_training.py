"""
Entry point for training - Step 7 experiment: larger architecture (yolov8s)
combined with the best-performing configuration found so far
(imgsz=640, SGD optimizer), to see if more model capacity helps further.

Run from the project root:
    python scripts/run_training.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_model

if __name__ == "__main__":
    train_model(
        model_name="yolov8s.pt",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        optimizer="SGD",
        project="runs_test",
        name="exp_yolov8s_imgsz640_sgd",
    )