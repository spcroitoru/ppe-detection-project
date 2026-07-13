"""
Entry point for training - full pipeline test with early stopping and
learning rate reduction (Step 6).
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
        epochs=50,          # more epochs, to see the effect of early stopping / LR reduction
        imgsz=640,
        batch=8,
        device=0,
        patience=10,
        optimizer="SGD",
        project="runs_test",
        name="exp_imgsz640_sgd",
    )