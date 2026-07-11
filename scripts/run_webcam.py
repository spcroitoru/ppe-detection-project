"""
Entry point for live webcam testing.
The actual logic lives in src/inference.py; this file just calls it.

Run from the project root:
    python scripts/run_webcam.py
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import run_webcam_inference
from src.config import PROJECT_ROOT


def find_latest_model(pattern: str = "**/weights/best.pt") -> str:
    """Recursively find the most recently saved best.pt (regardless of exact subfolder)."""
    candidates = glob.glob(os.path.join(PROJECT_ROOT, "runs*", pattern), recursive=True)
    if not candidates:
        raise FileNotFoundError("No 'best.pt' model found under 'runs*' folders. Train a model first.")
    return max(candidates, key=os.path.getmtime)  # most recently modified


MODEL_PATH = find_latest_model()

if __name__ == "__main__":
    print(f"Using model: {MODEL_PATH}")
    run_webcam_inference(model_path=MODEL_PATH)