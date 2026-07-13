"""
Training logic for the PPE detection YOLO model.
Reusable: parameters are passed as arguments, not hardcoded, so this can be
called both for smoke tests and for full training runs.
"""
import os

import torch
from ultralytics import YOLO

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    YAML_PATH,
    DEFAULT_TRAIN_PARAMS,
)

# enable MLflow integration (Ultralytics auto-detects these env variables)
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "ppe-detection"

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("runs/detect/runs_test/pipeline_test/weights/last.pt")
    model.train(resume=True)


def train_model(
    model_name: str = "yolov8n.pt",
    epochs: int = 3,
    imgsz: int = DEFAULT_TRAIN_PARAMS["imgsz"],
    batch: int = DEFAULT_TRAIN_PARAMS["batch"],
    device=DEFAULT_TRAIN_PARAMS["device"],
    patience: int = 10,
    optimizer: str = "auto",   # "auto", "SGD", "AdamW", etc.
    lr0: float = 0.01,
    lrf: float = 0.01,
    project: str = "runs_test",
    name: str = "smoke_test",
    exist_ok: bool = True,
):
    """
    Train a YOLO model with the given parameters.
    Returns the Ultralytics 'results' object (contains metrics, model paths).
    """
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (running on CPU)")

    model = YOLO(model_name)

    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        project=project,
        name=name,
        exist_ok=exist_ok,
    )

    print(f"\nTraining finished. Results saved to: {project}/{name}/")
    return results


if __name__ == "__main__":
    # direct run = default smoke test, equivalent to the original smoke_test_training.py
    train_model()