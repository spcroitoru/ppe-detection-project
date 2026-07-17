"""
Run PPE detection on a single static image (not webcam).
Used for the Docker image, where no camera is available - a fixed input
image is processed instead, to verify the packaged model works correctly.

Usage:
    python src/predict_image.py <input_image_path> <output_image_path>
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.inference import draw_detections
from src.config import PROJECT_ROOT

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")


def predict_image(input_path: str, output_path: str, conf: float = 0.25):
    """Run detection on a single image and save the annotated result."""
    from ultralytics import YOLO

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    model = YOLO(MODEL_PATH)
    results = model.predict(source=input_path, conf=conf, imgsz=640, verbose=False)

    result = results[0]
    frame = result.orig_img.copy()
    frame = draw_detections(frame, result, model.names)

    cv2.imwrite(output_path, frame)
    print(f"Detections: {len(result.boxes)}")
    print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/predict_image.py <input_image_path> <output_image_path>")
        sys.exit(1)

    predict_image(sys.argv[1], sys.argv[2])