"""
Live inference (webcam) logic for PPE detection.
Boxes are drawn manually, colored by class: GREEN = PPE present/correct,
RED = PPE missing/violation (per RED_CLASSES in config.py).
"""
import cv2
from ultralytics import YOLO

from src.config import RED_CLASSES, COLOR_GREEN, COLOR_RED


def draw_detections(frame, result, model_names):
    """
    Draw the detected boxes on a frame, colored by class.
    Modifies and returns the frame.
    """
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model_names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLOR_RED if cls_name in RED_CLASSES else COLOR_GREEN

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def run_webcam_inference(model_path: str, conf: float = 0.25, imgsz: int = 416, source=0):
    """
    Run live inference on the webcam, with color-coded detections.
    Press 'q' in the video window to quit.
    """
    model = YOLO(model_path)

    results_generator = model.predict(
        source=source,
        stream=True,
        conf=conf,
        imgsz=imgsz,
        show=False,
    )

    for result in results_generator:
        frame = result.orig_img.copy()
        frame = draw_detections(frame, result, model.names)

        cv2.imshow("PPE Detection - press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()