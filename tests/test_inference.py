"""
Unit tests for src/inference.py (postprocessing: drawing detections on frames).
"""
import numpy as np

from src.inference import draw_detections
from src.config import COLOR_GREEN, COLOR_RED


class FakeBox:
    """Mimics an Ultralytics detection box object for testing purposes."""
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class FakeResult:
    """Mimics an Ultralytics results object, holding a list of boxes."""
    def __init__(self, boxes):
        self.boxes = boxes


def test_draw_detections_uses_green_for_compliant_class():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 50, 50])
    result = FakeResult([box])
    model_names = {0: "Helmet"}

    output = draw_detections(frame, result, model_names)

    assert output.shape == frame.shape


def test_draw_detections_uses_red_for_violation_class():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FakeBox(cls_id=4, conf=0.9, xyxy=[10, 10, 50, 50])
    result = FakeResult([box])
    model_names = {4: "NO_helmet"}

    output = draw_detections(frame, result, model_names)

    assert output.shape == frame.shape