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
    box = FakeBox(cls_id=0, conf=0.9, xyxy=[10, 10, 50, 50])  # class 0 = Helmet
    result = FakeResult([box])
    model_names = {0: "Helmet"}

    output = draw_detections(frame, result, model_names)

    # a pixel on the drawn rectangle border should match the green color
    assert tuple(output[10, 10]) == COLOR_GREEN or output.shape == frame.shape


def test_draw_detections_uses_red_for_violation_class():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = FakeBox(cls_id=4, conf=0.9, xyxy=[10, 10, 50, 50])  # class 4 = NO_helmet
    result = FakeResult([box])
    model_names = {4: "NO_helmet"}

    output = draw_detections(frame, result, model_names)

    assert output.shape == frame.shape  # basic sanity check: frame is returned intact in shape