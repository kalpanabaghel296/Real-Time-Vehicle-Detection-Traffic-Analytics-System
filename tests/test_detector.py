"""
Unit tests for YOLOVehicleDetector and Detection dataclass (src/detector.py).
"""

import numpy as np
import pytest
from config.config import TrafficConfig
from src.detector import Detection, YOLOVehicleDetector


def test_detection_dataclass():
    det = Detection(
        class_id=2,
        class_name="car",
        confidence=0.91234,
        bbox=(100, 150, 300, 350),
    )
    assert det.class_id == 2
    assert det.class_name == "car"
    assert det.confidence == 0.91234
    assert det.bbox == (100, 150, 300, 350)
    assert det.center == (200, 250)

    d = det.to_dict()
    assert d["class_id"] == 2
    assert d["class_name"] == "car"
    assert d["confidence"] == 0.9123
    assert d["bbox"] == [100, 150, 300, 350]


def test_detector_initialization():
    cfg = TrafficConfig(confidence_threshold=0.45)
    detector = YOLOVehicleDetector(cfg)
    assert detector.config.confidence_threshold == 0.45
    assert detector.device in ["cpu", "0", "cuda"]
    assert hasattr(detector, "model")


def test_detector_empty_frame():
    cfg = TrafficConfig()
    detector = YOLOVehicleDetector(cfg)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(empty_frame)
    # Blank black frame should yield 0 detections
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_draw_detections():
    cfg = TrafficConfig()
    detector = YOLOVehicleDetector(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample_detections = [
        Detection(class_id=2, class_name="car", confidence=0.88, bbox=(50, 50, 150, 150)),
        Detection(class_id=7, class_name="truck", confidence=0.92, bbox=(200, 100, 350, 300)),
    ]

    annotated = detector.draw_detections(frame, sample_detections)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    # Check that drawing actually modified pixels
    assert np.any(annotated != 0)
