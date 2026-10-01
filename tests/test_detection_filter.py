"""
Unit tests for DetectionFilter
==============================
Verifies:
- Per-class confidence gating
- Physical size & aspect ratio priors
- Temporal multi-frame consistency confirmation
"""

import pytest
from dataclasses import dataclass
from typing import Tuple

from src.filters.detection_filter import DetectionFilter, DetectionFilterConfig


@dataclass
class DummyDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]


def test_confidence_filtering():
    cfg = DetectionFilterConfig(
        per_class_confidence={"car": 0.45, "truck": 0.35, "bus": 0.30, "motorcycle": 0.25}
    )
    filter_obj = DetectionFilter(cfg)

    dets = [
        DummyDetection(2, "car", 0.40, (100, 100, 200, 200)),   # Under 0.45 -> rejected
        DummyDetection(2, "car", 0.50, (100, 100, 200, 200)),   # Above 0.45 -> kept
        DummyDetection(7, "truck", 0.36, (100, 100, 200, 200)), # Above 0.35 -> kept
        DummyDetection(3, "motorcycle", 0.26, (100, 100, 200, 200)), # Above 0.25 -> kept
    ]

    res = filter_obj.filter_by_confidence(dets)
    assert len(res) == 3
    assert res[0].confidence == 0.50
    assert res[1].class_name == "truck"
    assert res[2].class_name == "motorcycle"


def test_size_priors_filtering():
    cfg = DetectionFilterConfig(enable_size_priors=True)
    filter_obj = DetectionFilter(cfg)

    dets = [
        # Normal car: area 100x80 = 8000, ratio 0.8 -> kept
        DummyDetection(2, "car", 0.9, (100, 100, 200, 180)),
        # Impossible car: area 5x5 = 25 (< 300 min_area) -> rejected
        DummyDetection(2, "car", 0.9, (100, 100, 105, 105)),
        # Impossible car aspect ratio: width=200, height=10 -> ratio 0.05 (< 0.25) -> rejected
        DummyDetection(2, "car", 0.9, (100, 100, 300, 110)),
        # Normal motorcycle: 30x50, area 1500, ratio 1.66 -> kept
        DummyDetection(3, "motorcycle", 0.9, (100, 100, 130, 150)),
    ]

    res = filter_obj.filter_by_size_priors(dets, frame_shape=(720, 1280))
    assert len(res) == 2
    assert res[0].class_name == "car"
    assert res[1].class_name == "motorcycle"


def test_temporal_consistency_flicker_rejection():
    cfg = DetectionFilterConfig(
        enable_temporal_consistency=True,
        temporal_window_frames=3,
        temporal_min_hits=2,
        temporal_iou_match=0.5,
    )
    filter_obj = DetectionFilter(cfg)

    # Frame 0: Persistent car A and Flicker car B
    car_a = DummyDetection(2, "car", 0.9, (100, 100, 200, 200))
    flicker_b = DummyDetection(2, "car", 0.9, (500, 500, 600, 600))
    res0 = filter_obj.filter_by_temporal_consistency([car_a, flicker_b], frame_idx=0)
    # Bootstrap frame allows initial candidates
    assert len(res0) == 2

    # Frame 1: Persistent car A slightly moved, Flicker B disappears, new single-frame glitch C appears
    car_a_moved = DummyDetection(2, "car", 0.9, (102, 103, 202, 203))
    glitch_c = DummyDetection(2, "car", 0.9, (700, 200, 800, 300))
    res1 = filter_obj.filter_by_temporal_consistency([car_a_moved, glitch_c], frame_idx=1)

    # Car A has hit from frame 0 and frame 1 (2 hits >= 2) -> confirmed
    # Glitch C only has 1 hit -> rejected
    assert len(res1) == 1
    assert res1[0].bbox == (102, 103, 202, 203)
