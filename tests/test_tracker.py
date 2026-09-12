"""
Unit tests for VehicleTracker and TrackedVehicle (src/tracker.py).
"""

from collections import deque
import numpy as np
import pytest
from config.config import TrafficConfig
from src.tracker import TrackedVehicle, VehicleTracker


def test_tracked_vehicle_dataclass():
    traj = deque(maxlen=10)
    traj.append((100, 200))
    traj.append((110, 220))

    veh = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox=(80, 160, 140, 280),
        centroid=(110, 220),
        previous_centroid=(100, 200),
        trajectory=traj,
        first_frame=0,
        last_frame=1,
    )

    assert veh.track_id == 1
    assert veh.class_name == "car"
    assert veh.displacement == (10.0, 20.0)

    d = veh.to_dict()
    assert d["track_id"] == 1
    assert d["confidence"] == 0.95
    assert d["centroid"] == [110, 220]
    assert d["previous_centroid"] == [100, 200]


def test_tracker_initialization():
    cfg = TrafficConfig(track_buffer=40)
    tracker = VehicleTracker(cfg)
    assert tracker.config.track_buffer == 40
    assert len(tracker.tracks) == 0
    assert hasattr(tracker, "model")


def test_tracker_empty_frame():
    cfg = TrafficConfig()
    tracker = VehicleTracker(cfg)
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.update(empty_frame, frame_idx=0)
    assert isinstance(tracks, list)
    assert len(tracks) == 0


def test_draw_tracks():
    cfg = TrafficConfig()
    tracker = VehicleTracker(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    sample_track = TrackedVehicle(
        track_id=5,
        class_id=2,
        class_name="car",
        confidence=0.89,
        bbox=(100, 100, 250, 250),
        centroid=(175, 175),
        previous_centroid=(170, 160),
        trajectory=deque([(170, 160), (175, 175)], maxlen=10),
    )

    annotated = tracker.draw_tracks(frame, [sample_track], draw_trajectory=True)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    assert np.any(annotated != 0)
