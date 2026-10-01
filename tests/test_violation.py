"""
Unit tests for direction estimation (src/direction.py) and wrong-way violation (src/violation.py).
"""

from collections import deque
import numpy as np
import pytest
from src.direction import (
    calculate_motion_vector,
    estimate_cardinal_direction,
    get_vehicle_direction,
)
from src.tracker import TrackedVehicle
from src.violation import WrongWayDetector
from config.config import TrafficConfig


# -----------------------------------------------------------------------------
# 1. Direction Estimation Unit Tests
# -----------------------------------------------------------------------------

def test_direction_estimation_down():
    traj = deque([(200, 100), (200, 150), (200, 200)])
    direction = get_vehicle_direction(traj, min_distance=15.0)
    assert direction == "DOWN"


def test_direction_estimation_up():
    traj = deque([(200, 200), (200, 150), (200, 100)])
    direction = get_vehicle_direction(traj, min_distance=15.0)
    assert direction == "UP"


def test_direction_estimation_right():
    traj = deque([(100, 200), (150, 200), (200, 200)])
    direction = get_vehicle_direction(traj, min_distance=15.0)
    assert direction == "RIGHT"


def test_direction_estimation_left():
    traj = deque([(200, 200), (150, 200), (100, 200)])
    direction = get_vehicle_direction(traj, min_distance=15.0)
    assert direction == "LEFT"


def test_direction_estimation_stationary():
    # Only moved 2 pixels (sub-threshold noise)
    traj = deque([(200, 200), (201, 202)])
    direction = get_vehicle_direction(traj, min_distance=15.0)
    assert direction == "STATIONARY"


# -----------------------------------------------------------------------------
# 2. Wrong-Way Detector Unit Tests
# -----------------------------------------------------------------------------

def make_test_tracked_vehicle(
    track_id: int,
    y_start: int,
    y_end: int,
    history_len: int = 10,
) -> TrackedVehicle:
    """Creates a synthetic vehicle traveling vertically."""
    traj = deque(maxlen=20)
    step = (y_end - y_start) // history_len
    for i in range(history_len):
        traj.append((300, y_start + step * i))

    cx, cy = traj[-1]
    return TrackedVehicle(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=0.92,
        bbox=(cx - 20, cy - 30, cx + 20, cy + 30),
        centroid=(cx, cy),
        previous_centroid=traj[-2],
        trajectory=traj,
    )


def test_wrong_way_temporal_confirmation(tmp_path):
    # Allowed direction: DOWN. Violating direction: UP
    detector = WrongWayDetector(
        allowed_direction="DOWN",
        confirm_frames=3,
        snapshots_dir=str(tmp_path),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vehicle moving UP (from y=300 to y=100)
    veh = make_test_tracked_vehicle(track_id=1, y_start=300, y_end=100)

    # Frame 1: Violation frame count = 1 (< 3) -> No event
    events_f1 = detector.update(frame, [veh], frame_idx=1)
    assert len(events_f1) == 0
    assert veh.violation_frames == 1
    assert veh.violation_alerted is False

    # Frame 2: Violation frame count = 2 (< 3) -> No event
    events_f2 = detector.update(frame, [veh], frame_idx=2)
    assert len(events_f2) == 0
    assert veh.violation_frames == 2

    # Frame 3: Violation frame count = 3 (>= 3) -> CONFIRMED ALERT!
    events_f3 = detector.update(frame, [veh], frame_idx=3)
    assert len(events_f3) == 1
    assert veh.violation_frames == 3
    assert veh.violation_alerted is True
    assert events_f3[0]["track_id"] == 1
    assert events_f3[0]["direction"] == "UP"

    # Frame 4: Next frame, already alerted -> No duplicate event
    events_f4 = detector.update(frame, [veh], frame_idx=4)
    assert len(events_f4) == 0


def test_snapshot_creation(tmp_path):
    detector = WrongWayDetector(
        allowed_direction="DOWN",
        confirm_frames=1,
        snapshots_dir=str(tmp_path),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    veh = make_test_tracked_vehicle(track_id=42, y_start=350, y_end=150)

    events = detector.update(frame, [veh], frame_idx=10)
    assert len(events) == 1

    snap_file = tmp_path / "violation_id42_f0010.jpg"
    assert snap_file.exists()
    assert snap_file.stat().st_size > 0


def test_legal_vehicle_no_alert(tmp_path):
    detector = WrongWayDetector(
        allowed_direction="DOWN",
        confirm_frames=3,
        snapshots_dir=str(tmp_path),
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Vehicle moving legally DOWN (from y=100 to y=300)
    veh = make_test_tracked_vehicle(track_id=99, y_start=100, y_end=300)

    for i in range(5):
        events = detector.update(frame, [veh], frame_idx=i)
        assert len(events) == 0

    assert len(detector.violations) == 0


def test_direction_estimation_aspect_ratio_diagonal_perspective():
    """Test that diagonal lane movement in widescreen perspective is classified as DOWN."""
    # Simulates vehicle moving from (1400, 400) to (1590, 474) on 1920x1080: dx=190, dy=74
    traj = deque([
        (1400, 400),
        (1450, 420),
        (1500, 440),
        (1550, 460),
        (1590, 474),
    ])
    direction = get_vehicle_direction(traj, min_distance=15.0, frame_size=(1920, 1080))
    assert direction == "DOWN"


def test_wrong_way_auto_calibration(tmp_path):
    """Test that AUTO mode calibrates baseline direction to DOWN from dominant traffic flow."""
    detector = WrongWayDetector(
        allowed_direction="AUTO",
        confirm_frames=3,
        snapshots_dir=str(tmp_path),
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Simulate 5 vehicles over 6 frames moving downwards
    vehicles = [
        make_test_tracked_vehicle(track_id=i, y_start=200, y_end=400, history_len=8)
        for i in range(1, 6)
    ]

    for f_idx in range(6):
        detector.update(frame, vehicles, frame_idx=f_idx)

    # Detector should have calibrated to DOWN
    assert detector.is_calibrated is True
    assert detector.inferred_allowed_direction == "DOWN"
    # No false violations during calibration
    assert len(detector.violations) == 0


def test_wrong_way_downstream_no_violation(tmp_path):
    """Test that vehicles with positive dy never trigger a violation against DOWN flow."""
    detector = WrongWayDetector(
        allowed_direction="DOWN",
        confirm_frames=2,
        snapshots_dir=str(tmp_path),
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Vehicle with diagonal downstream motion (dx=100, dy=40)
    traj = deque(maxlen=20)
    for i in range(10):
        traj.append((500 + i * 10, 200 + i * 4))
    veh = TrackedVehicle(
        track_id=10,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=(580, 210, 620, 270),
        centroid=traj[-1],
        previous_centroid=traj[-2],
        trajectory=traj,
    )

    for i in range(5):
        events = detector.update(frame, [veh], frame_idx=i)
        assert len(events) == 0

    assert len(detector.violations) == 0

