"""
Unit tests for VehicleCounter (src/counter.py).
"""

from collections import deque
import numpy as np
import pytest
from config.config import TrafficConfig
from src.tracker import TrackedVehicle
from src.counter import VehicleCounter


def make_test_vehicle(
    track_id: int,
    class_name: str,
    prev_centroid: tuple,
    curr_centroid: tuple,
) -> TrackedVehicle:
    """Helper to instantiate a TrackedVehicle for testing."""
    traj = deque([prev_centroid, curr_centroid], maxlen=10)
    cx, cy = curr_centroid
    return TrackedVehicle(
        track_id=track_id,
        class_id=2 if class_name == "car" else 7,
        class_name=class_name,
        confidence=0.90,
        bbox=(cx - 20, cy - 30, cx + 20, cy + 30),
        centroid=curr_centroid,
        previous_centroid=prev_centroid,
        trajectory=traj,
    )


def test_line_crossing_counted():
    # Counting line horizontal at y = 300, across 0..640
    line = ((0, 300), (640, 300))
    counter = VehicleCounter(counting_line=line, counting_direction="ANY")

    # Vehicle moves from y=280 to y=320 (crosses y=300)
    veh = make_test_vehicle(track_id=1, class_name="car", prev_centroid=(300, 280), curr_centroid=(300, 320))
    events = counter.update([veh], frame_shape=(480, 640))

    assert len(events) == 1
    assert counter.total_count == 1
    assert counter.counts_by_class["car"] == 1
    assert 1 in counter.counted_ids


def test_duplicate_counting_prevention():
    # Counting line at y = 300
    line = ((0, 300), (640, 300))
    counter = VehicleCounter(counting_line=line, counting_direction="ANY")

    # Frame 1: Vehicle 1 crosses line
    veh_f1 = make_test_vehicle(track_id=1, class_name="car", prev_centroid=(300, 280), curr_centroid=(300, 320))
    counter.update([veh_f1], frame_shape=(480, 640), frame_idx=1)
    assert counter.total_count == 1

    # Frame 2: Same vehicle (track_id=1) continues moving across/near line
    veh_f2 = make_test_vehicle(track_id=1, class_name="car", prev_centroid=(300, 320), curr_centroid=(300, 360))
    events_f2 = counter.update([veh_f2], frame_shape=(480, 640), frame_idx=2)

    # Must NOT count again
    assert len(events_f2) == 0
    assert counter.total_count == 1


def test_non_crossing_vehicle_not_counted():
    line = ((0, 300), (640, 300))
    counter = VehicleCounter(counting_line=line, counting_direction="ANY")

    # Vehicle moves from y=100 to y=150 (nowhere near y=300)
    veh = make_test_vehicle(track_id=2, class_name="truck", prev_centroid=(200, 100), curr_centroid=(200, 150))
    events = counter.update([veh], frame_shape=(480, 640))

    assert len(events) == 0
    assert counter.total_count == 0
    assert counter.counts_by_class["truck"] == 0


def test_class_wise_counts():
    line = ((0, 300), (640, 300))
    counter = VehicleCounter(counting_line=line, counting_direction="ANY")

    veh_car = make_test_vehicle(track_id=10, class_name="car", prev_centroid=(150, 280), curr_centroid=(150, 320))
    veh_truck = make_test_vehicle(track_id=11, class_name="truck", prev_centroid=(450, 280), curr_centroid=(450, 320))

    counter.update([veh_car, veh_truck], frame_shape=(480, 640))

    assert counter.total_count == 2
    assert counter.counts_by_class["car"] == 1
    assert counter.counts_by_class["truck"] == 1


def test_direction_filter_counting():
    line = ((0, 300), (640, 300))
    # Configured to count ONLY vehicles moving DOWN (dy > 0)
    counter = VehicleCounter(counting_line=line, counting_direction="DOWN")

    # Vehicle moving UP from y=320 to y=280 (dy = -40 < 0)
    veh_up = make_test_vehicle(track_id=20, class_name="car", prev_centroid=(300, 320), curr_centroid=(300, 280))
    events_up = counter.update([veh_up], frame_shape=(480, 640))

    # Should be ignored because direction is UP
    assert len(events_up) == 0
    assert counter.total_count == 0

    # Vehicle moving DOWN from y=280 to y=320 (dy = +40 > 0)
    veh_down = make_test_vehicle(track_id=21, class_name="car", prev_centroid=(300, 280), curr_centroid=(300, 320))
    events_down = counter.update([veh_down], frame_shape=(480, 640))

    assert len(events_down) == 1
    assert counter.total_count == 1


def test_vertical_line_orientation():
    cfg = TrafficConfig(line_orientation="VERTICAL")
    counter = VehicleCounter(cfg)
    # Default line should be vertical across x = 320 for 640x480 frame
    p1, p2 = counter._ensure_line_coordinates((480, 640))
    assert p1 == (320, 0)
    assert p2 == (320, 480)

    # Vehicle moving horizontally from x=300 to x=350 crosses vertical line at x=320
    veh = make_test_vehicle(track_id=50, class_name="car", prev_centroid=(300, 200), curr_centroid=(350, 200))
    events = counter.update([veh], frame_shape=(480, 640))
    assert len(events) == 1
    assert counter.total_count == 1

