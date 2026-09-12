"""
Unit tests for geometric and vector utilities (src/utils.py).
"""

import math
import pytest
from src.utils import (
    calculate_centroid,
    calculate_euclidean_distance,
    calculate_displacement_vector,
    calculate_iou,
    do_segments_intersect,
)


def test_calculate_centroid():
    # Box [x1, y1, x2, y2]
    bbox = (100, 200, 300, 400)
    cx, cy = calculate_centroid(bbox)
    assert cx == 200
    assert cy == 300


def test_calculate_euclidean_distance():
    pt1 = (0.0, 0.0)
    pt2 = (3.0, 4.0)
    dist = calculate_euclidean_distance(pt1, pt2)
    assert pytest.approx(dist, 1e-6) == 5.0


def test_calculate_displacement_vector():
    pt1 = (100.0, 150.0)
    pt2 = (120.0, 200.0)
    dx, dy = calculate_displacement_vector(pt1, pt2)
    assert dx == 20.0
    assert dy == 50.0


def test_calculate_iou_identical_boxes():
    box = (10.0, 10.0, 50.0, 50.0)
    iou = calculate_iou(box, box)
    assert pytest.approx(iou, 1e-6) == 1.0


def test_calculate_iou_disjoint_boxes():
    boxA = (0.0, 0.0, 10.0, 10.0)
    boxB = (20.0, 20.0, 30.0, 30.0)
    iou = calculate_iou(boxA, boxB)
    assert iou == 0.0


def test_calculate_iou_partial_overlap():
    boxA = (0.0, 0.0, 20.0, 20.0)  # Area = 400
    boxB = (10.0, 0.0, 30.0, 20.0)  # Area = 400
    # Overlap rectangle: x in [10, 20], y in [0, 20] -> Area = 200
    # Union area = 400 + 400 - 200 = 600
    # Expected IoU = 200 / 600 = 1/3 ~ 0.333333
    iou = calculate_iou(boxA, boxB)
    assert pytest.approx(iou, 1e-5) == 1.0 / 3.0


def test_segment_intersection_crossing():
    # Counting line horizontal at y = 300, from x=0 to x=640
    line_p1 = (0.0, 300.0)
    line_q1 = (640.0, 300.0)

    # Vehicle centroid jump: frame t (x=320, y=280) to frame t+1 (x=320, y=320)
    veh_p2 = (320.0, 280.0)
    veh_q2 = (320.0, 320.0)

    assert do_segments_intersect(line_p1, line_q1, veh_p2, veh_q2) is True


def test_segment_intersection_parallel_non_crossing():
    # Horizontal counting line
    line_p1 = (0.0, 300.0)
    line_q1 = (640.0, 300.0)

    # Vehicle moving horizontally above line without crossing
    veh_p2 = (100.0, 250.0)
    veh_q2 = (200.0, 250.0)

    assert do_segments_intersect(line_p1, line_q1, veh_p2, veh_q2) is False


def test_segment_intersection_moving_away():
    # Horizontal counting line at y = 300
    line_p1 = (0.0, 300.0)
    line_q1 = (640.0, 300.0)

    # Vehicle moving from y=350 to y=400 (moving further away)
    veh_p2 = (320.0, 350.0)
    veh_q2 = (320.0, 400.0)

    assert do_segments_intersect(line_p1, line_q1, veh_p2, veh_q2) is False
