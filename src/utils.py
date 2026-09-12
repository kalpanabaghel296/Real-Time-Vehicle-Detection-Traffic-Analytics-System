"""
Core Geometric & Vector Utilities
=================================
Provides foundational mathematical operations for:
- Centroid extraction from bounding boxes
- Vector displacement and Euclidean distance
- Line segment intersection (computational geometry orientation test)
- Intersection over Union (IoU) calculation
"""

import math
from typing import Tuple, Optional


def calculate_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Computes the geometric center (x, y) of a bounding box [x1, y1, x2, y2].

    Why Centroid?
    Tracking an entire bounding box across frames is sensitive to scale changes
    (e.g., vehicle coming closer to camera expands the box). The centroid provides
    a single robust coordinate representing object motion.

    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)

    Returns:
        Tuple of integer (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2.0)
    cy = int((y1 + y2) / 2.0)
    return (cx, cy)


def calculate_euclidean_distance(
    pt1: Tuple[float, float], pt2: Tuple[float, float]
) -> float:
    """
    Calculates straight-line 2D Euclidean distance between two points:
    distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    """
    return math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


def calculate_displacement_vector(
    pt1: Tuple[float, float], pt2: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Calculates 2D displacement vector from pt1 (past) to pt2 (current):
    dx = x2 - x1
    dy = y2 - y1

    Note: In computer vision image coordinates:
    - (0, 0) is at TOP-LEFT
    - +x moves RIGHT
    - +y moves DOWN
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return (dx, dy)


def calculate_iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float],
) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes:
    IoU = Area of Overlap / Area of Union

    Both boxes must be formatted as: [x1, y1, x2, y2]

    Returns:
        Float value in range [0.0, 1.0]
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    intersection_width = max(0.0, xB - xA)
    intersection_height = max(0.0, yB - yA)
    intersection_area = intersection_width * intersection_height

    # Compute areas of individual bounding boxes
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    # Compute union area
    union_area = areaA + areaB - intersection_area

    if union_area == 0.0:
        return 0.0

    return float(intersection_area / union_area)


def _orientation(
    p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]
) -> int:
    """
    Finds the orientation of ordered triplet (p, q, r).

    Uses the 2D cross product of vectors (q - p) and (r - q):
    cross_product = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    Returns:
        0 -> p, q and r are collinear
        1 -> Clockwise turn
        2 -> Counterclockwise turn
    """
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (
        float(q[0] - p[0]) * (r[1] - q[1])
    )
    if abs(val) < 1e-9:
        return 0  # Collinear
    return 1 if val > 0 else 2  # Clockwise or Counterclockwise


def _on_segment(
    p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]
) -> bool:
    """
    Given three collinear points p, q, r, checks if point q lies on line segment 'pr'.
    """
    return (
        q[0] <= max(p[0], r[0])
        and q[0] >= min(p[0], r[0])
        and q[1] <= max(p[1], r[1])
        and q[1] >= min(p[1], r[1])
    )


def do_segments_intersect(
    p1: Tuple[float, float],
    q1: Tuple[float, float],
    p2: Tuple[float, float],
    q2: Tuple[float, float],
) -> bool:
    """
    Determines if line segment 'p1-q1' intersects line segment 'p2-q2'.

    Why Segment Intersection instead of simple y-coordinate checks?
    1. If a vehicle moves fast (low frame rate or high speed), the centroid jumps
       from frame t (e.g. y=280) to frame t+1 (e.g. y=330), skipping y=300 entirely.
    2. Real-world counting lines are often slanted/diagonal across road perspectives.
    Checking intersection between segment (prev_centroid -> curr_centroid) and
    (line_start -> line_end) is mathematically robust and guarantees zero missed crossings.

    Args:
        p1, q1: Endpoints of the first line segment (e.g., vehicle trajectory step)
        p2, q2: Endpoints of the second line segment (e.g., virtual counting line)

    Returns:
        True if the two line segments intersect, False otherwise.
    """
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    # General case: segments intersect if orientations differ
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases: collinear points lying on segment
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True

    return False
