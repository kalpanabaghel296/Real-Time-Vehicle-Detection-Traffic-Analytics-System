"""
Direction Estimation Module
===========================
Estimates 2D vehicle movement direction in image coordinates.

Note on Image Coordinates:
In computer vision, the coordinate origin (0, 0) is at the TOP-LEFT of the image:
- Positive X (+dx) moves RIGHT
- Negative X (-dx) moves LEFT
- Positive Y (+dy) moves DOWN
- Negative Y (-dy) moves UP

This module provides image-plane motion estimation, not 3D world-coordinate vehicle heading.
"""

from collections import deque
import math
from typing import Tuple, Optional


def calculate_motion_vector(
    trajectory: deque,
    window: int = 8,
) -> Tuple[float, float, float]:
    """
    Computes displacement vector (dx, dy) and Euclidean distance across a recent trajectory window.

    Why a Window instead of single-frame step?
    Neural network bounding boxes inherently exhibit 1-3 pixel scale jitter frame-to-frame.
    Calculating displacement across a multi-frame window (e.g. 8 frames) smooths out
    high-frequency detector noise and provides a reliable directional vector.

    Args:
        trajectory: Deque of (x, y) centroid coordinates.
        window: Number of recent frames to analyze.

    Returns:
        Tuple of (dx, dy, euclidean_distance).
    """
    if len(trajectory) < 2:
        return (0.0, 0.0, 0.0)

    # Use the oldest point in the window and the current point
    sample_len = min(len(trajectory), window)
    start_pt = trajectory[-sample_len]
    end_pt = trajectory[-1]

    dx = float(end_pt[0] - start_pt[0])
    dy = float(end_pt[1] - start_pt[1])
    dist = math.sqrt(dx * dx + dy * dy)
    return (dx, dy, dist)


def estimate_cardinal_direction(
    dx: float,
    dy: float,
    min_distance: float = 15.0,
    frame_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Classifies 2D motion vector into a cardinal direction ('DOWN', 'UP', 'LEFT', 'RIGHT', or 'STATIONARY').

    Args:
        dx: Horizontal displacement in pixels.
        dy: Vertical displacement in pixels.
        min_distance: Minimum distance moved before assigning direction (filters stationary jitter).
        frame_size: Optional (width, height) for aspect-ratio normalized vector calculation.

    Returns:
        Direction string.
    """
    distance = math.sqrt(dx * dx + dy * dy)
    if distance < min_distance:
        return "STATIONARY"

    if frame_size is not None and frame_size[0] > 0 and frame_size[1] > 0:
        w, h = frame_size
        dx_norm = dx / w
        dy_norm = dy / h
    else:
        dx_norm = dx
        dy_norm = dy

    abs_dx_norm = abs(dx_norm)
    abs_dy_norm = abs(dy_norm)

    # In roadway perspectives (e.g. CCTV, dashboard, overhead), vehicles move along corridors.
    # Oncoming/departing lanes fan outward diagonally due to camera perspective (16:9 widescreen).
    # When significant vertical motion along the roadway corridor is present (|dy_norm| >= 0.6 * |dx_norm|),
    # the vehicle's longitudinal progression dominates over perspective fanning:
    if abs_dy_norm >= 0.6 * abs_dx_norm:
        return "DOWN" if dy > 0 else "UP"
    else:
        return "RIGHT" if dx > 0 else "LEFT"


def get_vehicle_direction(
    trajectory: deque,
    window: int = 8,
    min_distance: float = 15.0,
    frame_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Convenience function: computes displacement vector from trajectory and returns cardinal direction.
    """
    dx, dy, dist = calculate_motion_vector(trajectory, window=window)
    return estimate_cardinal_direction(
        dx, dy, min_distance=min_distance, frame_size=frame_size
    )
