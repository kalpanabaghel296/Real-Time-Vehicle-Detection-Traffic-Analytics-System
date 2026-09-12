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
) -> str:
    """
    Classifies 2D motion vector into a cardinal direction ('DOWN', 'UP', 'LEFT', 'RIGHT', or 'STATIONARY').

    Args:
        dx: Horizontal displacement in pixels.
        dy: Vertical displacement in pixels.
        min_distance: Minimum distance moved before assigning direction (filters stationary jitter).

    Returns:
        Direction string.
    """
    distance = math.sqrt(dx * dx + dy * dy)
    if distance < min_distance:
        return "STATIONARY"

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Determine dominant axis of motion
    if abs_dy >= abs_dx:
        return "DOWN" if dy > 0 else "UP"
    else:
        return "RIGHT" if dx > 0 else "LEFT"


def get_vehicle_direction(
    trajectory: deque,
    window: int = 8,
    min_distance: float = 15.0,
) -> str:
    """
    Convenience function: computes displacement vector from trajectory and returns cardinal direction.
    """
    dx, dy, dist = calculate_motion_vector(trajectory, window=window)
    return estimate_cardinal_direction(dx, dy, min_distance=min_distance)
