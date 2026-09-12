"""
Unified Visualizer & HUD Rendering Engine
=========================================
Combines all visual annotation layers onto the video frames:
- Persistent track bounding boxes, IDs, class names, and confidence badges
- Fading trajectory motion trails & centroid points
- Virtual counting line and label
- Traffic volume & class distribution HUD card (Top-Left)
- Real-time FPS and latency performance badge (Top-Right)
- Flashing wrong-way warning bars and red alert highlight boxes
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.tracker import TrackedVehicle
from src.counter import VehicleCounter
from src.violation import WrongWayDetector
from src.metrics import PerformanceMonitor
from src.detector import CLASS_COLORS, DEFAULT_COLOR


class Visualizer:
    """
    Orchestrates all overlay annotations onto video frames.
    """

    def __init__(
        self,
        show_trajectories: bool = True,
        show_traffic_hud: bool = True,
        show_performance_hud: bool = True,
        show_counting_line: bool = True,
    ):
        self.show_trajectories = show_trajectories
        self.show_traffic_hud = show_traffic_hud
        self.show_performance_hud = show_performance_hud
        self.show_counting_line = show_counting_line

    def render(
        self,
        frame: np.ndarray,
        tracks: List[TrackedVehicle],
        counter: Optional[VehicleCounter] = None,
        violation_detector: Optional[WrongWayDetector] = None,
        monitor: Optional[PerformanceMonitor] = None,
    ) -> np.ndarray:
        """
        Renders all annotation layers onto a copy of the input frame.

        Args:
            frame: Raw BGR input frame.
            tracks: Active TrackedVehicle instances in this frame.
            counter: Optional VehicleCounter instance for counting line & tallies.
            violation_detector: Optional WrongWayDetector for violation alerts.
            monitor: Optional PerformanceMonitor for FPS and latency badge.

        Returns:
            Annotated BGR frame ready for display or VideoWriter.
        """
        annotated = frame.copy()

        # Layer 1: Virtual Counting Line
        if self.show_counting_line and counter is not None:
            line_p1, line_p2 = counter._ensure_line_coordinates(frame.shape[:2])
            line_color = (0, 255, 255)  # Yellow
            cv2.line(annotated, line_p1, line_p2, line_color, 2, cv2.LINE_AA)

            mid_x = (line_p1[0] + line_p2[0]) // 2
            mid_y = (line_p1[1] + line_p2[1]) // 2
            cv2.putText(
                annotated,
                "--- COUNTING LINE ---",
                (max(10, mid_x - 110), max(20, mid_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                line_color,
                2,
                cv2.LINE_AA,
            )

        # Layer 2: Vehicle Trajectories & Tracking Badges
        for veh in tracks:
            x1, y1, x2, y2 = veh.bbox
            color = CLASS_COLORS.get(veh.class_name.lower(), DEFAULT_COLOR)

            # Check if this vehicle is currently violating direction
            is_violating = (
                violation_detector is not None
                and veh.violation_frames >= violation_detector.confirm_frames
            )

            box_color = (0, 0, 255) if is_violating else color  # Red if violating

            # Draw trajectory trail
            if self.show_trajectories and len(veh.trajectory) > 1:
                pts = list(veh.trajectory)
                for j in range(1, len(pts)):
                    thickness = int(np.sqrt(float(j) / len(pts) * 9.0)) + 1
                    cv2.line(annotated, pts[j - 1], pts[j], box_color, thickness)

            # Draw centroid point
            cv2.circle(annotated, veh.centroid, 4, (0, 255, 255), -1)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2 if not is_violating else 3)

            # Badge: ID: 1 | Car | 0.92
            dir_str = f" [{veh.direction}]" if veh.direction else ""
            label = f"ID: {veh.track_id} | {veh.class_name.capitalize()} | {veh.confidence:.2f}{dir_str}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            badge_y1 = max(0, y1 - text_h - baseline - 6)
            badge_y2 = y1

            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y2),
                box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

            # If violating, draw floating warning tag
            if is_violating:
                alert_text = "! WRONG WAY !"
                cv2.rectangle(annotated, (x1, y2), (x1 + 110, y2 + 18), (0, 0, 220), -1)
                cv2.putText(
                    annotated,
                    alert_text,
                    (x1 + 5, y2 + 14),
                    font,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Layer 3: Emergency Screen Warning Bar (if active violation exists)
        if violation_detector is not None:
            has_violation = any(
                veh.violation_frames >= violation_detector.confirm_frames
                for veh in tracks
            )
            if has_violation:
                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                cv2.putText(
                    annotated,
                    "[!] WRONG-WAY TRAFFIC ALERT ACTIVE [!]",
                    (w // 2 - 180, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # Layer 4: Traffic Volume HUD (Top-Left)
        if self.show_traffic_hud and counter is not None:
            annotated = counter.draw_hud(annotated, active_track_count=len(tracks))

        # Layer 5: Performance HUD (Top-Right)
        if self.show_performance_hud and monitor is not None:
            annotated = monitor.draw_hud(annotated)

        return annotated
