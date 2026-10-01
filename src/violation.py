"""
Wrong-Way Vehicle Violation Detection Module
============================================
Detects vehicles driving counter to legal roadway flow.
Features:
- Configurable allowed flow direction
- Multi-frame temporal confirmation window (prevents single-frame detector false alerts)
- Automated violation snapshot saving with red alert bounding boxes and banners
- Structured event logging
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.tracker import TrackedVehicle
from src.direction import (
    calculate_motion_vector,
    estimate_cardinal_direction,
    get_vehicle_direction,
)


class WrongWayDetector:
    """
    Monitors tracked vehicle trajectories against legal traffic flow rules.
    """

    def __init__(
        self,
        config: Optional[TrafficConfig] = None,
        allowed_direction: Optional[str] = None,
        confirm_frames: Optional[int] = None,
        snapshots_dir: Optional[str] = None,
    ):
        """
        Args:
            config: TrafficConfig instance.
            allowed_direction: Legal traffic heading ("DOWN", "UP", "LEFT", "RIGHT").
            confirm_frames: Consecutive frames of violation required before firing alert.
            snapshots_dir: Output directory where alert snapshot images are written.
        """
        self.config = config or TrafficConfig()
        self.allowed_direction = (allowed_direction or self.config.allowed_direction).upper()
        self.confirm_frames = confirm_frames or self.config.violation_confirm_frames
        self.min_track_history = getattr(self.config, "min_track_history_for_violation", 5)

        snap_path_str = snapshots_dir or self.config.snapshots_dir
        self.snapshots_dir = Path(snap_path_str)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Audit registry of confirmed violations
        self.violations: List[Dict[str, Any]] = []

        # Vector calibration samples for AUTO baseline flow inference
        self.calibration_samples_dx: List[float] = []
        self.calibration_samples_dy: List[float] = []
        self.min_calibration_samples: int = 25
        self.inferred_allowed_direction: Optional[str] = (
            self.allowed_direction if self.allowed_direction != "AUTO" else None
        )
        self.is_calibrated: bool = self.allowed_direction != "AUTO"

        # Direction counts maintained for backward compatibility
        self.direction_counts: Dict[str, int] = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0}

    def _is_opposite_direction(
        self,
        current_dir: str,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> bool:
        """
        Checks if estimated direction directly opposes configured or auto-inferred legal traffic flow.
        Also validates physical counter-flow displacement along the travel corridor.
        """
        effective_allowed = self.inferred_allowed_direction or self.allowed_direction
        if not effective_allowed or effective_allowed == "AUTO":
            return False  # Calibrating baseline flow

        opposites = {
            "DOWN": "UP",
            "UP": "DOWN",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        target_opposite = opposites.get(effective_allowed.upper())
        if not target_opposite or current_dir != target_opposite:
            return False

        # Verify physical counter-flow displacement along the travel corridor
        if effective_allowed == "DOWN" and dy >= 0:
            return False
        elif effective_allowed == "UP" and dy <= 0:
            return False
        elif effective_allowed == "RIGHT" and dx >= 0:
            return False
        elif effective_allowed == "LEFT" and dx <= 0:
            return False

        return True

    def update(
        self,
        frame: np.ndarray,
        tracks: List[TrackedVehicle],
        frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active tracks for wrong-way movement in the current frame.

        Args:
            frame: Current raw BGR image.
            tracks: Active TrackedVehicle instances from the tracker.
            frame_idx: Current frame index.

        Returns:
            List of newly confirmed violation event dictionaries.
        """
        new_violations: List[Dict[str, Any]] = []
        h, w = frame.shape[:2]
        frame_size = (w, h)

        # In AUTO mode, calibrate baseline traffic flow from stable moving vectors
        if self.allowed_direction == "AUTO" and not self.is_calibrated:
            for veh in tracks:
                if len(veh.trajectory) >= self.min_track_history:
                    dx, dy, dist = calculate_motion_vector(
                        veh.trajectory, window=self.config.trajectory_history_length
                    )
                    if dist >= self.config.min_movement_distance:
                        self.calibration_samples_dx.append(dx / w)
                        self.calibration_samples_dy.append(dy / h)

            if len(self.calibration_samples_dy) >= self.min_calibration_samples:
                total_abs_v = sum(abs(y) for y in self.calibration_samples_dy)
                total_abs_h = sum(abs(x) for x in self.calibration_samples_dx)
                net_v = sum(self.calibration_samples_dy)
                net_h = sum(self.calibration_samples_dx)

                if total_abs_v >= total_abs_h:
                    self.inferred_allowed_direction = "DOWN" if net_v > 0 else "UP"
                else:
                    self.inferred_allowed_direction = "RIGHT" if net_h > 0 else "LEFT"

                self.is_calibrated = True
                print(
                    f"[*] [AUTO-FLOW] Calibrated baseline flow: {self.inferred_allowed_direction} "
                    f"(vertical_abs={total_abs_v:.2f}, horizontal_abs={total_abs_h:.2f}, "
                    f"samples={len(self.calibration_samples_dy)})"
                )

        # Violation alerts are only evaluated after baseline flow is established
        can_check_violations = (self.allowed_direction != "AUTO") or self.is_calibrated

        for veh in tracks:
            # 1. Update vehicle direction estimate from trajectory history
            dx, dy, dist = calculate_motion_vector(
                veh.trajectory, window=self.config.trajectory_history_length
            )
            current_dir = estimate_cardinal_direction(
                dx, dy, min_distance=self.config.min_movement_distance, frame_size=frame_size
            )
            veh.direction = current_dir

            if current_dir in self.direction_counts:
                self.direction_counts[current_dir] += 1

            # Skip immature tracks for violation checking until trajectory stabilizes
            if not can_check_violations or len(veh.trajectory) < self.min_track_history:
                continue

            # 2. Check for opposite direction violation
            if self._is_opposite_direction(current_dir, dx=dx, dy=dy):
                veh.violation_frames += 1

                # 3. Temporal Confirmation Gate:
                # Require N consecutive violation frames before raising an alarm
                if (
                    veh.violation_frames >= self.confirm_frames
                    and not veh.violation_alerted
                ):
                    veh.violation_alerted = True
                    snapshot_path = self._save_violation_snapshot(
                        frame, veh, frame_idx
                    )

                    effective_allowed = (
                        self.inferred_allowed_direction or self.allowed_direction
                    )
                    event = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "frame_idx": frame_idx,
                        "track_id": veh.track_id,
                        "class_name": veh.class_name,
                        "direction": veh.direction,
                        "allowed_direction": effective_allowed,
                        "confidence": round(float(veh.confidence), 4),
                        "snapshot_path": str(snapshot_path),
                    }
                    self.violations.append(event)
                    new_violations.append(event)
            else:
                # Vehicle is moving legally or stationary: decay/reset violation counter
                if veh.violation_frames > 0:
                    veh.violation_frames = max(0, veh.violation_frames - 1)

        return new_violations

    def _save_violation_snapshot(
        self,
        frame: np.ndarray,
        veh: TrackedVehicle,
        frame_idx: int,
    ) -> Path:
        """
        Annotates the current frame with a prominent red violation badge and saves to disk.
        """
        snapshot = frame.copy()
        x1, y1, x2, y2 = veh.bbox

        # Draw intense red bounding box
        alert_color = (0, 0, 255)  # BGR Red
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), alert_color, 3)

        # Header warning banner
        effective_allowed = self.inferred_allowed_direction or self.allowed_direction
        banner_text = (
            f"VIOLATION: ID {veh.track_id} {veh.class_name.upper()} | "
            f"Moving: {veh.direction} | Allowed: {effective_allowed}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(snapshot, (0, 0), (snapshot.shape[1], 40), (0, 0, 180), -1)
        cv2.putText(
            snapshot,
            banner_text,
            (15, 26),
            font,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        filename = f"violation_id{veh.track_id}_f{frame_idx:04d}.jpg"
        save_path = self.snapshots_dir / filename
        cv2.imwrite(str(save_path), snapshot)
        return save_path

    def draw_alerts(
        self,
        frame: np.ndarray,
        tracks: List[TrackedVehicle],
    ) -> np.ndarray:
        """
        Renders warning banners and red alert boxes for vehicles currently in confirmed violation.

        Args:
            frame: Image array to annotate.
            tracks: Active TrackedVehicle instances.

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()
        alert_color = (0, 0, 255)  # Red
        has_active_violation = False

        for veh in tracks:
            # Draw persistent red highlight if vehicle is confirmed in wrong-way motion
            if veh.violation_frames >= self.confirm_frames:
                has_active_violation = True
                x1, y1, x2, y2 = veh.bbox

                # Thick red bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), alert_color, 3)

                # Floating "! WRONG WAY !" warning badge
                label = f"! WRONG WAY ! (ID: {veh.track_id})"
                cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + 190, y1), alert_color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 5, max(15, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # If any violation is currently active, render top flashing alert bar
        if has_active_violation:
            h, w = annotated.shape[:2]
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.putText(
                annotated,
                "[!] WRONG-WAY TRAFFIC ALERT ACTIVE [!]",
                (w // 2 - 180, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated


def main():
    """CLI runner to test direction detection and wrong-way violation alerting on a video."""
    import argparse
    from src.video_processor import VideoReader, VideoWriterHelper
    from src.tracker import VehicleTracker

    parser = argparse.ArgumentParser(description="Test wrong-way vehicle violation detection.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/input/traffic.mp4",
        help="Path to input video (default: data/input/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/violation_test.mp4",
        help="Destination path for violation output video",
    )
    parser.add_argument(
        "--allowed",
        type=str,
        default="DOWN",
        help="Allowed legal traffic flow: 'DOWN' or 'UP' (default: DOWN)",
    )
    parser.add_argument(
        "--confirm-frames",
        type=int,
        default=4,
        help="Consecutive frames required to confirm wrong-way violation (default: 4)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live preview window",
    )
    args = parser.parse_args()

    cfg = TrafficConfig(
        video_source=args.source,
        allowed_direction=args.allowed,
        violation_confirm_frames=args.confirm_frames,
    )
    tracker = VehicleTracker(cfg)
    detector = WrongWayDetector(cfg)

    print(f"[x] Initialized WrongWayDetector:")
    print(f"    - Allowed Direction: {detector.allowed_direction}")
    print(f"    - Confirmation Window: {detector.confirm_frames} frames")
    print(f"    - Snapshots Target: {detector.snapshots_dir.resolve()}")

    with VideoReader(args.source) as reader:
        meta = reader.metadata
        print(f"[*] Input Video: {meta.width}x{meta.height} @ {meta.fps} FPS ({meta.total_frames} frames)")

        with VideoWriterHelper(
            output_path=args.output,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
        ) as writer:
            for idx, frame in reader.read_frames():
                # 1. Update Tracker
                active_tracks = tracker.update(frame, idx)

                # 2. Update Wrong-Way Detector
                new_violations = detector.update(frame, active_tracks, idx)
                for v in new_violations:
                    print(
                        f"  [ALERT] Frame {idx:03d} | Confirmed WRONG-WAY: {v['class_name']} "
                        f"(ID: {v['track_id']}) moving {v['direction']}! Snapshot saved: {v['snapshot_path']}"
                    )

                # 3. Draw Tracks and Alert Badges
                annotated = tracker.draw_tracks(frame, active_tracks, draw_trajectory=True)
                annotated = detector.draw_alerts(annotated, active_tracks)

                writer.write(annotated)

                if args.preview:
                    cv2.imshow("Wrong-Way Violation Detection Test", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.preview:
                cv2.destroyAllWindows()

            print("=" * 60)
            print("[x] WRONG-WAY VIOLATION SUMMARY:")
            print(f"    - Total Confirmed Violations: {len(detector.violations)}")
            for v in detector.violations:
                print(f"    - Frame {v['frame_idx']} | ID: {v['track_id']} ({v['class_name']}) | Snapshot: {v['snapshot_path']}")
            print(f"    - Output video saved to: {args.output}")
            print("=" * 60)


if __name__ == "__main__":
    main()
