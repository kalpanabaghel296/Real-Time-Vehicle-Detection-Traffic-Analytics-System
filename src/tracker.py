"""
Multi-Object Vehicle Tracking Module (ByteTrack)
================================================
Maintains persistent identities for moving vehicles across frames using ByteTrack.
Tracks trajectory history, centroid positions, frame lifespans, and handles occlusions.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.utils import calculate_centroid, calculate_displacement_vector
from src.detector import CLASS_COLORS, DEFAULT_COLOR


@dataclass
class TrackedVehicle:
    """
    State container for a single tracked vehicle over time.

    Attributes:
        track_id: Unique persistent ID assigned by the tracker.
        class_id: Numeric COCO class ID.
        class_name: Human-readable class name ('car', 'truck', etc.).
        confidence: Most recent detection confidence score.
        bbox: Current bounding box (x1, y1, x2, y2).
        centroid: Current center point (cx, cy).
        previous_centroid: Center point in the prior observed frame.
        trajectory: FIFO queue of historical centroid coordinates.
        first_frame: Frame index when this vehicle was first detected.
        last_frame: Frame index when this vehicle was most recently updated.
        counted: Flag indicating whether this vehicle was already counted at the line.
        direction: Estimated direction of movement ('DOWN', 'UP', 'LEFT', 'RIGHT').
        violation_frames: Consecutive frames vehicle has violated traffic rules.
        violation_alerted: Flag indicating whether a wrong-way snapshot was saved.
    """

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[int, int]
    previous_centroid: Optional[Tuple[int, int]] = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    first_frame: int = 0
    last_frame: int = 0
    counted: bool = False
    direction: Optional[str] = None
    violation_frames: int = 0
    violation_alerted: bool = False

    @property
    def displacement(self) -> Tuple[float, float]:
        """Returns 2D displacement (dx, dy) from previous centroid to current centroid."""
        if self.previous_centroid is None:
            return (0.0, 0.0)
        return calculate_displacement_vector(self.previous_centroid, self.centroid)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current tracked vehicle state to a dictionary."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
            "centroid": list(self.centroid),
            "previous_centroid": list(self.previous_centroid) if self.previous_centroid else None,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "counted": self.counted,
            "direction": self.direction,
        }


class VehicleTracker:
    """
    High-level Multi-Object Tracking manager.

    Integrates ByteTrack via Ultralytics with persistent state tracking,
    trajectory history accumulation, and stale track pruning.
    """

    def __init__(self, config: Optional[TrafficConfig] = None):
        self.config = config or TrafficConfig()

        # Device selection: GPU if available, else CPU
        if self.config.device is not None:
            self.device = self.config.device
        else:
            self.device = "0" if torch.cuda.is_available() else "cpu"

        # Load YOLO model for combined detection + ByteTrack tracking
        self.model = YOLO(self.config.model_name)

        # Active tracked vehicles mapped by track_id: Dict[int, TrackedVehicle]
        self.tracks: Dict[int, TrackedVehicle] = {}

    def update(self, frame: np.ndarray, frame_idx: int) -> List[TrackedVehicle]:
        """
        Updates tracking state for a new video frame.

        Args:
            frame: Input BGR image array.
            frame_idx: Current sequential frame index.

        Returns:
            List of active TrackedVehicle instances present in this frame.
        """
        if frame is None or frame.size == 0:
            return []

        # Run ByteTrack through Ultralytics
        # persist=True maintains Kalman filter states across consecutive frames
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.input_size,
            device=self.device,
            classes=self.config.target_class_ids,
            verbose=False,
        )

        active_in_frame: List[TrackedVehicle] = []
        if not results:
            return active_in_frame

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return active_in_frame

        # Some candidate boxes may not have received a track ID yet (unmatched low confidence)
        if boxes.id is None:
            return active_in_frame

        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = boxes.id.cpu().numpy().astype(int)

        current_frame_ids = set()

        for i in range(len(track_ids)):
            tid = int(track_ids[i])
            cid = int(class_ids[i])
            conf = float(confs[i])
            x1, y1, x2, y2 = coords[i]
            bbox = (int(x1), int(y1), int(x2), int(y2))
            current_centroid = calculate_centroid(bbox)
            cname = self.config.class_names.get(
                cid, result.names.get(cid, f"class_{cid}")
            )

            current_frame_ids.add(tid)

            if tid in self.tracks:
                # Update existing track
                veh = self.tracks[tid]
                veh.previous_centroid = veh.centroid
                veh.centroid = current_centroid
                veh.bbox = bbox
                veh.confidence = conf
                veh.class_id = cid
                veh.class_name = cname
                veh.last_frame = frame_idx
                veh.trajectory.append(current_centroid)
            else:
                # Initialize new tracked vehicle
                traj = deque(maxlen=self.config.trajectory_history_length)
                traj.append(current_centroid)
                veh = TrackedVehicle(
                    track_id=tid,
                    class_id=cid,
                    class_name=cname,
                    confidence=conf,
                    bbox=bbox,
                    centroid=current_centroid,
                    previous_centroid=None,
                    trajectory=traj,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                )
                self.tracks[tid] = veh

            active_in_frame.append(self.tracks[tid])

        # Prune stale tracks that have not been observed within track_buffer frames
        stale_threshold = frame_idx - self.config.track_buffer
        expired_ids = [
            tid for tid, veh in self.tracks.items()
            if veh.last_frame < stale_threshold
        ]
        for tid in expired_ids:
            del self.tracks[tid]

        return active_in_frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackedVehicle],
        draw_trajectory: bool = True,
    ) -> np.ndarray:
        """
        Renders bounding boxes, persistent Track IDs, labels, and trajectory trails.

        Badge Format:
        ID: 1 | Car | 0.92

        Args:
            frame: Original BGR image.
            tracks: List of TrackedVehicle objects in the current frame.
            draw_trajectory: Whether to draw historical centroid trails.

        Returns:
            Annotated BGR image.
        """
        annotated = frame.copy()

        for veh in tracks:
            x1, y1, x2, y2 = veh.bbox
            color = CLASS_COLORS.get(veh.class_name.lower(), DEFAULT_COLOR)

            # 1. Draw Trajectory Trail (fading lines from past to present)
            if draw_trajectory and len(veh.trajectory) > 1:
                points = list(veh.trajectory)
                for j in range(1, len(points)):
                    thickness = int(np.sqrt(float(j) / len(points) * 9.0)) + 1
                    cv2.line(annotated, points[j - 1], points[j], color, thickness)

            # 2. Draw Current Centroid Point
            cv2.circle(annotated, veh.centroid, 4, (0, 255, 255), -1)

            # 3. Draw Bounding Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 4. Draw Header Badge: "ID: 1 | Car | 0.92"
            label = f"ID: {veh.track_id} | {veh.class_name.capitalize()} | {veh.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            badge_y1 = max(0, y1 - text_h - baseline - 6)
            badge_y2 = y1

            # Background rectangle for text readability
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y2),
                color,
                -1,
            )

            # White text
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

        return annotated


def main():
    """CLI runner to test ByteTrack vehicle tracking on a video stream."""
    import argparse
    from src.video_processor import VideoReader, VideoWriterHelper

    parser = argparse.ArgumentParser(description="Test ByteTrack vehicle tracking on video.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/input/traffic.mp4",
        help="Path to input video (default: data/input/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/tracking_test.mp4",
        help="Destination path for tracked video",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live preview window",
    )
    args = parser.parse_args()

    cfg = TrafficConfig(video_source=args.source)
    tracker = VehicleTracker(cfg)
    print(f"[x] Initialized VehicleTracker (ByteTrack) on device: {tracker.device}")

    with VideoReader(args.source) as reader:
        meta = reader.metadata
        print(f"[*] Input Video: {meta.width}x{meta.height} @ {meta.fps} FPS ({meta.total_frames} frames)")

        with VideoWriterHelper(
            output_path=args.output,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
        ) as writer:
            unique_ids = set()

            for idx, frame in reader.read_frames():
                active_tracks = tracker.update(frame, idx)
                for t in active_tracks:
                    unique_ids.add(t.track_id)

                annotated = tracker.draw_tracks(frame, active_tracks, draw_trajectory=True)

                # HUD overlay
                hud_text = f"Frame: {idx:03d} | Active Tracks: {len(active_tracks)} | Total Unique Vehicles: {len(unique_ids)}"
                cv2.putText(
                    annotated,
                    hud_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(annotated)

                if args.preview:
                    cv2.imshow("ByteTrack Vehicle Tracking Test", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.preview:
                cv2.destroyAllWindows()

            print(f"[x] Tracking test complete! Processed {meta.total_frames} frames.")
            print(f"    - Unique tracked vehicles identified: {len(unique_ids)}")
            print(f"    - Output saved to: {args.output}")


if __name__ == "__main__":
    main()
