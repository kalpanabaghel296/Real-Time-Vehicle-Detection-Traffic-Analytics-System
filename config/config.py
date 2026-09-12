"""
Centralized Configuration Module
=================================
Defines all hyper-parameters, thresholds, and paths for the traffic analytics system.
Organized as a clean Python dataclass for type-safety and easy inspection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class TrafficConfig:
    """
    Configuration parameters for the Real-Time Vehicle Detection & Traffic Analytics System.
    """

    # -------------------------------------------------------------------------
    # 1. Video Input & Output Paths
    # -------------------------------------------------------------------------
    # Path to video file, image folder, RTSP stream URL, or integer (e.g. 0 for webcam)
    video_source: str = "data/input/traffic.mp4"

    # Directory destinations for generated artifacts
    output_video_dir: str = "outputs/videos"
    snapshots_dir: str = "outputs/snapshots"
    logs_dir: str = "outputs/logs"
    models_dir: str = "models"

    # Output toggles
    save_output_video: bool = True
    save_snapshots: bool = True
    log_events_to_csv: bool = True
    show_preview: bool = True

    # -------------------------------------------------------------------------
    # 2. YOLO Object Detection Settings
    # -------------------------------------------------------------------------
    # Model name or local file path (Ultralytics auto-downloads 'yolov8n.pt' if not present)
    model_name: str = "yolov8n.pt"

    # Inference resolution: YOLO scales the frame while preserving aspect ratio
    input_size: Tuple[int, int] = (640, 640)
    imgsz: int = 640

    # Minimum confidence score [0.0 - 1.0] for a candidate detection box to be accepted
    # Set to 0.35 to balance high precision with recall on moving vehicles
    confidence_threshold: float = 0.35

    # Non-Maximum Suppression (NMS) Intersection-over-Union (IoU) threshold
    # Boxes with IoU > iou_threshold with a higher confidence box are suppressed
    iou_threshold: float = 0.50

    # Device selection: 'cuda', 'cuda:0', or 'cpu'. If None, auto-selects GPU if available.
    device: Optional[str] = None

    # COCO Class IDs for traffic entities:
    # 2: car, 3: motorcycle, 5: bus, 7: truck
    # Reference: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    target_class_ids: List[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # Human-readable mapping of class IDs to labels
    class_names: dict = field(
        default_factory=lambda: {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
        }
    )

    # -------------------------------------------------------------------------
    # 3. Multi-Object Tracking Settings (ByteTrack)
    # -------------------------------------------------------------------------
    # Threshold for associating high-confidence detections
    track_thresh: float = 0.45

    # Number of frames to maintain a track before deleting it if detection is lost
    track_buffer: int = 50

    # Minimum IoU matching score for bounding box association
    match_thresh: float = 0.80

    # -------------------------------------------------------------------------
    # 4. Virtual Counting Line Settings
    # -------------------------------------------------------------------------
    # Coordinates of the counting line: ((x1, y1), (x2, y2))
    # Normalized coordinates [0.0 - 1.0] relative to frame width and height,
    # or absolute pixel coordinates. If None, auto-calculated across the middle.
    counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    # Line orientation: "AUTO", "HORIZONTAL" (for UP/DOWN flow), "VERTICAL" (for LEFT/RIGHT flow)
    line_orientation: str = "AUTO"

    # Allowed counting flow: "ANY", "DOWN", "UP", "LEFT", "RIGHT"
    counting_direction: str = "ANY"

    # -------------------------------------------------------------------------
    # 5. Direction & Wrong-Way Detection Settings
    # -------------------------------------------------------------------------
    # Configured legal flow of traffic on the lane: "AUTO", "DOWN", "UP", "LEFT", "RIGHT"
    # When "AUTO", the system infers the legal flow automatically from dominant traffic vectors.
    allowed_direction: str = "AUTO"

    # Centroid history buffer length (number of past frame positions kept per vehicle)
    trajectory_history_length: int = 30

    # Minimum pixel movement required to calculate direction (filters stationary/jittery detections)
    min_movement_distance: float = 15.0

    # Temporal confirmation window: consecutive frames a vehicle must violate allowed direction
    # before raising a confirmed violation alert (prevents single-frame detector noise)
    violation_confirm_frames: int = 3

    # -------------------------------------------------------------------------
    # 6. Performance & Optimization Settings
    # -------------------------------------------------------------------------
    # Frame skipping: process 1 out of every (frame_skip + 1) frames (0 = process every frame)
    frame_skip: int = 0

    # Target output video FPS (if None, matches source video FPS)
    output_fps: Optional[float] = None

    def ensure_directories(self) -> None:
        """Ensures all output directories exist on disk."""
        for path_str in [
            self.output_video_dir,
            self.snapshots_dir,
            self.logs_dir,
            self.models_dir,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)
