"""
YOLO Vehicle Detector Module
============================
Integrates Ultralytics YOLOv8 for vehicle object detection, class filtering,
confidence thresholding, and bounding box visualization.

Note:
This module utilizes a pretrained YOLOv8 model (trained on the MS COCO dataset).
It performs inference to detect traffic objects (car, motorcycle, bus, truck)
without claiming custom model training.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig


@dataclass
class Detection:
    """
    Standardized container for a single object detection.

    Attributes:
        class_id: Numeric class index from the model (e.g. 2 for COCO car).
        class_name: Human-readable label (e.g. 'car', 'truck').
        confidence: Prediction confidence score in range [0.0, 1.0].
        bbox: Bounding box coordinates in pixel space (x1, y1, x2, y2).
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict[str, Any]:
        """Converts detection to a serializable dictionary format."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox),
        }

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the centroid coordinate (cx, cy) of the detection."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)


# Color map (BGR) for distinct vehicle classes
CLASS_COLORS = {
    "car": (255, 140, 0),        # Deep sky blue / orange
    "motorcycle": (0, 215, 255), # Amber / yellow
    "bus": (50, 205, 50),        # Lime green
    "truck": (180, 105, 255),    # Hot pink / purple
    "person": (0, 165, 255),     # Orange
}
DEFAULT_COLOR = (0, 255, 0)      # Green


class YOLOVehicleDetector:
    """
    Vehicle detector wrapping Ultralytics YOLOv8.

    Handles:
    - Model initialization & device selection (CUDA GPU if available, else CPU)
    - Forward inference
    - Confidence & NMS thresholding
    - Filtering strictly for vehicle classes (car, motorcycle, bus, truck)
    - Latency profiling (preprocess, inference, postprocess ms)
    """

    def __init__(self, config: Optional[TrafficConfig] = None):
        self.config = config or TrafficConfig()

        # Device selection: Auto-detect CUDA GPU unless specified
        if self.config.device is not None:
            self.device = self.config.device
        else:
            self.device = "0" if torch.cuda.is_available() else "cpu"

        # Load pretrained YOLO model
        model_source = self.config.model_name
        self.model = YOLO(model_source)

        # Profile timing from the most recent forward pass
        self.last_latency = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
            "total_ms": 0.0,
        }

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs object detection on a single BGR frame.

        Args:
            frame: Input image array (BGR format).

        Returns:
            List of Detection objects filtered for target vehicle classes.
        """
        if frame is None or frame.size == 0:
            return []

        # Determine inference resolution
        imgsz = getattr(self.config, "imgsz", None) or self.config.input_size[0]
        if str(imgsz).lower() == "auto":
            h, w = frame.shape[:2]
            imgsz = 1280 if max(h, w) >= 1920 else 640

        # Run inference through Ultralytics YOLO
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=imgsz,
            device=self.device,
            classes=self.config.target_class_ids,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]

        # Extract timing metrics if available
        if hasattr(result, "speed") and isinstance(result.speed, dict):
            pre_ms = result.speed.get("preprocess", 0.0)
            inf_ms = result.speed.get("inference", 0.0)
            post_ms = result.speed.get("postprocess", 0.0)
            self.last_latency = {
                "preprocess_ms": pre_ms,
                "inference_ms": inf_ms,
                "postprocess_ms": post_ms,
                "total_ms": pre_ms + inf_ms + post_ms,
            }

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract detection arrays
        coords = boxes.xyxy.cpu().numpy()  # [N, 4] array of (x1, y1, x2, y2)
        confs = boxes.conf.cpu().numpy()   # [N] array of confidences
        class_ids = boxes.cls.cpu().numpy().astype(int)  # [N] array of class IDs

        for i in range(len(coords)):
            cid = int(class_ids[i])
            conf = float(confs[i])
            x1, y1, x2, y2 = coords[i]

            # Label lookup
            cname = self.config.class_names.get(
                cid, result.names.get(cid, f"class_{cid}")
            )

            detection = Detection(
                class_id=cid,
                class_name=cname,
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            )
            detections.append(detection)

        return detections

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Renders bounding boxes and label badges onto a copy of the frame.

        Args:
            frame: Original BGR image.
            detections: List of Detection instances to render.

        Returns:
            Annotated BGR image.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name.lower(), DEFAULT_COLOR)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Format label badge: "car: 0.91"
            label = f"{det.class_name} {det.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw label background banner
            banner_y1 = max(0, y1 - text_h - baseline - 4)
            banner_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, banner_y1),
                (x1 + text_w + 6, banner_y2),
                color,
                -1,
            )

            # Draw label text (white or black depending on contrast)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        return annotated


def main():
    """CLI runner to test YOLO vehicle detection on a video stream."""
    import argparse
    from src.video_processor import VideoReader, VideoWriterHelper

    parser = argparse.ArgumentParser(description="Run YOLO vehicle detector on a video.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample/traffic_sample.mp4",
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/detection_test.mp4",
        help="Destination path for detection video",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live preview",
    )
    args = parser.parse_args()

    cfg = TrafficConfig(
        video_source=args.source,
        confidence_threshold=args.conf,
    )
    detector = YOLOVehicleDetector(cfg)
    print(f"[x] Initialized YOLO detector: {cfg.model_name} on device: {detector.device}")

    with VideoReader(args.source) as reader:
        meta = reader.metadata
        print(f"[*] Input Video: {meta.width}x{meta.height} @ {meta.fps} FPS ({meta.total_frames} frames)")

        with VideoWriterHelper(
            output_path=args.output,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
        ) as writer:
            total_detections = 0
            for idx, frame in reader.read_frames():
                detections = detector.detect(frame)
                total_detections += len(detections)
                annotated = detector.draw_detections(frame, detections)

                # Overlay latency & detection count
                lat = detector.last_latency
                hud_text = (
                    f"Frame {idx:03d} | Detections: {len(detections)} | "
                    f"Infer: {lat['inference_ms']:.1f}ms | Device: {detector.device}"
                )
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
                    cv2.imshow("YOLO Vehicle Detection Test", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.preview:
                cv2.destroyAllWindows()

            print(f"[x] Detection test complete! Processed {meta.total_frames} frames.")
            print(f"    - Total detections found: {total_detections}")
            print(f"    - Output saved to: {args.output}")


if __name__ == "__main__":
    main()
