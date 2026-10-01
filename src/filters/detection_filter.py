"""
Detection Filter and Post-Processing
====================================
Implements post-detection optimization:
1. Per-Class Optimal Confidence Thresholds
2. Class-Specific Physical Size & Aspect Ratio Priors
3. Temporal Consistency Verification (eliminates single-frame flicker)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.utils import calculate_iou


@dataclass
class DetectionFilterConfig:
    """Configuration for post-detection filters."""

    # Per-class confidence thresholds optimized via empirical F1 sweep
    per_class_confidence: Dict[str, float] = field(
        default_factory=lambda: {
            "car": 0.40,
            "truck": 0.35,
            "bus": 0.35,
            "motorcycle": 0.30,
        }
    )

    # Physical Size & Aspect Ratio Priors (w = x2 - x1, h = y2 - y1, ratio = h / w)
    enable_size_priors: bool = True
    class_size_priors: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "car": {
                "min_area": 300.0,
                "max_area": 250000.0,
                "min_aspect_ratio": 0.25,
                "max_aspect_ratio": 3.00,
            },
            "truck": {
                "min_area": 800.0,
                "max_area": 450000.0,
                "min_aspect_ratio": 0.20,
                "max_aspect_ratio": 3.50,
            },
            "bus": {
                "min_area": 800.0,
                "max_area": 450000.0,
                "min_aspect_ratio": 0.20,
                "max_aspect_ratio": 3.50,
            },
            "motorcycle": {
                "min_area": 120.0,
                "max_area": 80000.0,
                "min_aspect_ratio": 0.40,
                "max_aspect_ratio": 4.00,
            },
        }
    )

    # Temporal Consistency Filter
    enable_temporal_consistency: bool = True
    temporal_window_frames: int = 3
    temporal_min_hits: int = 2
    temporal_iou_match: float = 0.40


class DetectionFilter:
    """
    Applies per-class confidence, size priors, and temporal consistency
    to raw object detection proposals.
    """

    def __init__(self, config: Optional[DetectionFilterConfig] = None):
        self.config = config or DetectionFilterConfig()
        # Ring buffer storing detections from prior frames: deque of List[dict]
        self._history: deque = deque(maxlen=self.config.temporal_window_frames)
        self._last_frame_idx: int = -1

    def reset(self) -> None:
        """Clears temporal state buffer (e.g. at video boundary)."""
        self._history.clear()
        self._last_frame_idx = -1

    def filter_by_confidence(self, detections: List[Any]) -> List[Any]:
        """
        Filters detections by class-specific confidence thresholds.
        """
        filtered = []
        for det in detections:
            cname = getattr(det, "class_name", "").lower()
            conf = float(getattr(det, "confidence", 1.0))
            thresh = self.config.per_class_confidence.get(cname, 0.35)
            if conf >= thresh:
                filtered.append(det)
        return filtered

    def filter_by_size_priors(
        self, detections: List[Any], frame_shape: Optional[Tuple[int, int]] = None
    ) -> List[Any]:
        """
        Filters detections based on physical size (bounding box area)
        and geometry (aspect ratio = height / width).
        """
        if not self.config.enable_size_priors:
            return detections

        filtered = []
        for det in detections:
            cname = getattr(det, "class_name", "").lower()
            priors = self.config.class_size_priors.get(cname)
            if priors is None:
                filtered.append(det)
                continue

            bbox = getattr(det, "bbox", None)
            if bbox is None:
                filtered.append(det)
                continue

            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            area = w * h
            if w <= 0 or h <= 0:
                continue

            ratio = h / w

            # Reject if outside physical limits
            if area < priors["min_area"] or area > priors["max_area"]:
                continue
            if ratio < priors["min_aspect_ratio"] or ratio > priors["max_aspect_ratio"]:
                continue

            # Reject boxes covering more than 75% of frame (unrealistic global hallucination)
            if frame_shape is not None:
                f_h, f_w = frame_shape[:2]
                frame_area = f_w * f_h
                if area > 0.75 * frame_area:
                    continue

            filtered.append(det)

        return filtered

    def filter_by_temporal_consistency(
        self, detections: List[Any], frame_idx: int
    ) -> List[Any]:
        """
        Filters candidate detections through a multi-frame temporal confirmation buffer.
        A detection must have overlapping matches in at least M of the last N frames
        to be confirmed, eliminating 1-frame transient flicker.
        """
        if not self.config.enable_temporal_consistency:
            return detections

        # If discontinuous jump or first frame, initialize buffer
        if self._last_frame_idx < 0 or frame_idx > self._last_frame_idx + 5:
            self._history.clear()

        self._last_frame_idx = frame_idx

        # If history is empty, record current detections and return them
        # (allows pipeline to bootstrap on initial frames)
        if len(self._history) < self.config.temporal_min_hits - 1:
            self._history.append([getattr(d, "bbox") for d in detections])
            return detections

        confirmed = []
        for det in detections:
            bbox = getattr(det, "bbox")
            hits = 1  # current frame counts as 1 hit

            for past_boxes in self._history:
                matched = False
                for pb in past_boxes:
                    if calculate_iou(bbox, pb) >= self.config.temporal_iou_match:
                        matched = True
                        break
                if matched:
                    hits += 1

            if hits >= self.config.temporal_min_hits:
                confirmed.append(det)

        # Update history with current candidate boxes (so future frames can match them)
        self._history.append([getattr(d, "bbox") for d in detections])
        return confirmed

    def apply(
        self,
        detections: List[Any],
        frame_shape: Optional[Tuple[int, int]] = None,
        frame_idx: int = 0,
    ) -> List[Any]:
        """
        Applies full pipeline of post-detection filters.
        """
        if not detections:
            # Still update temporal history for clean state tracking
            if self.config.enable_temporal_consistency:
                self._history.append([])
                self._last_frame_idx = frame_idx
            return []

        # 1. Per-class confidence gating
        step1 = self.filter_by_confidence(detections)

        # 2. Class-specific size & aspect ratio priors
        step2 = self.filter_by_size_priors(step1, frame_shape)

        # 3. Temporal multi-frame confirmation
        step3 = self.filter_by_temporal_consistency(step2, frame_idx)

        return step3
