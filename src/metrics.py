"""
Performance Monitoring Module
=============================
Measures and profiles actual runtime execution metrics:
- Real-time Instantaneous and Average FPS (Frames Per Second)
- Stage-by-stage latency breakdown:
  * Preprocessing time (ms)
  * Model inference time (ms)
  * Tracking and Analytics time (ms)
  * Total frame processing time (ms)
- Rolling window smoothing to eliminate transient timing spikes
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Dict, Any, Optional
import cv2
import numpy as np


@dataclass
class FrameMetrics:
    """Stores high-precision latency measurements for a single frame."""

    frame_idx: int
    preprocess_ms: float
    inference_ms: float
    tracking_ms: float
    analytics_ms: float
    total_ms: float
    fps: float


class PerformanceMonitor:
    """
    High-precision pipeline profiler using monotonic system timers.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent frames to use for rolling averages.
        """
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.total_frames_profiled: int = 0
        self.start_wall_time: Optional[float] = None
        self.last_frame_timestamp: Optional[float] = None

        # Stage timer checkpoints
        self._t_start: float = 0.0
        self._t_pre_end: float = 0.0
        self._t_inf_end: float = 0.0
        self._t_track_end: float = 0.0
        self._t_post_end: float = 0.0

    def start_frame(self) -> None:
        """Marks the start of processing for a new video frame."""
        now = time.perf_counter()
        if self.start_wall_time is None:
            self.start_wall_time = now
        self._t_start = now
        self.last_frame_timestamp = now

    def mark_preprocessed(self) -> None:
        """Marks the completion of frame resizing and color space conversion."""
        self._t_pre_end = time.perf_counter()

    def mark_inference_complete(self) -> None:
        """Marks the completion of the deep learning forward pass."""
        self._t_inf_end = time.perf_counter()

    def mark_tracking_complete(self) -> None:
        """Marks the completion of multi-object association and Kalman filtering."""
        self._t_track_end = time.perf_counter()

    def end_frame(self, frame_idx: int = 0) -> FrameMetrics:
        """
        Finalizes frame profiling, computes stage latencies, and updates rolling stats.
        """
        now = time.perf_counter()
        self._t_post_end = now

        # Compute durations in milliseconds (1s = 1000ms)
        pre_ms = max(0.0, (self._t_pre_end - self._t_start) * 1000.0)
        inf_ms = max(0.0, (self._t_inf_end - self._t_pre_end) * 1000.0)
        track_ms = max(0.0, (self._t_track_end - self._t_inf_end) * 1000.0)
        post_ms = max(0.0, (self._t_post_end - self._t_track_end) * 1000.0)
        total_ms = max(0.001, (now - self._t_start) * 1000.0)

        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        metrics = FrameMetrics(
            frame_idx=frame_idx,
            preprocess_ms=pre_ms,
            inference_ms=inf_ms,
            tracking_ms=track_ms,
            analytics_ms=post_ms,
            total_ms=total_ms,
            fps=fps,
        )

        self.history.append(metrics)
        self.total_frames_profiled += 1
        return metrics

    @property
    def current_fps(self) -> float:
        """Instantaneous FPS from the most recent frame."""
        if not self.history:
            return 0.0
        return self.history[-1].fps

    @property
    def avg_fps(self) -> float:
        """Rolling window average FPS."""
        if not self.history:
            return 0.0
        avg_total = sum(m.total_ms for m in self.history) / len(self.history)
        return (1000.0 / avg_total) if avg_total > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Rolling window average inference latency in milliseconds."""
        if not self.history:
            return 0.0
        return sum(m.inference_ms for m in self.history) / len(self.history)

    @property
    def avg_total_ms(self) -> float:
        """Rolling window average total frame processing latency in milliseconds."""
        if not self.history:
            return 0.0
        return sum(m.total_ms for m in self.history) / len(self.history)

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated performance statistics."""
        if not self.history:
            return {
                "total_frames": 0,
                "avg_fps": 0.0,
                "avg_inference_ms": 0.0,
                "avg_total_ms": 0.0,
            }

        n = len(self.history)
        return {
            "total_frames": self.total_frames_profiled,
            "avg_fps": round(self.avg_fps, 2),
            "avg_preprocess_ms": round(sum(m.preprocess_ms for m in self.history) / n, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "avg_tracking_ms": round(sum(m.tracking_ms for m in self.history) / n, 2),
            "avg_analytics_ms": round(sum(m.analytics_ms for m in self.history) / n, 2),
            "avg_total_ms": round(self.avg_total_ms, 2),
        }

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlays performance metrics (FPS and latency) onto the top-right corner of the frame.

        Format:
        FPS: 28.4
        Inference: 35 ms
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        badge_w = 175
        badge_h = 55
        badge_x = w - badge_w - 15
        badge_y = 15

        # Background badge with dark transparency
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (15, 15, 15),
            -1,
        )
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
        cv2.rectangle(
            annotated,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (60, 60, 60),
            1,
        )

        # Performance text
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps_val = self.avg_fps
        inf_val = self.avg_inference_ms

        fps_color = (0, 255, 0) if fps_val >= 20.0 else (0, 165, 255)
        cv2.putText(
            annotated,
            f"FPS: {fps_val:.1f}",
            (badge_x + 10, badge_y + 22),
            font,
            0.55,
            fps_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Inference: {inf_val:.1f} ms",
            (badge_x + 10, badge_y + 44),
            font,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return annotated
