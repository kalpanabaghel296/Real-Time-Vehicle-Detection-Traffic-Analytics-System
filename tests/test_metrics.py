"""
Unit tests for PerformanceMonitor (src/metrics.py).
"""

import time
import numpy as np
import pytest
from src.metrics import PerformanceMonitor, FrameMetrics


def test_performance_monitor_initialization():
    monitor = PerformanceMonitor(window_size=15)
    assert monitor.window_size == 15
    assert monitor.total_frames_profiled == 0
    assert monitor.avg_fps == 0.0


def test_frame_metrics_recording():
    monitor = PerformanceMonitor(window_size=10)

    for i in range(5):
        monitor.start_frame()
        time.sleep(0.002)  # 2 ms simulate preprocess
        monitor.mark_preprocessed()
        time.sleep(0.005)  # 5 ms simulate inference
        monitor.mark_inference_complete()
        time.sleep(0.001)  # 1 ms simulate tracking
        monitor.mark_tracking_complete()
        time.sleep(0.001)  # 1 ms simulate analytics
        m = monitor.end_frame(frame_idx=i)

        assert isinstance(m, FrameMetrics)
        assert m.frame_idx == i
        assert m.preprocess_ms > 0
        assert m.inference_ms > 0
        assert m.tracking_ms > 0
        assert m.total_ms > 0
        assert m.fps > 0

    assert monitor.total_frames_profiled == 5
    summary = monitor.get_summary()
    assert summary["total_frames"] == 5
    assert summary["avg_fps"] > 0
    assert summary["avg_inference_ms"] > 0
    assert summary["avg_total_ms"] > 0


def test_draw_hud():
    monitor = PerformanceMonitor()
    monitor.start_frame()
    monitor.mark_preprocessed()
    monitor.mark_inference_complete()
    monitor.mark_tracking_complete()
    monitor.end_frame(0)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = monitor.draw_hud(frame)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    # Check that text was drawn onto the image
    assert np.any(annotated != 0)
