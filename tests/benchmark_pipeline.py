"""
Pipeline Benchmark & Inference Optimization Suite
=================================================
Runs rigorous, empirical benchmarks across varied resolutions, models, and frame-skip settings.
Measures ACTUAL runtime metrics (no fabricated numbers) to populate documentation and reports.
"""

from pathlib import Path
import sys
import time
from typing import List, Dict, Any
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.video_processor import VideoReader
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.violation import WrongWayDetector
from src.metrics import PerformanceMonitor


def run_benchmark_configuration(
    video_path: str,
    input_size: tuple,
    frame_skip: int,
    max_frames: int = 50,
) -> Dict[str, Any]:
    """Runs end-to-end pipeline benchmark under specified configuration."""
    cfg = TrafficConfig(
        video_source=video_path,
        input_size=input_size,
        frame_skip=frame_skip,
    )

    tracker = VehicleTracker(cfg)
    counter = VehicleCounter(cfg)
    detector = WrongWayDetector(cfg)
    monitor = PerformanceMonitor()

    processed_frames = 0
    start_time = time.perf_counter()

    with VideoReader(video_path) as reader:
        for idx, frame in reader.read_frames(frame_skip=frame_skip):
            if processed_frames >= max_frames:
                break

            monitor.start_frame()

            # Preprocessing simulation (resize to model input)
            monitor.mark_preprocessed()

            # Detection & Tracking
            active_tracks = tracker.update(frame, idx)
            monitor.mark_inference_complete()

            # Analytics (Counter + Violation)
            counter.update(active_tracks, frame.shape[:2], idx)
            detector.update(frame, active_tracks, idx)
            monitor.mark_tracking_complete()

            monitor.end_frame(idx)
            processed_frames += 1

    total_wall_time = time.perf_counter() - start_time
    actual_fps = processed_frames / total_wall_time if total_wall_time > 0 else 0.0
    summary = monitor.get_summary()

    return {
        "resolution": f"{input_size[0]}x{input_size[1]}",
        "frame_skip": frame_skip,
        "processed_frames": processed_frames,
        "wall_time_s": round(total_wall_time, 2),
        "actual_fps": round(actual_fps, 2),
        "avg_inference_ms": summary["avg_inference_ms"],
        "avg_total_ms": summary["avg_total_ms"],
    }


def main():
    video_source = "data/input/traffic.mp4"
    print("=" * 75)
    print("TRAFFIC PIPELINE BENCHMARK: MEASURING ACTUAL RUNTIME PERFORMANCE")
    print(f"Video Source: {video_source} (Benchmarking first 40 frames per config)")
    print("=" * 75)

    configurations = [
        {"size": (640, 640), "skip": 0, "name": "YOLOv8n @ 640x640 (Baseline)"},
        {"size": (480, 480), "skip": 0, "name": "YOLOv8n @ 480x480 (Downscaled)"},
        {"size": (320, 320), "skip": 0, "name": "YOLOv8n @ 320x320 (Lightweight)"},
        {"size": (640, 640), "skip": 1, "name": "YOLOv8n @ 640x640 (Frame Skip=1)"},
    ]

    results: List[Dict[str, Any]] = []

    for c in configurations:
        print(f"[*] Running configuration: {c['name']} ...")
        res = run_benchmark_configuration(
            video_path=video_source,
            input_size=c["size"],
            frame_skip=c["skip"],
            max_frames=40,
        )
        res["config_name"] = c["name"]
        results.append(res)
        print(f"    -> FPS: {res['actual_fps']} | Avg Inference: {res['avg_inference_ms']} ms | Total: {res['avg_total_ms']} ms")

    print("\n" + "=" * 75)
    print("FINAL MEASURED BENCHMARK TABLE (COPY TO DOCS/PERFORMANCE.MD)")
    print("=" * 75)
    print("| Configuration | Resolution | Frame Skip | Real-Time FPS | Avg Inference Latency | Total Frame Latency |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(
            f"| {r['config_name']} | {r['resolution']} | {r['frame_skip']} | "
            f"**{r['actual_fps']}** | {r['avg_inference_ms']} ms | {r['avg_total_ms']} ms |"
        )
    print("=" * 75)


if __name__ == "__main__":
    main()
