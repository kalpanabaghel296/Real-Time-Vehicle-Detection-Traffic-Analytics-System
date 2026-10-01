"""
Real-Time Traffic & Vehicle Analytics System
============================================
Master CLI entry point integrating the complete Computer Vision pipeline:
1. Video Capture & Streaming
2. YOLOv8 Vehicle Detection
3. ByteTrack Multi-Object Tracking
4. Line-Based Vehicle Counting & Class Breakdown
5. Direction Estimation & Wrong-Way Violation Detection
6. Automated Evidence Snapshot Saving
7. Real-Time Performance Profiling (FPS & Latency)
8. Structured CSV/JSON Event Logging
9. Annotated Output Video Generation
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from config.config import TrafficConfig
from src.video_processor import VideoReader, VideoWriterHelper
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.violation import WrongWayDetector
from src.metrics import PerformanceMonitor
from src.logger import EventLogger
from src.visualizer import Visualizer


def run_pipeline(
    source: Any = "data/input/traffic.mp4",
    output: str = "outputs/videos/processed_video.mp4",
    allowed_direction: str = "AUTO",
    line_y: float = 0.35,
    conf_thresh: float = 0.35,
    frame_skip: int = 0,
    preview: bool = False,
    reset_logs: bool = False,
    imgsz: Any = 640,
    line_orientation: str = "AUTO",
    ignore_opposing_median: bool = False,
) -> Dict[str, Any]:
    """Runs the unified traffic analytics pipeline."""
    print("=" * 75)
    print("REAL-TIME VEHICLE DETECTION & TRAFFIC ANALYTICS SYSTEM")
    print("=" * 75)

    # Convert numeric camera source string to int if applicable
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())

    # Parse imgsz
    effective_imgsz = int(imgsz) if str(imgsz).isdigit() else str(imgsz)

    # 1. Initialize Configuration
    cfg = TrafficConfig(
        video_source=str(source),
        confidence_threshold=conf_thresh,
        allowed_direction=allowed_direction,
        frame_skip=frame_skip,
        imgsz=effective_imgsz,
        line_orientation=line_orientation,
        ignore_opposing_median=ignore_opposing_median,
    )
    cfg.ensure_directories()

    # If reset_logs is requested, purge previous snapshots so new runs start clean
    if reset_logs:
        snapshots_dir = Path(cfg.snapshots_dir)
        if snapshots_dir.exists():
            for p in snapshots_dir.glob("*.jpg"):
                try:
                    p.unlink()
                except Exception:
                    pass

    # 2. Instantiate Pipeline Modules
    print("[*] Initializing Computer Vision Modules...")
    tracker = VehicleTracker(cfg)
    counter = VehicleCounter(cfg, counting_direction="ANY")
    violation_detector = WrongWayDetector(cfg, allowed_direction=allowed_direction)
    monitor = PerformanceMonitor()
    logger = EventLogger(cfg, clear_existing=reset_logs)
    visualizer = Visualizer()

    print(f"[x] Pipeline ready:")
    print(f"    - Input Source: {source}")
    print(f"    - Model: {cfg.model_name} (Device: {tracker.device}, imgsz: {effective_imgsz})")
    print(f"    - Allowed Direction: {allowed_direction}")
    print(f"    - Confidence Threshold: {conf_thresh}")
    print(f"    - Counting Line Orientation: {line_orientation}")
    print(f"    - Output Video: {output}")
    print(f"    - Events Log: {logger.csv_path}")

    start_wall_time = time.perf_counter()

    with VideoReader(source) as reader:
        meta = reader.metadata
        stream_type = "Live Camera Stream" if reader.is_camera else "Video File"
        print(f"\n[*] Processing {stream_type}: {meta.width}x{meta.height} @ {meta.fps:.1f} FPS ({meta.total_frames} frames)")

        # Configure counting line: Horizontal for vertical traffic, Vertical for horizontal traffic
        norm_orient = line_orientation.upper()
        is_vertical_line = norm_orient == "VERTICAL" or (norm_orient == "AUTO" and allowed_direction in ["LEFT", "RIGHT"])
        if is_vertical_line:
            line_pos_px = int(meta.width * line_y)
            line_pixel_y = line_pos_px
            counter.counting_line = ((line_pos_px, 0), (line_pos_px, meta.height))
            line_axis_desc = f"x={line_pos_px}"
        else:
            line_pos_px = int(meta.height * line_y)
            line_pixel_y = line_pos_px
            counter.counting_line = ((0, line_pos_px), (meta.width, line_pos_px))
            line_axis_desc = f"y={line_pos_px}"

        with VideoWriterHelper(
            output_path=output,
            fps=meta.fps if frame_skip == 0 else meta.fps / (frame_skip + 1),
            frame_size=(meta.width, meta.height),
        ) as writer:
            frame_idx = 0
            for idx, frame in reader.read_frames(frame_skip=frame_skip):
                monitor.start_frame()

                # Stage 1: Detection & Multi-Object Tracking
                monitor.mark_preprocessed()
                active_tracks = tracker.update(frame, idx)
                if cfg.ignore_opposing_median:
                    active_tracks = [
                        t for t in active_tracks
                        if not (t.centroid[1] < 0.35 * meta.height and t.centroid[0] > 0.65 * meta.width)
                    ]
                monitor.mark_inference_complete()

                # Stage 2: Vehicle Counting
                new_crossings = counter.update(active_tracks, frame.shape[:2], idx)
                for ev in new_crossings:
                    logger.log_event(
                        event_type="LINE_CROSSING",
                        track_id=ev["track_id"],
                        class_name=ev["class_name"],
                        direction="CROSSING",
                        confidence=ev["confidence"],
                        frame_idx=idx,
                        details=f"Crossed counting line at {line_axis_desc}",
                    )
                    print(
                        f"  [COUNT] Frame {idx:03d} | {ev['class_name'].capitalize()} "
                        f"(ID: {ev['track_id']}) crossed line | Total: {counter.total_count}"
                    )

                # Stage 3: Direction & Wrong-Way Violation Detection
                new_violations = violation_detector.update(frame, active_tracks, idx)
                for v in new_violations:
                    logger.log_event(
                        event_type="WRONG_WAY_VIOLATION",
                        track_id=v["track_id"],
                        class_name=v["class_name"],
                        direction=v["direction"],
                        confidence=v["confidence"],
                        frame_idx=idx,
                        snapshot_path=v["snapshot_path"],
                        details=f"Heading {v['direction']} (Allowed: {v['allowed_direction']})",
                    )
                    print(
                        f"  [ALERT] Frame {idx:03d} | WRONG-WAY VIOLATION: {v['class_name'].capitalize()} "
                        f"(ID: {v['track_id']}) moving {v['direction']}! Snapshot: {v['snapshot_path']}"
                    )

                monitor.mark_tracking_complete()

                # Stage 4: Visual Overlay & Video Writing
                annotated = visualizer.render(
                    frame,
                    active_tracks,
                    counter=counter,
                    violation_detector=violation_detector,
                    monitor=monitor,
                )
                writer.write(annotated)

                monitor.end_frame(idx)
                frame_idx += 1

                if preview:
                    cv2.imshow("Real-Time Traffic Analytics", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:
                        print("\n[!] Stream preview terminated by user (pressed 'q' / ESC).")
                        break

            if preview:
                cv2.destroyAllWindows()

    total_wall_elapsed = time.perf_counter() - start_wall_time
    logger.save_json()
    perf_summary = monitor.get_summary()

    # Print Executive Summary Report
    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETE: SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Video Frames Processed: {frame_idx}")
    print(f"Total Wall-Clock Time:        {total_wall_elapsed:.2f} seconds")
    print(f"Average Processing Speed:     {perf_summary['avg_fps']} FPS")
    print(f"Average Inference Latency:    {perf_summary['avg_inference_ms']} ms")
    print(f"Average Total Frame Latency:  {perf_summary['avg_total_ms']} ms")
    print("-" * 75)
    print(f"Total Vehicles Counted:       {counter.total_count}")
    print(f"  - Cars:                     {counter.counts_by_class.get('car', 0)}")
    print(f"  - Trucks:                   {counter.counts_by_class.get('truck', 0)}")
    print(f"  - Buses:                    {counter.counts_by_class.get('bus', 0)}")
    print(f"  - Motorcycles:              {counter.counts_by_class.get('motorcycle', 0)}")
    print("-" * 75)
    print(f"Total Confirmed Violations:   {len(violation_detector.violations)}")
    for v in violation_detector.violations:
        print(f"  - Frame {v['frame_idx']} | ID {v['track_id']} ({v['class_name']}) | Snapshot: {v['snapshot_path']}")
    print("-" * 75)
    print(f"Generated Outputs:")
    print(f"  [x] Video:     {Path(output).resolve()}")
    print(f"  [x] CSV Log:   {logger.csv_path.resolve()}")
    print(f"  [x] JSON Log:  {logger.json_path.resolve()}")
    print(f"  [x] Snapshots: {violation_detector.snapshots_dir.resolve()}")
    print("=" * 75)

    summary_data = {
        "total_frames": frame_idx,
        "wall_clock_seconds": round(total_wall_elapsed, 2),
        "avg_fps": perf_summary.get("avg_fps", 0.0),
        "avg_inference_ms": perf_summary.get("avg_inference_ms", 0.0),
        "avg_total_ms": perf_summary.get("avg_total_ms", 0.0),
        "total_counted": counter.total_count,
        "counts_by_class": counter.counts_by_class,
        "total_violations": len(violation_detector.violations),
        "source": str(source),
        "allowed_direction": allowed_direction,
        "effective_allowed_direction": violation_detector.inferred_allowed_direction or allowed_direction,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = Path(cfg.logs_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Traffic & Vehicle Analytics System (Production CLI)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/input/traffic.mp4",
        help="Path to input video, webcam index (e.g. 0), or RTSP URL (default: data/input/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/processed_video.mp4",
        help="Path for processed output MP4 (default: outputs/videos/processed_video.mp4)",
    )
    parser.add_argument(
        "--allowed",
        type=str,
        default="AUTO",
        choices=["AUTO", "DOWN", "UP", "LEFT", "RIGHT"],
        help="Configured legal traffic flow direction: 'AUTO', 'DOWN', 'UP', 'LEFT', 'RIGHT' (default: AUTO)",
    )
    parser.add_argument(
        "--line-y",
        type=float,
        default=0.35,
        help="Normalized coordinate for virtual counting line [0.0 - 1.0] (default: 0.35)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="YOLO detection confidence threshold (default: 0.35)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Frame skipping factor: 0=process every frame, 1=skip alternate frames (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Inference resolution: 640, 1280, or 'auto' (default: 640)",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="AUTO",
        choices=["AUTO", "HORIZONTAL", "VERTICAL"],
        help="Counting line orientation: 'AUTO', 'HORIZONTAL', 'VERTICAL' (default: AUTO)",
    )
    parser.add_argument(
        "--reset-logs",
        action="store_true",
        help="Clear prior event logs before processing",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display OpenCV live window preview",
    )
    parser.add_argument(
        "--ignore-median",
        action="store_true",
        help="Filter out opposing carriageway median background traffic on divided highways",
    )
    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        output=args.output,
        allowed_direction=args.allowed,
        line_y=args.line_y,
        conf_thresh=args.conf,
        frame_skip=args.skip,
        preview=args.preview,
        reset_logs=args.reset_logs,
        imgsz=args.imgsz,
        line_orientation=args.orientation,
        ignore_opposing_median=args.ignore_median,
    )


if __name__ == "__main__":
    main()
