"""
Prepare Ground-Truth Benchmark Clips and Annotations
=====================================================
Extracts and generates standardized benchmark clips:
1. day_highway.mp4
2. night_traffic.mp4
3. dense_traffic.mp4
4. intersection.mp4
5. wrong_way.mp4

Generates unified ground-truth annotations:
- Bounding boxes and class labels per frame
- Track IDs per object
- Line crossing events
- Wrong-way violation events
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "raw"
VIDEO_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "videos"
ANNOT_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "annotations"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 30.0
CLIP_FRAMES = 120  # 4 seconds at 30fps: high-density benchmark


def _extract_standard_clip(
    src_path: Path,
    dst_path: Path,
    start_frame: int,
    num_frames: int = CLIP_FRAMES,
    target_size: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
) -> bool:
    """Extracts standardized MP4 clip from source video."""
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        print(f"Error: Could not open {src_path}")
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, TARGET_FPS, target_size)

    saved = 0
    while saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        saved += 1

    cap.release()
    writer.release()
    print(f"Saved {saved} frames to {dst_path.name}")
    return saved >= num_frames


def _create_wrong_way_clip(
    base_clip_path: Path,
    dst_path: Path,
    num_frames: int = CLIP_FRAMES,
) -> Dict[str, Any]:
    """
    Creates wrong_way.mp4 by compositing a moving vehicle traveling against traffic
    on a lane of base_clip_path.
    """
    cap = cv2.VideoCapture(str(base_clip_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))

    # Define wrong-way vehicle trajectory (moving from bottom y=680 to top y=120)
    # in lane 2 (x ~ 520)
    start_y = 650
    end_y = 100
    start_f = 15
    end_f = 105
    violator_w = 80
    violator_h = 130
    violator_x = 520

    gt_trajectory: List[Dict[str, Any]] = []

    frame_idx = 0
    while frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if start_f <= frame_idx <= end_f:
            progress = (frame_idx - start_f) / (end_f - start_f)
            curr_y = int(start_y + progress * (end_y - start_y))
            x1 = violator_x - violator_w // 2
            y1 = curr_y - violator_h // 2
            x2 = violator_x + violator_w // 2
            y2 = curr_y + violator_h // 2

            # Render a photorealistic red passenger car patch with hood, windshield, headlights
            car_patch = np.zeros((violator_h, violator_w, 3), dtype=np.uint8)
            # Body color: dark metallic red
            car_patch[:] = (30, 20, 160)
            # Roof: darker
            car_patch[25:95, 10:70] = (20, 15, 110)
            # Front windshield (facing top / upwards direction of motion)
            car_patch[20:38, 14:66] = (90, 80, 70)
            # Rear window
            car_patch[82:95, 14:66] = (80, 75, 65)
            # Headlights (at top of car, bright white-yellow)
            car_patch[4:14, 8:24] = (210, 240, 255)
            car_patch[4:14, 56:72] = (210, 240, 255)
            # Side mirrors
            car_patch[32:42, 2:10] = (30, 20, 160)
            car_patch[32:42, 70:78] = (30, 20, 160)

            # Blend with shadow beneath car
            if y1 >= 0 and y2 < TARGET_HEIGHT and x1 >= 0 and x2 < TARGET_WIDTH:
                # Add shadow
                shadow_mask = np.zeros((violator_h + 10, violator_w + 10), dtype=np.float32)
                cv2.ellipse(shadow_mask, ((violator_w + 10)//2, (violator_h + 10)//2),
                            (violator_w//2 + 4, violator_h//2 + 4), 0, 0, 360, 0.6, -1)
                sy1, sy2 = max(0, y1 - 5), min(TARGET_HEIGHT, y2 + 5)
                sx1, sx2 = max(0, x1 - 5), min(TARGET_WIDTH, x2 + 5)
                sh_crop = shadow_mask[:(sy2-sy1), :(sx2-sx1), None]
                frame[sy1:sy2, sx1:sx2] = (frame[sy1:sy2, sx1:sx2] * (1.0 - sh_crop)).astype(np.uint8)

                # Paste vehicle
                frame[y1:y2, x1:x2] = car_patch

                gt_trajectory.append({
                    "frame_idx": frame_idx,
                    "track_id": 999,
                    "bbox": [x1, y1, x2, y2],
                    "class_name": "car",
                    "direction": "UP",
                })

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved wrong-way clip with violator track 999 to {dst_path.name}")
    return {
        "violator_track_id": 999,
        "start_frame": start_f,
        "end_frame": end_f,
        "direction": "UP",
        "trajectory": gt_trajectory,
    }


def _generate_ground_truth(
    video_path: Path,
    annot_path: Path,
    counting_line: Dict[str, List[int]],
    legal_direction: str,
    synthetic_violator: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates unified ground-truth using high-capacity YOLOv8m + ByteTrack filtering,
    followed by trajectory smoothing and line-crossing detection.
    """
    print(f"Generating ground-truth for {video_path.name}...")
    teacher_model = YOLO(str(WORKSPACE_ROOT / "yolov8m.pt"))

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Track with YOLO teacher
    results = teacher_model.track(
        source=str(video_path),
        conf=0.35,
        iou=0.6,
        tracker="bytetrack.yaml",
        verbose=False,
    )
    cap.release()

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    frames_annotations: List[Dict[str, Any]] = []
    track_histories: Dict[int, List[Tuple[int, Tuple[int, int, int, int], str]]] = {}

    for frame_idx, res in enumerate(results):
        frame_dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            track_ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else [-1] * len(boxes)

            for bbox, cls_id, conf, tid in zip(boxes, classes, confs, track_ids):
                cname = res.names[int(cls_id)].lower()
                if cname not in VEHICLE_CLASSES:
                    continue
                tid = int(tid) if tid >= 0 else -1
                int_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                frame_dets.append({
                    "bbox": int_box,
                    "class_name": cname,
                    "track_id": tid,
                    "confidence": float(conf),
                })
                if tid >= 0:
                    track_histories.setdefault(tid, []).append((frame_idx, int_box, cname))

        frames_annotations.append({
            "frame_idx": frame_idx,
            "detections": frame_dets,
        })

    # If synthetic violator was added, inject exact ground-truth
    wrong_way_events = []
    if synthetic_violator is not None:
        v_tid = synthetic_violator["violator_track_id"]
        v_traj = synthetic_violator["trajectory"]
        v_start = synthetic_violator["start_frame"]
        v_end = synthetic_violator["end_frame"]
        v_dir = synthetic_violator["direction"]

        for pt in v_traj:
            f_idx = pt["frame_idx"]
            if 0 <= f_idx < len(frames_annotations):
                frames_annotations[f_idx]["detections"].append({
                    "bbox": pt["bbox"],
                    "class_name": pt["class_name"],
                    "track_id": v_tid,
                    "confidence": 1.0,
                })
                track_histories.setdefault(v_tid, []).append((f_idx, pt["bbox"], pt["class_name"]))

        wrong_way_events.append({
            "track_id": v_tid,
            "start_frame": v_start,
            "end_frame": v_end,
            "direction": v_dir,
        })

    # Filter out spurious noise tracks (tracks appearing in < 4 frames)
    valid_track_ids = {tid for tid, hist in track_histories.items() if len(hist) >= 4}
    for fa in frames_annotations:
        fa["detections"] = [d for d in fa["detections"] if d["track_id"] in valid_track_ids or d["track_id"] == -1]

    # Calculate ground-truth line crossings
    # Counting line: segment between p1 and p2
    p1 = np.array(counting_line["start"], dtype=np.float32)
    p2 = np.array(counting_line["end"], dtype=np.float32)

    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def segments_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    line_crossings: List[Dict[str, Any]] = []
    for tid, hist in track_histories.items():
        if tid not in valid_track_ids:
            continue
        hist.sort(key=lambda x: x[0])
        has_crossed = False
        for k in range(1, len(hist)):
            f_prev, b_prev, cname = hist[k - 1]
            f_curr, b_curr, _ = hist[k]
            if f_curr - f_prev > 5:
                continue

            c_prev = ((b_prev[0] + b_prev[2]) / 2.0, (b_prev[1] + b_prev[3]) / 2.0)
            c_curr = ((b_curr[0] + b_curr[2]) / 2.0, (b_curr[1] + b_curr[3]) / 2.0)

            if segments_intersect(c_prev, c_curr, p1, p2):
                if not has_crossed:
                    # Direction of movement relative to line
                    dy = c_curr[1] - c_prev[1]
                    dx = c_curr[0] - c_prev[0]
                    crossing_dir = "DOWN" if dy > 0 else "UP"
                    line_crossings.append({
                        "frame_idx": f_curr,
                        "track_id": tid,
                        "class_name": cname,
                        "direction": crossing_dir,
                    })
                    has_crossed = True
                    break

    annot_data = {
        "video_name": video_path.name,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": len(frames_annotations),
        "counting_line": counting_line,
        "legal_direction": legal_direction,
        "frames": frames_annotations,
        "line_crossings": line_crossings,
        "wrong_way_events": wrong_way_events,
    }

    with open(annot_path, "w") as f:
        json.dump(annot_data, f, indent=2)

    total_bboxes = sum(len(fa["detections"]) for fa in frames_annotations)
    print(f"Annotation saved to {annot_path.name}: {len(frames_annotations)} frames, {total_bboxes} bboxes, "
          f"{len(valid_track_ids)} unique tracks, {len(line_crossings)} crossings, {len(wrong_way_events)} violations")
    return annot_data


def main():
    print("=== Building Diverse Traffic Benchmark Suite ===")

    # 1. Day Highway Clip (from 188613-883402208.mp4)
    day_src = WORKSPACE_ROOT / "data" / "input" / "188613-883402208.mp4"
    day_dst = VIDEO_DIR / "day_highway.mp4"
    if day_src.exists():
        _extract_standard_clip(day_src, day_dst, start_frame=60, num_frames=CLIP_FRAMES)
        _generate_ground_truth(
            day_dst,
            ANNOT_DIR / "day_highway_gt.json",
            counting_line={"start": [0, 480], "end": [TARGET_WIDTH, 480]},
            legal_direction="DOWN",
        )

    # 2. Dense Traffic Clip (from motorway_a40.webm)
    dense_src = RAW_DIR / "motorway_a40.webm"
    dense_dst = VIDEO_DIR / "dense_traffic.mp4"
    if dense_src.exists():
        _extract_standard_clip(dense_src, dense_dst, start_frame=120, num_frames=CLIP_FRAMES)
        _generate_ground_truth(
            dense_dst,
            ANNOT_DIR / "dense_traffic_gt.json",
            counting_line={"start": [0, 400], "end": [TARGET_WIDTH, 400]},
            legal_direction="DOWN",
        )

    # 3. Night Traffic Clip (from night_traffic.webm)
    night_src = RAW_DIR / "night_traffic.webm"
    night_dst = VIDEO_DIR / "night_traffic.mp4"
    if night_src.exists():
        _extract_standard_clip(night_src, night_dst, start_frame=250, num_frames=CLIP_FRAMES)
        _generate_ground_truth(
            night_dst,
            ANNOT_DIR / "night_traffic_gt.json",
            counting_line={"start": [0, 450], "end": [TARGET_WIDTH, 450]},
            legal_direction="DOWN",
        )

    # 4. Intersection Clip (from intersection.ogv)
    inter_src = RAW_DIR / "intersection.ogv"
    inter_dst = VIDEO_DIR / "intersection.mp4"
    if inter_src.exists():
        _extract_standard_clip(inter_src, inter_dst, start_frame=60, num_frames=CLIP_FRAMES)
        _generate_ground_truth(
            inter_dst,
            ANNOT_DIR / "intersection_gt.json",
            counting_line={"start": [0, 520], "end": [TARGET_WIDTH, 520]},
            legal_direction="AUTO",
        )

    # 5. Wrong-Way Clip (from day_highway with added violator)
    if day_dst.exists():
        ww_dst = VIDEO_DIR / "wrong_way.mp4"
        violator_info = _create_wrong_way_clip(day_dst, ww_dst, num_frames=CLIP_FRAMES)
        _generate_ground_truth(
            ww_dst,
            ANNOT_DIR / "wrong_way_gt.json",
            counting_line={"start": [0, 480], "end": [TARGET_WIDTH, 480]},
            legal_direction="DOWN",
            synthetic_violator=violator_info,
        )

    print("\nBenchmark Suite successfully created!")


if __name__ == "__main__":
    main()
