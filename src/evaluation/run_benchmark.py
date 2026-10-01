"""
Benchmark Evaluation Runner
============================
Executes the traffic analytics pipeline over standardized benchmark clips,
computes detection, tracking, counting, and violation metrics, and outputs
comprehensive evaluation results and error analysis.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.config import TrafficConfig
from src.detector import YOLOVehicleDetector
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.violation import WrongWayDetector
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.evaluation.tracking_evaluator import TrackingEvaluator
from src.evaluation.counting_evaluator import CountingEvaluator
from src.evaluation.violation_evaluator import ViolationEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer


def run_benchmark_on_video(
    video_path: Path,
    annot_path: Path,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.50,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    Executes the full pipeline on one video and evaluates all metrics.
    """
    print(f"\n=======================================================")
    print(f"Evaluating: {video_path.name}")
    print(f"Model: {model_name} | Conf: {conf_threshold} | IoU: {iou_threshold}")
    print(f"=======================================================")

    with open(annot_path, "r") as f:
        gt_data = json.load(f)

    # 1. Initialize Pipeline Modules
    cfg = TrafficConfig()
    cfg.confidence_threshold = conf_threshold
    cfg.iou_threshold = iou_threshold
    cfg.model_name = model_name

    counting_line = gt_data["counting_line"]
    line_start = tuple(counting_line["start"])
    line_end = tuple(counting_line["end"])
    legal_dir = gt_data["legal_direction"]

    detector = YOLOVehicleDetector(config=cfg)
    tracker = VehicleTracker(config=cfg)
    counter = VehicleCounter(config=cfg, counting_line=(line_start, line_end))
    violation_detector = WrongWayDetector(
        config=cfg,
        allowed_direction=legal_dir,
        snapshots_dir=str(output_dir / "snapshots" / video_path.stem),
    )

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    pred_detections: List[Dict[str, Any]] = []
    pred_tracks: List[Dict[str, Any]] = []
    pred_crossings: List[Dict[str, Any]] = []
    pred_violations: List[Dict[str, Any]] = []

    frame_images: List[np.ndarray] = []
    inference_times: List[float] = []
    tracking_times: List[float] = []
    total_frame_times: List[float] = []

    frame_idx = 0
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame image copy for error analyzer visual crops (keep first 120 frames in memory)
        frame_images.append(frame.copy())

        # Detection phase
        td0 = time.perf_counter()
        detections = detector.detect(frame)
        td1 = time.perf_counter()
        inference_times.append(td1 - td0)

        for det in detections:
            pred_detections.append({
                "frame_idx": frame_idx,
                "bbox": list(det.bbox),
                "class_name": det.class_name,
                "confidence": float(det.confidence),
            })

        # Tracking phase
        tt0 = time.perf_counter()
        active_tracks = tracker.update(frame, frame_idx=frame_idx)
        tt1 = time.perf_counter()
        tracking_times.append(tt1 - tt0)

        for trk in active_tracks:
            pred_tracks.append({
                "frame_idx": frame_idx,
                "track_id": trk.track_id,
                "bbox": list(trk.bbox),
                "class_name": trk.class_name,
                "confidence": float(trk.confidence),
            })

        # Counting phase
        frame_shape = (frame.shape[0], frame.shape[1])
        newly_counted = counter.update(active_tracks, frame_shape=frame_shape, frame_idx=frame_idx)
        for c_ev in newly_counted:
            pred_crossings.append({
                "frame_idx": frame_idx,
                "track_id": c_ev["track_id"],
                "class_name": c_ev["class_name"],
                "direction": "DOWN" if c_ev["displacement"][1] > 0 else "UP",
            })

        # Violation phase
        new_alerts = violation_detector.update(frame, active_tracks, frame_idx=frame_idx)
        for alert in new_alerts:
            pred_violations.append({
                "frame_idx": alert["frame_idx"],
                "track_id": alert["track_id"],
                "class_name": alert["class_name"],
                "direction": alert["direction"],
                "confidence": alert.get("confidence", 1.0),
            })

        t_end = time.perf_counter()
        total_frame_times.append(t_end - t0)
        frame_idx += 1

    cap.release()

    # 2. Extract Ground Truth Components
    gt_detections: List[Dict[str, Any]] = []
    gt_tracks: List[Dict[str, Any]] = []
    for f in gt_data["frames"]:
        f_idx = f["frame_idx"]
        for d in f["detections"]:
            gt_detections.append({
                "frame_idx": f_idx,
                "bbox": d["bbox"],
                "class_name": d["class_name"],
            })
            if d.get("track_id", -1) >= 0:
                gt_tracks.append({
                    "frame_idx": f_idx,
                    "track_id": d["track_id"],
                    "bbox": d["bbox"],
                    "class_name": d["class_name"],
                })

    gt_crossings = gt_data.get("line_crossings", [])
    gt_violations = gt_data.get("wrong_way_events", [])

    # 3. Compute Metrics with Evaluators
    det_eval = DetectionEvaluator(iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    det_metrics = det_eval.evaluate(gt_detections, pred_detections, conf_threshold=conf_threshold)

    trk_eval = TrackingEvaluator(iou_threshold=0.50)
    trk_metrics = trk_eval.evaluate(gt_tracks, pred_tracks)

    cnt_eval = CountingEvaluator(temporal_tolerance_frames=15)
    cnt_metrics = cnt_eval.evaluate(gt_crossings, pred_crossings)

    vio_eval = ViolationEvaluator(fps=fps, temporal_tolerance_frames=30)
    vio_metrics = vio_eval.evaluate(gt_violations, pred_violations, total_video_frames=total_frames)

    # 4. Detailed Error Analysis
    err_analyzer = ErrorAnalyzer(
        iou_threshold=0.50,
        output_dir=str(output_dir / "errors" / video_path.stem),
    )
    pred_nominal = [p for p in pred_detections if p["confidence"] >= conf_threshold]
    frames_dict = {i: img for i, img in enumerate(frame_images)}
    error_summary = err_analyzer.analyze(
        ground_truths=gt_detections,
        predictions=pred_nominal,
        frames_dict=frames_dict,
        save_visual_crops=True,
    )

    # 5. Speed / Latency Performance
    avg_inf_ms = float(np.mean(inference_times) * 1000.0) if inference_times else 0.0
    avg_trk_ms = float(np.mean(tracking_times) * 1000.0) if tracking_times else 0.0
    avg_fps = float(len(total_frame_times) / np.sum(total_frame_times)) if total_frame_times else 0.0

    results = {
        "video": video_path.name,
        "model": model_name,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "frames_processed": frame_idx,
        "latency": {
            "avg_fps": round(avg_fps, 2),
            "inference_ms": round(avg_inf_ms, 2),
            "tracking_ms": round(avg_trk_ms, 2),
        },
        "detection": {
            "mAP50": round(det_metrics.map50, 4),
            "mAP50_95": round(det_metrics.map50_95, 4),
            "precision": round(det_metrics.precision, 4),
            "recall": round(det_metrics.recall, 4),
            "f1": round(det_metrics.f1, 4),
            "per_class_ap50": {k: round(v.get("ap50", 0.0), 4) for k, v in det_metrics.per_class.items()},
            "size_breakdown": det_metrics.size_breakdown,
        },
        "tracking": {
            "mota": round(trk_metrics.mota, 4),
            "hota": round(trk_metrics.hota, 4),
            "idf1": round(trk_metrics.idf1, 4),
            "id_switches": trk_metrics.id_switches,
            "fragmentations": trk_metrics.fragmentations,
            "mostly_tracked": trk_metrics.mostly_tracked,
            "mostly_lost": trk_metrics.mostly_lost,
            "total_gt_tracks": trk_metrics.total_gt_tracks,
            "total_pred_tracks": trk_metrics.total_pred_tracks,
        },
        "counting": {
            "gt_total": cnt_metrics.gt_count,
            "pred_total": cnt_metrics.predicted_count,
            "mae": round(float(cnt_metrics.absolute_error), 2),
            "relative_error_pct": round(float(cnt_metrics.relative_error) * 100.0, 2),
            "precision": round(cnt_metrics.precision, 4),
            "recall": round(cnt_metrics.recall, 4),
            "f1": round(cnt_metrics.f1, 4),
            "duplicate_counts": cnt_metrics.duplicate_crossings,
            "missed_counts": cnt_metrics.missed_crossings,
        },
        "violation": {
            "precision": round(vio_metrics.precision, 4),
            "recall": round(vio_metrics.recall, 4),
            "f1": round(vio_metrics.f1, 4),
            "far_per_min": round(vio_metrics.false_alarm_rate_per_min, 4),
            "total_predicted_alerts": vio_metrics.total_predicted_alerts,
            "true_positive_alerts": vio_metrics.true_violations,
            "false_alarms": vio_metrics.false_alarms,
        },
        "error_counts": error_summary["summary_counts"],
    }

    # Print video summary
    print(f"Results for {video_path.name}:")
    print(f"  Speed: {avg_fps:.1f} FPS | Inf: {avg_inf_ms:.1f}ms | Trk: {avg_trk_ms:.1f}ms")
    print(f"  Detection: mAP50={det_metrics.map50:.3f} | mAP50-95={det_metrics.map50_95:.3f} | P={det_metrics.precision:.3f} | R={det_metrics.recall:.3f}")
    print(f"  Tracking:  MOTA={trk_metrics.mota:.3f} | HOTA={trk_metrics.hota:.3f} | IDF1={trk_metrics.idf1:.3f} | IDSW={trk_metrics.id_switches}")
    print(f"  Counting:  GT={cnt_metrics.gt_count} | Pred={cnt_metrics.predicted_count} | Acc={100.0 - cnt_metrics.relative_error * 100:.1f}%")
    print(f"  Violation: P={vio_metrics.precision:.3f} | R={vio_metrics.recall:.3f} | FAR={vio_metrics.false_alarm_rate_per_min:.2f}/min")
    return results


def main():
    parser = argparse.ArgumentParser(description="Traffic Analytics Benchmark Runner")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--output-dir", type=str, default="data/evaluation/baseline_outputs", help="Output directory")
    args = parser.parse_args()

    video_dir = WORKSPACE_ROOT / "data" / "evaluation" / "videos"
    annot_dir = WORKSPACE_ROOT / "data" / "evaluation" / "annotations"
    out_dir = WORKSPACE_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        "day_highway.mp4",
        "dense_traffic.mp4",
        "night_traffic.mp4",
        "intersection.mp4",
        "wrong_way.mp4",
    ]

    all_results = []
    for vid_name in videos:
        v_path = video_dir / vid_name
        a_path = annot_dir / f"{v_path.stem}_gt.json"
        if not v_path.exists() or not a_path.exists():
            print(f"Warning: {vid_name} or its annotation missing, skipping.")
            continue
        res = run_benchmark_on_video(
            video_path=v_path,
            annot_path=a_path,
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            output_dir=out_dir,
        )
        all_results.append(res)

    # Compute Macro-Averages Across Benchmark
    macro_map50 = float(np.mean([r["detection"]["mAP50"] for r in all_results]))
    macro_map50_95 = float(np.mean([r["detection"]["mAP50_95"] for r in all_results]))
    macro_det_p = float(np.mean([r["detection"]["precision"] for r in all_results]))
    macro_det_r = float(np.mean([r["detection"]["recall"] for r in all_results]))
    macro_mota = float(np.mean([r["tracking"]["mota"] for r in all_results]))
    macro_hota = float(np.mean([r["tracking"]["hota"] for r in all_results]))
    macro_idf1 = float(np.mean([r["tracking"]["idf1"] for r in all_results]))
    total_idsw = int(np.sum([r["tracking"]["id_switches"] for r in all_results]))
    total_gt_cross = int(np.sum([r["counting"]["gt_total"] for r in all_results]))
    total_pred_cross = int(np.sum([r["counting"]["pred_total"] for r in all_results]))
    avg_counting_acc = float(np.mean([100.0 - r["counting"]["relative_error_pct"] for r in all_results]))
    avg_far_min = float(np.mean([r["violation"]["far_per_min"] for r in all_results]))
    avg_fps = float(np.mean([r["latency"]["avg_fps"] for r in all_results]))

    summary_payload = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "macro_metrics": {
            "mAP50": round(macro_map50, 4),
            "mAP50_95": round(macro_map50_95, 4),
            "precision": round(macro_det_p, 4),
            "recall": round(macro_det_r, 4),
            "mota": round(macro_mota, 4),
            "hota": round(macro_hota, 4),
            "idf1": round(macro_idf1, 4),
            "total_id_switches": total_idsw,
            "total_gt_crossings": total_gt_cross,
            "total_pred_crossings": total_pred_cross,
            "avg_counting_accuracy_pct": round(avg_counting_acc, 2),
            "avg_far_per_min": round(avg_far_min, 4),
            "avg_fps": round(avg_fps, 2),
        },
        "per_video_results": all_results,
    }

    # Save Results JSON
    json_path = out_dir / "baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nFull benchmark results saved to: {json_path}")

    # Generate Markdown Summary Report
    md_report = f"""# Baseline Benchmark Evaluation Report

**Generated:** {summary_payload['benchmark_timestamp']}  
**Model:** `{args.model}` (Confidence: {args.conf}, NMS IoU: {args.iou})

## 1. Executive Summary Across 5 Diverse Clips

| Metric | Macro Average | Notes |
|---|---|---|
| **mAP@50** | **{macro_map50:.4f}** | Primary detection accuracy at 0.50 IoU |
| **mAP@50-95** | **{macro_map50_95:.4f}** | Stringent multi-threshold localization |
| **Precision** | **{macro_det_p:.4f}** | Vehicle detection precision |
| **Recall** | **{macro_det_r:.4f}** | Vehicle detection recall |
| **MOTA** | **{macro_mota:.4f}** | Multiple Object Tracking Accuracy |
| **HOTA** | **{macro_hota:.4f}** | Higher Order Tracking Accuracy |
| **IDF1** | **{macro_idf1:.4f}** | Identity F1 Preservation |
| **Total ID Switches** | **{total_idsw}** | Total identity switches across clips |
| **Counting Accuracy** | **{avg_counting_acc:.1f}%** | 100 - Relative Error % |
| **Violation FAR/min** | **{avg_far_min:.2f}** | False wrong-way alarms per minute |
| **Processing Speed** | **{avg_fps:.1f} FPS** | End-to-end analytics throughput |

---

## 2. Per-Clip Detailed Breakdown

| Video Clip | Scenario | mAP50 | MOTA | HOTA | IDF1 | IDSW | GT Count | Pred Count | FAR/min | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for r in all_results:
        md_report += (
            f"| `{r['video']}` | {r['video'].replace('.mp4', '').replace('_', ' ').title()} | "
            f"{r['detection']['mAP50']:.3f} | {r['tracking']['mota']:.3f} | {r['tracking']['hota']:.3f} | "
            f"{r['tracking']['idf1']:.3f} | {r['tracking']['id_switches']} | {r['counting']['gt_total']} | "
            f"{r['counting']['pred_total']} | {r['violation']['far_per_min']:.2f} | {r['latency']['avg_fps']:.1f} |\n"
        )

    md_report += "\n---\n## 3. Class-Level AP@50 Breakdown\n\n| Video Clip | Car AP50 | Truck AP50 | Bus AP50 | Motorcycle AP50 |\n|---|---|---|---|---|\n"
    for r in all_results:
        pcap = r["detection"]["per_class_ap50"]
        md_report += (
            f"| `{r['video']}` | {pcap.get('car', 0.0):.3f} | {pcap.get('truck', 0.0):.3f} | "
            f"{pcap.get('bus', 0.0):.3f} | {pcap.get('motorcycle', 0.0):.3f} |\n"
        )

    md_report += "\n---\n## 4. Size Breakdown (Small vs Medium vs Large)\n\n| Video Clip | Small GT (P/R) | Med GT (P/R) | Large GT (P/R) |\n|---|---|---|---|\n"
    for r in all_results:
        sb = r["detection"]["size_breakdown"]
        s_str = f"{sb['small']['gt']} ({sb['small']['precision']:.2f}/{sb['small']['recall']:.2f})"
        m_str = f"{sb['medium']['gt']} ({sb['medium']['precision']:.2f}/{sb['medium']['recall']:.2f})"
        l_str = f"{sb['large']['gt']} ({sb['large']['precision']:.2f}/{sb['large']['recall']:.2f})"
        md_report += f"| `{r['video']}` | {s_str} | {m_str} | {l_str} |\n"

    report_path = out_dir / "baseline_report.md"
    with open(report_path, "w") as f:
        f.write(md_report)
    print(f"Summary Markdown report written to: {report_path}")


if __name__ == "__main__":
    main()
