"""
Unit Tests for Evaluation Subpackage (src/evaluation/)
======================================================
Tests:
- DetectionEvaluator (mAP50, mAP50-95, Precision, Recall, F1, size breakdown, PR curves)
- TrackingEvaluator (HOTA, MOTA, IDF1, ID switches, fragmentations, MT/ML)
- CountingEvaluator (Line-crossing precision, recall, duplicate detection)
- ViolationEvaluator (Wrong-way alert precision, recall, FAR, confirmation window sweep)
- ErrorAnalyzer (Forensic categorization of FP, FN, Misclass, Duplicate, Confusion Matrix)
"""

import numpy as np
import pytest
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.evaluation.tracking_evaluator import TrackingEvaluator
from src.evaluation.counting_evaluator import CountingEvaluator
from src.evaluation.violation_evaluator import ViolationEvaluator
from src.evaluation.error_analysis import ErrorAnalyzer


# -----------------------------------------------------------------------------
# 1. Detection Evaluator Tests
# -----------------------------------------------------------------------------

def test_detection_evaluator_perfect_match():
    evaluator = DetectionEvaluator(class_names=["car", "truck"])
    gt = [
        {"frame_idx": 0, "bbox": (100, 100, 200, 200), "class_name": "car"},
        {"frame_idx": 0, "bbox": (300, 300, 450, 450), "class_name": "truck"},
    ]
    preds = [
        {"frame_idx": 0, "bbox": (100, 100, 200, 200), "class_name": "car", "confidence": 0.95},
        {"frame_idx": 0, "bbox": (300, 300, 450, 450), "class_name": "truck", "confidence": 0.90},
    ]

    metrics = evaluator.evaluate(gt, preds, conf_threshold=0.35)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.map50 == 1.0
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 0


def test_detection_evaluator_size_breakdown():
    evaluator = DetectionEvaluator(class_names=["car"])
    gt = [
        # Small box (< 32x32): 20x20 = 400 px^2
        {"frame_idx": 0, "bbox": (50, 50, 70, 70), "class_name": "car"},
        # Medium box (32x32 to 96x96): 50x50 = 2500 px^2
        {"frame_idx": 0, "bbox": (100, 100, 150, 150), "class_name": "car"},
        # Large box (> 96x96): 120x120 = 14400 px^2
        {"frame_idx": 0, "bbox": (200, 200, 320, 320), "class_name": "car"},
    ]
    preds = [
        {"frame_idx": 0, "bbox": (50, 50, 70, 70), "class_name": "car", "confidence": 0.8},
        {"frame_idx": 0, "bbox": (100, 100, 150, 150), "class_name": "car", "confidence": 0.8},
        {"frame_idx": 0, "bbox": (200, 200, 320, 320), "class_name": "car", "confidence": 0.8},
    ]
    metrics = evaluator.evaluate(gt, preds)
    assert metrics.size_breakdown["small"]["gt"] == 1
    assert metrics.size_breakdown["medium"]["gt"] == 1
    assert metrics.size_breakdown["large"]["gt"] == 1
    assert metrics.size_breakdown["small"]["tp"] == 1


def test_detection_evaluator_confidence_curves():
    evaluator = DetectionEvaluator(class_names=["car"])
    gt = [{"frame_idx": 0, "bbox": (10, 10, 50, 50), "class_name": "car"}]
    preds = [
        {"frame_idx": 0, "bbox": (10, 10, 50, 50), "class_name": "car", "confidence": 0.9},
        {"frame_idx": 0, "bbox": (200, 200, 250, 250), "class_name": "car", "confidence": 0.3},
    ]
    curves = evaluator.generate_confidence_curves(gt, preds, steps=10)
    assert len(curves["confidences"]) == 10
    assert len(curves["precisions"]) == 10
    assert len(curves["recalls"]) == 10


# -----------------------------------------------------------------------------
# 2. Tracking Evaluator Tests
# -----------------------------------------------------------------------------

def test_tracking_evaluator_mota_and_id_switch():
    evaluator = TrackingEvaluator(iou_threshold=0.50)
    # Ground truth: 1 persistent car across frames 0, 1, 2 with GT ID 1
    gt = [
        {"frame_idx": 0, "track_id": 1, "bbox": (100, 100, 150, 150), "class_name": "car"},
        {"frame_idx": 1, "track_id": 1, "bbox": (105, 105, 155, 155), "class_name": "car"},
        {"frame_idx": 2, "track_id": 1, "bbox": (110, 110, 160, 160), "class_name": "car"},
    ]
    # Prediction: matches frame 0 (Pred ID 10), but switches to Pred ID 20 at frame 2
    preds = [
        {"frame_idx": 0, "track_id": 10, "bbox": (100, 100, 150, 150), "class_name": "car", "confidence": 0.9},
        {"frame_idx": 1, "track_id": 10, "bbox": (105, 105, 155, 155), "class_name": "car", "confidence": 0.9},
        {"frame_idx": 2, "track_id": 20, "bbox": (110, 110, 160, 160), "class_name": "car", "confidence": 0.9},
    ]

    metrics = evaluator.evaluate(gt, preds)
    assert metrics.id_switches == 1
    assert metrics.mostly_tracked == 1
    assert metrics.mostly_lost == 0
    assert metrics.det_precision == 1.0
    assert metrics.det_recall == 1.0


def test_tracking_evaluator_hota_and_idf1():
    evaluator = TrackingEvaluator(iou_threshold=0.50)
    gt = [
        {"frame_idx": 0, "track_id": 1, "bbox": (50, 50, 90, 90), "class_name": "car"},
        {"frame_idx": 1, "track_id": 1, "bbox": (55, 55, 95, 95), "class_name": "car"},
    ]
    preds = [
        {"frame_idx": 0, "track_id": 1, "bbox": (50, 50, 90, 90), "class_name": "car", "confidence": 0.95},
        {"frame_idx": 1, "track_id": 1, "bbox": (55, 55, 95, 95), "class_name": "car", "confidence": 0.95},
    ]
    metrics = evaluator.evaluate(gt, preds)
    assert metrics.hota == 1.0
    assert metrics.idf1 == 1.0
    assert metrics.mota == 1.0
    assert metrics.id_switches == 0


# -----------------------------------------------------------------------------
# 3. Counting Evaluator Tests
# -----------------------------------------------------------------------------

def test_counting_evaluator_metrics():
    evaluator = CountingEvaluator(temporal_tolerance_frames=10)
    gt_crossings = [
        {"frame_idx": 50, "track_id": 1, "class_name": "car"},
        {"frame_idx": 100, "track_id": 2, "class_name": "truck"},
        {"frame_idx": 150, "track_id": 3, "class_name": "car"},
    ]
    pred_crossings = [
        # True crossing
        {"frame_idx": 52, "track_id": 10, "class_name": "car"},
        # Duplicate crossing of GT 1
        {"frame_idx": 55, "track_id": 11, "class_name": "car"},
        # True crossing
        {"frame_idx": 98, "track_id": 12, "class_name": "truck"},
        # Missed GT 3 at frame 150
    ]

    metrics = evaluator.evaluate(gt_crossings, pred_crossings)
    assert metrics.gt_count == 3
    assert metrics.predicted_count == 3
    assert metrics.true_crossings == 2
    assert metrics.duplicate_crossings == 1
    assert metrics.missed_crossings == 1
    assert metrics.class_breakdown["truck"]["predicted_count"] == 1


# -----------------------------------------------------------------------------
# 4. Violation Evaluator Tests
# -----------------------------------------------------------------------------

def test_violation_evaluator_metrics():
    evaluator = ViolationEvaluator(fps=30.0, temporal_tolerance_frames=20)
    gt_violations = [
        {"start_frame": 100, "end_frame": 150, "track_id": 1, "direction": "UP"},
    ]
    pred_violations = [
        # True violation alert at frame 110 (delay = 10 frames)
        {"frame_idx": 110, "track_id": 1, "direction": "UP"},
        # False alarm at frame 400
        {"frame_idx": 400, "track_id": 99, "direction": "UP"},
    ]

    metrics = evaluator.evaluate(gt_violations, pred_violations, total_video_frames=600)
    assert metrics.true_violations == 1
    assert metrics.false_alarms == 1
    assert metrics.missed_violations == 0
    assert metrics.avg_detection_delay_frames == 10.0
    assert metrics.total_gt_violations == 1
    assert 4 in metrics.confirmation_window_sweep


# -----------------------------------------------------------------------------
# 5. Error Analyzer Tests
# -----------------------------------------------------------------------------

def test_error_analyzer_categorization(tmp_path):
    analyzer = ErrorAnalyzer(class_names=["car", "truck"], output_dir=str(tmp_path))
    gt = [
        {"frame_idx": 0, "bbox": (100, 100, 200, 200), "class_name": "car"},
        {"frame_idx": 0, "bbox": (300, 300, 400, 400), "class_name": "truck"},
    ]
    preds = [
        # Misclassification: Car predicted as Truck on GT 0
        {"frame_idx": 0, "bbox": (100, 100, 200, 200), "class_name": "truck", "confidence": 0.85},
        # False Positive: Spurious Car detection on background
        {"frame_idx": 0, "bbox": (500, 500, 600, 600), "class_name": "car", "confidence": 0.90},
        # GT 1 (truck at 300, 300) is missed -> False Negative
    ]

    report = analyzer.analyze(gt, preds, save_visual_crops=False)
    counts = report["summary_counts"]
    assert counts["MISCLASSIFICATION"] == 1
    assert counts["FALSE_POSITIVE"] == 1
    assert counts["FALSE_NEGATIVE"] == 1

    # Check confusion matrix
    cm_raw = np.array(report["confusion_matrix_raw"])
    assert cm_raw.shape == (3, 3)  # car, truck, background
    # Check that errors.json was exported
    assert (tmp_path / "errors.json").exists()
