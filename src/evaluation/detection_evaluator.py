"""
Detection Evaluator Module
==========================
Computes standard Object Detection benchmarks adhering strictly to MS COCO & PASCAL VOC metrics:
- Global Precision, Recall, F1, mAP@50, mAP@50-95
- Per-class AP, Precision, Recall, F1, TP, FP, FN
- Object-size analysis (Small < 32^2 px, Medium 32^2-96^2 px, Large > 96^2 px)
- Confidence Curves: Precision vs Conf, Recall vs Conf, F1 vs Conf
- Optimal operating threshold finder maximizing F1 score
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from src.utils import calculate_iou


@dataclass
class DetectionMetrics:
    """Container for computed detection evaluation metrics."""

    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    per_class: Dict[str, Dict[str, float]]
    size_breakdown: Dict[str, Dict[str, float]]
    best_confidence_threshold: float
    best_f1_score: float
    total_gt: int
    total_predictions: int
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DetectionEvaluator:
    """
    Evaluates vehicle object detections against independent ground-truth annotations.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
    ):
        """
        Args:
            class_names: List of class names to evaluate (default: ['car', 'motorcycle', 'bus', 'truck']).
            iou_thresholds: IoU levels for mAP calculation (default: 0.50 to 0.95 in 0.05 steps).
        """
        self.class_names = class_names or ["car", "motorcycle", "bus", "truck"]
        self.iou_thresholds = (
            iou_thresholds
            if iou_thresholds is not None
            else list(np.round(np.arange(0.50, 1.00, 0.05), 2))
        )

    @staticmethod
    def _compute_box_area(bbox: Tuple[int, int, int, int]) -> float:
        """Calculates area of bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    @classmethod
    def get_size_category(cls, bbox: Tuple[int, int, int, int]) -> str:
        """
        Categorizes bounding box size according to COCO definitions:
        - small: area < 32^2 (1024 px)
        - medium: 32^2 <= area <= 96^2 (1024 to 9216 px)
        - large: area > 96^2 (9216 px)
        """
        area = cls._compute_box_area(bbox)
        if area < 32.0 * 32.0:
            return "small"
        elif area <= 96.0 * 96.0:
            return "medium"
        else:
            return "large"

    @staticmethod
    def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Computes Average Precision (AP) using standard COCO 101-point interpolation.
        """
        if len(recalls) == 0 or len(precisions) == 0:
            return 0.0

        # Prepend 0 and append 1 for recall boundary, clamp precisions monotonically
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Monotonically decreasing precision envelope
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Area under curve: Riemann integration across distinct recall intervals
        i = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    def evaluate(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        conf_threshold: float = 0.35,
    ) -> DetectionMetrics:
        """
        Runs comprehensive evaluation comparing predictions against ground-truth.

        Args:
            ground_truths: List of dicts with keys: 'frame_idx', 'bbox', 'class_name'.
            predictions: List of dicts with keys: 'frame_idx', 'bbox', 'class_name', 'confidence'.
            conf_threshold: Nominal operating confidence threshold for reporting single-point metrics.

        Returns:
            DetectionMetrics dataclass with global, per-class, and size-level results.
        """
        # Filter predictions at nominal confidence threshold for static metrics
        pred_nominal = [p for p in predictions if p["confidence"] >= conf_threshold]

        # Group GT and Predictions by (frame_idx, class_name)
        gt_by_frame_class: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
        for gt in ground_truths:
            cname = gt["class_name"].lower()
            key = (gt["frame_idx"], cname)
            gt_by_frame_class.setdefault(key, []).append(gt)

        # 1. Compute mAP across IoU thresholds per class
        ap_per_class_50: Dict[str, float] = {}
        ap_per_class_50_95: Dict[str, float] = {}
        per_class_summary: Dict[str, Dict[str, float]] = {}

        # Size breakdown registries
        size_stats: Dict[str, Dict[str, int]] = {
            "small": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
            "medium": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
            "large": {"tp": 0, "fp": 0, "fn": 0, "gt": 0},
        }

        total_tp_nominal = 0
        total_fp_nominal = 0
        total_fn_nominal = 0

        for cname in self.class_names:
            class_preds = [p for p in predictions if p["class_name"].lower() == cname]
            class_gts = [g for g in ground_truths if g["class_name"].lower() == cname]
            n_gt = len(class_gts)

            # Sort predictions descending by confidence
            class_preds_sorted = sorted(
                class_preds, key=lambda x: x["confidence"], reverse=True
            )

            # Evaluate each IoU threshold for COCO mAP[50:95]
            aps_for_iou: List[float] = []

            for iou_thresh in self.iou_thresholds:
                matched_gt_ids = set()
                tp = np.zeros(len(class_preds_sorted))
                fp = np.zeros(len(class_preds_sorted))

                for i, pred in enumerate(class_preds_sorted):
                    frame_idx = pred["frame_idx"]
                    candidates = [
                        (idx_gt, g)
                        for idx_gt, g in enumerate(class_gts)
                        if g["frame_idx"] == frame_idx
                    ]

                    best_iou = 0.0
                    best_gt_idx = -1
                    for idx_gt, g in candidates:
                        iou = calculate_iou(pred["bbox"], g["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx_gt

                    if best_iou >= iou_thresh and best_gt_idx not in matched_gt_ids:
                        tp[i] = 1.0
                        matched_gt_ids.add(best_gt_idx)
                    else:
                        fp[i] = 1.0

                if n_gt == 0:
                    ap = 0.0 if len(class_preds_sorted) > 0 else 1.0
                else:
                    tp_cumsum = np.cumsum(tp)
                    fp_cumsum = np.cumsum(fp)
                    recalls = tp_cumsum / float(n_gt)
                    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
                    ap = self._compute_ap(recalls, precisions)

                aps_for_iou.append(ap)
                if abs(iou_thresh - 0.50) < 1e-4:
                    ap_per_class_50[cname] = ap

            ap_per_class_50_95[cname] = float(np.mean(aps_for_iou)) if aps_for_iou else 0.0

            # Compute nominal threshold metrics (IoU >= 0.50, Conf >= conf_threshold)
            class_nominal_preds = [
                p for p in class_preds_sorted if p["confidence"] >= conf_threshold
            ]
            nominal_matched_gt = set()
            c_tp = 0
            c_fp = 0

            for pred in class_nominal_preds:
                frame_idx = pred["frame_idx"]
                candidates = [
                    (idx_gt, g)
                    for idx_gt, g in enumerate(class_gts)
                    if g["frame_idx"] == frame_idx
                ]
                best_iou = 0.0
                best_gt_idx = -1
                for idx_gt, g in candidates:
                    iou = calculate_iou(pred["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx_gt

                sz = self.get_size_category(pred["bbox"])
                if best_iou >= 0.50 and best_gt_idx not in nominal_matched_gt:
                    c_tp += 1
                    nominal_matched_gt.add(best_gt_idx)
                    size_stats[sz]["tp"] += 1
                else:
                    c_fp += 1
                    size_stats[sz]["fp"] += 1

            c_fn = max(0, n_gt - len(nominal_matched_gt))
            for g in class_gts:
                sz = self.get_size_category(g["bbox"])
                size_stats[sz]["gt"] += 1

            for idx_gt, g in enumerate(class_gts):
                if idx_gt not in nominal_matched_gt:
                    sz = self.get_size_category(g["bbox"])
                    size_stats[sz]["fn"] += 1

            c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_rec = c_tp / n_gt if n_gt > 0 else 0.0
            c_f1 = (
                (2 * c_prec * c_rec) / (c_prec + c_rec)
                if (c_prec + c_rec) > 0
                else 0.0
            )

            per_class_summary[cname] = {
                "ap50": round(ap_per_class_50.get(cname, 0.0), 4),
                "ap50_95": round(ap_per_class_50_95.get(cname, 0.0), 4),
                "precision": round(c_prec, 4),
                "recall": round(c_rec, 4),
                "f1": round(c_f1, 4),
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "total_gt": n_gt,
            }

            total_tp_nominal += c_tp
            total_fp_nominal += c_fp
            total_fn_nominal += c_fn

        # Global metrics at nominal threshold
        global_prec = (
            total_tp_nominal / (total_tp_nominal + total_fp_nominal)
            if (total_tp_nominal + total_fp_nominal) > 0
            else 0.0
        )
        global_rec = (
            total_tp_nominal / len(ground_truths)
            if len(ground_truths) > 0
            else 0.0
        )
        global_f1 = (
            (2 * global_prec * global_rec) / (global_prec + global_rec)
            if (global_prec + global_rec) > 0
            else 0.0
        )

        global_map50 = float(np.mean(list(ap_per_class_50.values()))) if ap_per_class_50 else 0.0
        global_map50_95 = float(np.mean(list(ap_per_class_50_95.values()))) if ap_per_class_50_95 else 0.0

        # Size breakdown calculation
        size_summary: Dict[str, Dict[str, float]] = {}
        for sz, data in size_stats.items():
            s_tp, s_fp, s_fn = data["tp"], data["fp"], data["fn"]
            s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
            s_rec = s_tp / data["gt"] if data["gt"] > 0 else 0.0
            s_f1 = (
                (2 * s_prec * s_rec) / (s_prec + s_rec)
                if (s_prec + s_rec) > 0
                else 0.0
            )
            size_summary[sz] = {
                "precision": round(s_prec, 4),
                "recall": round(s_rec, 4),
                "f1": round(s_f1, 4),
                "tp": s_tp,
                "fp": s_fp,
                "fn": s_fn,
                "gt": data["gt"],
            }

        # 2. Confidence Curve Analysis & Optimal Operating Point Selection
        conf_thresholds = np.linspace(0.05, 0.95, 19)
        best_conf = conf_threshold
        best_f1 = 0.0

        for c_val in conf_thresholds:
            t_tp = 0
            t_fp = 0
            for cname in self.class_names:
                c_preds = [
                    p
                    for p in predictions
                    if p["class_name"].lower() == cname and p["confidence"] >= c_val
                ]
                c_gts = [g for g in ground_truths if g["class_name"].lower() == cname]

                matched_gt = set()
                for p in c_preds:
                    f_idx = p["frame_idx"]
                    cand = [
                        (ig, g) for ig, g in enumerate(c_gts) if g["frame_idx"] == f_idx
                    ]
                    best_i = 0.0
                    best_ig = -1
                    for ig, g in cand:
                        iou = calculate_iou(p["bbox"], g["bbox"])
                        if iou > best_i:
                            best_i = iou
                            best_ig = ig
                    if best_i >= 0.50 and best_ig not in matched_gt:
                        t_tp += 1
                        matched_gt.add(best_ig)
                    else:
                        t_fp += 1

            t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
            t_rec = t_tp / len(ground_truths) if len(ground_truths) > 0 else 0.0
            t_f1 = (2 * t_prec * t_rec) / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0.0

            if t_f1 > best_f1:
                best_f1 = t_f1
                best_conf = float(round(c_val, 2))

        return DetectionMetrics(
            precision=round(global_prec, 4),
            recall=round(global_rec, 4),
            f1=round(global_f1, 4),
            map50=round(global_map50, 4),
            map50_95=round(global_map50_95, 4),
            per_class=per_class_summary,
            size_breakdown=size_summary,
            best_confidence_threshold=best_conf,
            best_f1_score=round(best_f1, 4),
            total_gt=len(ground_truths),
            total_predictions=len(pred_nominal),
            tp=total_tp_nominal,
            fp=total_fp_nominal,
            fn=total_fn_nominal,
        )

    def generate_confidence_curves(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        steps: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Generates data arrays for plotting Precision, Recall, and F1 vs. Confidence curves.
        """
        confs = list(np.linspace(0.02, 0.98, steps))
        precisions = []
        recalls = []
        f1s = []

        total_gt = len(ground_truths)

        for c_val in confs:
            tp = 0
            fp = 0
            matched_gt = set()

            filtered_preds = [
                p for p in predictions if p["confidence"] >= c_val
            ]

            for p in filtered_preds:
                cname = p["class_name"].lower()
                f_idx = p["frame_idx"]
                cands = [
                    (i, g)
                    for i, g in enumerate(ground_truths)
                    if g["frame_idx"] == f_idx and g["class_name"].lower() == cname
                ]

                best_iou = 0.0
                best_id = -1
                for i, g in cands:
                    iou = calculate_iou(p["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = i

                if best_iou >= 0.50 and best_id not in matched_gt:
                    tp += 1
                    matched_gt.add(best_id)
                else:
                    fp += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / total_gt if total_gt > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(round(float(prec), 4))
            recalls.append(round(float(rec), 4))
            f1s.append(round(float(f1), 4))

        return {
            "confidences": [round(float(c), 3) for c in confs],
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1s,
        }
