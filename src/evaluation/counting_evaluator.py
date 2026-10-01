"""
Counting Evaluator Module
=========================
Evaluates virtual line-crossing vehicle counts against independent ground truth:
- GT Count vs. Predicted Count
- Absolute Error and Relative Error (%)
- Crossing Precision, Recall, and F1 score
- Explicit tracking of Missed Crossings, Duplicate Crossings, and Spurious Crossings
- Per-class volume comparison
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Set


@dataclass
class CountingMetrics:
    """Container for computed vehicle counting metrics."""

    gt_count: int
    predicted_count: int
    absolute_error: int
    relative_error: float
    precision: float
    recall: float
    f1: float
    true_crossings: int
    missed_crossings: int
    duplicate_crossings: int
    false_crossings: int
    class_breakdown: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CountingEvaluator:
    """
    Evaluates line-crossing event accuracy against annotated crossing logs.
    """

    def __init__(self, temporal_tolerance_frames: int = 15):
        """
        Args:
            temporal_tolerance_frames: Maximum frame difference to match a GT crossing with a predicted crossing.
        """
        self.temporal_tolerance_frames = temporal_tolerance_frames

    def evaluate(
        self,
        ground_truth_crossings: List[Dict[str, Any]],
        predicted_crossings: List[Dict[str, Any]],
    ) -> CountingMetrics:
        """
        Compares predicted line-crossing events with ground-truth crossing events.

        Args:
            ground_truth_crossings: List of dicts with:
                {'frame_idx': int, 'track_id': int, 'class_name': str}
            predicted_crossings: List of dicts with:
                {'frame_idx': int, 'track_id': int, 'class_name': str}

        Returns:
            CountingMetrics dataclass.
        """
        gt_count = len(ground_truth_crossings)
        pred_count = len(predicted_crossings)
        abs_err = abs(gt_count - pred_count)
        rel_err = float(abs_err) / float(gt_count) if gt_count > 0 else 0.0

        # Sort crossings chronologically by frame
        gt_sorted = sorted(ground_truth_crossings, key=lambda x: x["frame_idx"])
        pred_sorted = sorted(predicted_crossings, key=lambda x: x["frame_idx"])

        matched_gts: Set[int] = set()
        matched_preds: Set[int] = set()
        duplicate_preds: Set[int] = set()
        gt_matched_by_pred_ids: Dict[int, List[int]] = {}

        for p_idx, pred in enumerate(pred_sorted):
            p_frame = pred["frame_idx"]
            p_class = pred["class_name"].lower()

            best_gt_idx = -1
            min_frame_diff = float("inf")

            for g_idx, gt in enumerate(gt_sorted):
                frame_diff = abs(gt["frame_idx"] - p_frame)
                if frame_diff <= self.temporal_tolerance_frames:
                    if gt["class_name"].lower() == p_class:
                        if frame_diff < min_frame_diff:
                            min_frame_diff = frame_diff
                            best_gt_idx = g_idx

            if best_gt_idx != -1:
                if best_gt_idx not in matched_gts:
                    matched_gts.add(best_gt_idx)
                    matched_preds.add(p_idx)
                    gt_matched_by_pred_ids[best_gt_idx] = [p_idx]
                else:
                    # Vehicle already matched to this GT crossing -> duplicate count
                    duplicate_preds.add(p_idx)
                    gt_matched_by_pred_ids[best_gt_idx].append(p_idx)

        tp = len(matched_gts)
        fn = gt_count - tp
        fp_dup = len(duplicate_preds)
        fp_spurious = pred_count - len(matched_preds) - fp_dup
        fp_total = fp_dup + fp_spurious

        prec = float(tp) / float(tp + fp_total) if (tp + fp_total) > 0 else (1.0 if pred_count == 0 else 0.0)
        rec = float(tp) / float(gt_count) if gt_count > 0 else 1.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        # Per-class counting breakdown
        all_classes = set(
            [g["class_name"].lower() for g in ground_truth_crossings]
            + [p["class_name"].lower() for p in predicted_crossings]
        )
        class_breakdown: Dict[str, Dict[str, Any]] = {}
        for cname in all_classes:
            c_gt = len([g for g in ground_truth_crossings if g["class_name"].lower() == cname])
            c_pred = len([p for p in predicted_crossings if p["class_name"].lower() == cname])
            c_abs = abs(c_gt - c_pred)
            c_rel = float(c_abs) / float(c_gt) if c_gt > 0 else 0.0
            class_breakdown[cname] = {
                "gt_count": c_gt,
                "predicted_count": c_pred,
                "absolute_error": c_abs,
                "relative_error": round(c_rel, 4),
            }

        return CountingMetrics(
            gt_count=gt_count,
            predicted_count=pred_count,
            absolute_error=abs_err,
            relative_error=round(rel_err, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1=round(f1, 4),
            true_crossings=tp,
            missed_crossings=fn,
            duplicate_crossings=fp_dup,
            false_crossings=fp_spurious,
            class_breakdown=class_breakdown,
        )
