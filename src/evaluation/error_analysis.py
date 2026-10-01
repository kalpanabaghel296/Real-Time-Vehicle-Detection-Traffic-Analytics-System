"""
Error Analysis & Forensic Categorization Module
==============================================
Systematically analyzes, categorizes, and audits all detection and tracking errors:
- False Positive (FP): Prediction on background / non-vehicle
- False Negative (FN): Missed ground-truth vehicle
- Misclassification: Overlapping box with incorrect class (Car -> Truck, Truck -> Bus)
- Localization Error: Correct class with poor boundary alignment (IoU < 0.50)
- Duplicate Detection: Multiple bounding boxes predicting a single vehicle
- Generates structured errors.json report
- Exports annotated visual error crops into categorized folders
- Computes raw and normalized Confusion Matrices
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Set
import cv2
import numpy as np
from src.utils import calculate_iou


@dataclass
class DetectionError:
    """Detailed record of a single categorized error instance."""

    frame_idx: int
    error_type: str  # 'FALSE_POSITIVE', 'FALSE_NEGATIVE', 'MISCLASSIFICATION', 'LOCALIZATION', 'DUPLICATE'
    ground_truth_class: str
    predicted_class: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    iou: float
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ErrorAnalyzer:
    """
    Performs forensic audit and categorization of computer vision detection errors.
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.50,
        output_dir: str = "outputs/evaluation",
    ):
        self.class_names = class_names or ["car", "motorcycle", "bus", "truck"]
        self.iou_threshold = iou_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        frames_dict: Optional[Dict[int, np.ndarray]] = None,
        save_visual_crops: bool = True,
        max_crops_per_category: int = 25,
    ) -> Dict[str, Any]:
        """
        Categorizes all errors across predictions and ground truths.

        Args:
            ground_truths: List of GT dicts ('frame_idx', 'bbox', 'class_name').
            predictions: List of Pred dicts ('frame_idx', 'bbox', 'class_name', 'confidence').
            frames_dict: Optional mapping of frame_idx -> raw BGR frame array for saving visual crops.
            save_visual_crops: Whether to export annotated image crops.
            max_crops_per_category: Limit visual image exports per folder to avoid disk bloat.

        Returns:
            Dictionary with categorized error lists, confusion matrix, and summary counts.
        """
        all_frames = sorted(
            list(set([g["frame_idx"] for g in ground_truths] + [p["frame_idx"] for p in predictions]))
        )

        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for g in ground_truths:
            gt_by_frame.setdefault(g["frame_idx"], []).append(g)

        pred_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for p in predictions:
            pred_by_frame.setdefault(p["frame_idx"], []).append(p)

        errors: List[DetectionError] = []
        error_counts = {
            "FALSE_POSITIVE": 0,
            "FALSE_NEGATIVE": 0,
            "MISCLASSIFICATION": 0,
            "LOCALIZATION": 0,
            "DUPLICATE": 0,
        }

        # Confusion Matrix Registry: class_labels + 'background'
        labels = self.class_names + ["background"]
        label_idx = {name: i for i, name in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=int)

        crops_saved = {k: 0 for k in error_counts.keys()}

        # Create visual crop destination subdirectories
        if save_visual_crops:
            for cat in ["false_positives", "false_negatives", "misclassifications", "localization_errors", "duplicates"]:
                (self.output_dir / cat).mkdir(parents=True, exist_ok=True)

        for f in all_frames:
            gts = gt_by_frame.get(f, [])
            preds = pred_by_frame.get(f, [])
            raw_frame = frames_dict.get(f) if frames_dict else None

            # Matrix of IoUs between all predictions and all GTs in this frame
            matched_gts: Set[int] = set()
            matched_preds: Set[int] = set()

            # Sort predictions descending by confidence
            sorted_pred_indices = sorted(
                range(len(preds)), key=lambda i: preds[i]["confidence"], reverse=True
            )

            for p_idx in sorted_pred_indices:
                pred = preds[p_idx]
                p_cls = pred["class_name"].lower()
                p_box = pred["bbox"]
                p_conf = pred["confidence"]

                # Find candidate GT with highest IoU
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gt in enumerate(gts):
                    iou = calculate_iou(p_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_gt_idx != -1 and best_iou >= self.iou_threshold:
                    gt = gts[best_gt_idx]
                    gt_cls = gt["class_name"].lower()

                    if best_gt_idx in matched_gts:
                        # DUPLICATE DETECTION: Another prediction already claimed this GT
                        err = DetectionError(
                            frame_idx=f,
                            error_type="DUPLICATE",
                            ground_truth_class=gt_cls,
                            predicted_class=p_cls,
                            confidence=p_conf,
                            bbox=p_box,
                            iou=best_iou,
                            severity="MEDIUM",
                            details=f"Duplicate prediction on GT #{best_gt_idx} ({gt_cls})",
                        )
                        errors.append(err)
                        error_counts["DUPLICATE"] += 1
                        self._save_crop(raw_frame, p_box, f, "duplicates", crops_saved, max_crops_per_category, f"conf_{p_conf:.2f}")

                    elif p_cls != gt_cls:
                        # MISCLASSIFICATION: Overlaps GT vehicle but wrong class
                        err = DetectionError(
                            frame_idx=f,
                            error_type="MISCLASSIFICATION",
                            ground_truth_class=gt_cls,
                            predicted_class=p_cls,
                            confidence=p_conf,
                            bbox=p_box,
                            iou=best_iou,
                            severity="HIGH" if p_conf >= 0.70 else "MEDIUM",
                            details=f"Predicted {p_cls} instead of actual {gt_cls}",
                        )
                        errors.append(err)
                        error_counts["MISCLASSIFICATION"] += 1
                        matched_gts.add(best_gt_idx)
                        matched_preds.add(p_idx)
                        r_idx = label_idx.get(gt_cls, label_idx["background"])
                        c_idx = label_idx.get(p_cls, label_idx["background"])
                        cm[r_idx, c_idx] += 1
                        self._save_crop(raw_frame, p_box, f, "misclassifications", crops_saved, max_crops_per_category, f"{gt_cls}_as_{p_cls}")

                    else:
                        # TRUE POSITIVE (Correct detection)
                        matched_gts.add(best_gt_idx)
                        matched_preds.add(p_idx)
                        r_idx = label_idx.get(gt_cls, label_idx["background"])
                        c_idx = label_idx.get(p_cls, label_idx["background"])
                        cm[r_idx, c_idx] += 1

                elif best_gt_idx != -1 and 0.15 <= best_iou < self.iou_threshold:
                    gt = gts[best_gt_idx]
                    gt_cls = gt["class_name"].lower()
                    if p_cls == gt_cls:
                        # LOCALIZATION ERROR: Correct class but boundary is offset
                        err = DetectionError(
                            frame_idx=f,
                            error_type="LOCALIZATION",
                            ground_truth_class=gt_cls,
                            predicted_class=p_cls,
                            confidence=p_conf,
                            bbox=p_box,
                            iou=best_iou,
                            severity="MEDIUM",
                            details=f"IoU {best_iou:.2f} is below matching threshold {self.iou_threshold}",
                        )
                        errors.append(err)
                        error_counts["LOCALIZATION"] += 1
                        self._save_crop(raw_frame, p_box, f, "localization_errors", crops_saved, max_crops_per_category, f"iou_{best_iou:.2f}")
                    else:
                        # Low IoU + wrong class = False Positive
                        err = DetectionError(
                            frame_idx=f,
                            error_type="FALSE_POSITIVE",
                            ground_truth_class="background",
                            predicted_class=p_cls,
                            confidence=p_conf,
                            bbox=p_box,
                            iou=best_iou,
                            severity="HIGH" if p_conf >= 0.70 else "LOW",
                            details=f"Spurious detection on background with conf={p_conf:.2f}",
                        )
                        errors.append(err)
                        error_counts["FALSE_POSITIVE"] += 1
                        r_idx = label_idx["background"]
                        c_idx = label_idx.get(p_cls, label_idx["background"])
                        cm[r_idx, c_idx] += 1
                        self._save_crop(raw_frame, p_box, f, "false_positives", crops_saved, max_crops_per_category, f"fp_{p_cls}_{p_conf:.2f}")

                else:
                    # FALSE POSITIVE: Prediction on background
                    err = DetectionError(
                        frame_idx=f,
                        error_type="FALSE_POSITIVE",
                        ground_truth_class="background",
                        predicted_class=p_cls,
                        confidence=p_conf,
                        bbox=p_box,
                        iou=best_iou,
                        severity="HIGH" if p_conf >= 0.70 else "LOW",
                        details=f"Spurious detection on background with conf={p_conf:.2f}",
                    )
                    errors.append(err)
                    error_counts["FALSE_POSITIVE"] += 1
                    r_idx = label_idx["background"]
                    c_idx = label_idx.get(p_cls, label_idx["background"])
                    cm[r_idx, c_idx] += 1
                    self._save_crop(raw_frame, p_box, f, "false_positives", crops_saved, max_crops_per_category, f"fp_{p_cls}_{p_conf:.2f}")

            # Check for FALSE NEGATIVES (Unmatched ground-truth vehicles)
            for g_idx, gt in enumerate(gts):
                if g_idx not in matched_gts:
                    gt_cls = gt["class_name"].lower()
                    err = DetectionError(
                        frame_idx=f,
                        error_type="FALSE_NEGATIVE",
                        ground_truth_class=gt_cls,
                        predicted_class="background",
                        confidence=0.0,
                        bbox=gt["bbox"],
                        iou=0.0,
                        severity="HIGH",
                        details=f"Missed {gt_cls} vehicle",
                    )
                    errors.append(err)
                    error_counts["FALSE_NEGATIVE"] += 1
                    r_idx = label_idx.get(gt_cls, label_idx["background"])
                    c_idx = label_idx["background"]
                    cm[r_idx, c_idx] += 1
                    self._save_crop(raw_frame, gt["bbox"], f, "false_negatives", crops_saved, max_crops_per_category, f"fn_{gt_cls}")

        # Compute normalized confusion matrix
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.zeros_like(cm, dtype=float)
        np.divide(cm.astype(float), row_sums, out=cm_normalized, where=row_sums != 0)

        # Export structured error report to JSON
        report_data = {
            "summary_counts": error_counts,
            "total_errors": len(errors),
            "confusion_matrix_labels": labels,
            "confusion_matrix_raw": cm.tolist(),
            "confusion_matrix_normalized": np.round(cm_normalized, 4).tolist(),
            "errors": [e.to_dict() for e in errors],
        }

        report_file = self.output_dir / "errors.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        return report_data

    @staticmethod
    def _save_crop(
        frame: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        subdir_name: str,
        crops_saved: Dict[str, int],
        max_crops: int,
        tag: str,
    ) -> None:
        """Saves a cropped region of interest containing the error."""
        if frame is None or frame.size == 0:
            return

        cat_key = subdir_name.upper().rstrip("S")
        if crops_saved.get(cat_key, 0) >= max_crops:
            return

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # Add 15px context margin
        pad = 15
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(w, x2 + pad)
        cy2 = min(h, y2 + pad)

        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2].copy()
            # Draw highlight box on crop
            cv2.rectangle(
                crop,
                (x1 - cx1, y1 - cy1),
                (x2 - cx1, y2 - cy1),
                (0, 0, 255),
                2,
            )
            crops_saved[cat_key] = crops_saved.get(cat_key, 0) + 1
            filename = f"f{frame_idx:04d}_{tag}_{crops_saved[cat_key]:02d}.jpg"
            save_path = Path("outputs/evaluation") / subdir_name / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), crop)
