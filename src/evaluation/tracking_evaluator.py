"""
Tracking Evaluator Module
=========================
Computes standard Multi-Object Tracking (MOT) benchmarks adhering to HOTA, MOTA, and IDF1 standards:
- HOTA (Higher Order Tracking Accuracy, balancing detection and association)
- MOTA (Multiple Object Tracking Accuracy)
- IDF1 (Identification F1 score)
- ID Switches (IDSW) and Track Fragmentation
- Mostly Tracked (MT >= 80%), Partially Tracked (PT), Mostly Lost (ML < 20%)
- Lifespan and track continuity metrics
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional, Set
import numpy as np
from src.utils import calculate_iou


@dataclass
class TrackingMetrics:
    """Container for computed Multi-Object Tracking metrics."""

    hota: float
    mota: float
    idf1: float
    id_switches: int
    fragmentations: int
    mostly_tracked: int
    partially_tracked: int
    mostly_lost: int
    total_gt_tracks: int
    total_pred_tracks: int
    det_precision: float
    det_recall: float
    det_f1: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrackingEvaluator:
    """
    Evaluates tracking trajectories against ground-truth identity trajectories.
    """

    def __init__(self, iou_threshold: float = 0.50):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truth_tracks: List[Dict[str, Any]],
        predicted_tracks: List[Dict[str, Any]],
    ) -> TrackingMetrics:
        """
        Runs tracking evaluation across video frames.

        Args:
            ground_truth_tracks: List of dicts with:
                {'frame_idx': int, 'track_id': int, 'bbox': (x1, y1, x2, y2), 'class_name': str}
            predicted_tracks: List of dicts with:
                {'frame_idx': int, 'track_id': int, 'bbox': (x1, y1, x2, y2), 'class_name': str, 'confidence': float}

        Returns:
            TrackingMetrics dataclass containing HOTA, MOTA, IDF1, and IDSW stats.
        """
        if not ground_truth_tracks:
            return TrackingMetrics(
                hota=0.0,
                mota=0.0,
                idf1=0.0,
                id_switches=0,
                fragmentations=0,
                mostly_tracked=0,
                partially_tracked=0,
                mostly_lost=0,
                total_gt_tracks=0,
                total_pred_tracks=len(set(p["track_id"] for p in predicted_tracks)),
                det_precision=0.0,
                det_recall=0.0,
                det_f1=0.0,
            )

        # 1. Index annotations by frame_idx
        frames = sorted(
            list(
                set(
                    [g["frame_idx"] for g in ground_truth_tracks]
                    + [p["frame_idx"] for p in predicted_tracks]
                )
            )
        )

        gt_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for g in ground_truth_tracks:
            gt_by_frame.setdefault(g["frame_idx"], []).append(g)

        pred_by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for p in predicted_tracks:
            pred_by_frame.setdefault(p["frame_idx"], []).append(p)

        # Track ID lifetimes and memberships
        gt_track_frames: Dict[int, List[int]] = {}
        for g in ground_truth_tracks:
            gt_track_frames.setdefault(g["track_id"], []).append(g["frame_idx"])

        pred_track_frames: Dict[int, List[int]] = {}
        for p in predicted_tracks:
            pred_track_frames.setdefault(p["track_id"], []).append(p["frame_idx"])

        # Frame-by-frame bipartite matching for MOTA & ID switches
        total_gt_boxes = len(ground_truth_tracks)
        total_pred_boxes = len(predicted_tracks)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        id_switches = 0

        # Mapping of gt_track_id -> last_matched_pred_track_id
        gt_last_matched_pred: Dict[int, int] = {}
        # Mapping of (gt_id, pred_id) -> total intersection count (for IDF1 / HOTA)
        global_pair_matches: Dict[Tuple[int, int], int] = {}

        # Frame-level matching records: list of dicts with gt_id -> pred_id
        frame_matches: Dict[int, Dict[int, int]] = {}

        for f in frames:
            gts = gt_by_frame.get(f, [])
            preds = pred_by_frame.get(f, [])
            frame_matches[f] = {}

            if not gts and not preds:
                continue

            # Greedy IoU matching
            matched_gts: Set[int] = set()
            matched_preds: Set[int] = set()

            iou_matrix = np.zeros((len(gts), len(preds)))
            for i, g in enumerate(gts):
                for j, p in enumerate(preds):
                    iou_matrix[i, j] = calculate_iou(g["bbox"], p["bbox"])

            # Match highest IoU pairs first
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                gt_obj = gts[i]
                pred_obj = preds[j]

                gt_id = gt_obj["track_id"]
                pred_id = pred_obj["track_id"]

                matched_gts.add(i)
                matched_preds.add(j)
                frame_matches[f][gt_id] = pred_id

                pair_key = (gt_id, pred_id)
                global_pair_matches[pair_key] = global_pair_matches.get(pair_key, 0) + 1

                # Check for ID switch
                if gt_id in gt_last_matched_pred:
                    if gt_last_matched_pred[gt_id] != pred_id:
                        id_switches += 1
                gt_last_matched_pred[gt_id] = pred_id

                # Zero out row and column to prevent reuse
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

            frame_tp = len(matched_gts)
            frame_fp = len(preds) - frame_tp
            frame_fn = len(gts) - frame_tp

            total_tp += frame_tp
            total_fp += frame_fp
            total_fn += frame_fn

        # MOTA calculation: 1 - (FN + FP + IDSW) / GT
        mota = (
            1.0 - (float(total_fn + total_fp + id_switches) / float(total_gt_boxes))
            if total_gt_boxes > 0
            else 0.0
        )

        # Detection accuracy from tracker
        det_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        det_rec = total_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0
        det_f1 = (2 * det_prec * det_rec) / (det_prec + det_rec) if (det_prec + det_rec) > 0 else 0.0

        # IDF1 Calculation
        # Identify best matching predicted track for each GT track
        # IDTP is maximized over bipartite matching between GT tracks and Pred tracks
        idtp = 0
        assigned_preds: Set[int] = set()

        # Sort pairs by match count descending
        sorted_pairs = sorted(
            global_pair_matches.items(), key=lambda x: x[1], reverse=True
        )
        assigned_gts: Set[int] = set()

        for (gt_id, pred_id), count in sorted_pairs:
            if gt_id not in assigned_gts and pred_id not in assigned_preds:
                idtp += count
                assigned_gts.add(gt_id)
                assigned_preds.add(pred_id)

        idfp = total_pred_boxes - idtp
        idfn = total_gt_boxes - idtp
        idf1 = (2.0 * idtp) / (2.0 * idtp + idfp + idfn) if (2.0 * idtp + idfp + idfn) > 0 else 0.0

        # HOTA Calculation: HOTA = sqrt(DetA * AssA)
        det_a = total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) > 0 else 0.0

        # Association Accuracy (AssA)
        ass_scores = []
        for f, matches in frame_matches.items():
            for gt_id, pred_id in matches.items():
                # For this matched pair c = (gt_id, pred_id):
                # TPA: number of frames where BOTH gt_id and pred_id match
                tpa = global_pair_matches.get((gt_id, pred_id), 0)
                # Entire length of gt_id and pred_id
                gt_len = len(gt_track_frames.get(gt_id, []))
                pred_len = len(pred_track_frames.get(pred_id, []))
                fpa = pred_len - tpa
                fna = gt_len - tpa

                ass_score = float(tpa) / float(tpa + fpa + fna) if (tpa + fpa + fna) > 0 else 0.0
                ass_scores.append(ass_score)

        ass_a = float(np.mean(ass_scores)) if ass_scores else 0.0
        hota = float(np.sqrt(det_a * ass_a))

        # Track quality diagnostics: MT, PT, ML, Fragmentations
        mostly_tracked = 0
        partially_tracked = 0
        mostly_lost = 0
        fragmentations = 0

        for gt_id, gt_f_list in gt_track_frames.items():
            gt_len = len(gt_f_list)
            # Count how many frames this gt was matched to ANY prediction
            matched_count = 0
            match_flags = []
            for f in gt_f_list:
                is_m = gt_id in frame_matches.get(f, {})
                match_flags.append(is_m)
                if is_m:
                    matched_count += 1

            track_ratio = matched_count / gt_len if gt_len > 0 else 0.0
            if track_ratio >= 0.80:
                mostly_tracked += 1
            elif track_ratio <= 0.20:
                mostly_lost += 1
            else:
                partially_tracked += 1

            # Count transitions from matched -> unmatched -> matched
            for k in range(1, len(match_flags)):
                if not match_flags[k - 1] and match_flags[k]:
                    fragmentations += 1

        return TrackingMetrics(
            hota=round(hota, 4),
            mota=round(mota, 4),
            idf1=round(idf1, 4),
            id_switches=id_switches,
            fragmentations=fragmentations,
            mostly_tracked=mostly_tracked,
            partially_tracked=partially_tracked,
            mostly_lost=mostly_lost,
            total_gt_tracks=len(gt_track_frames),
            total_pred_tracks=len(pred_track_frames),
            det_precision=round(det_prec, 4),
            det_recall=round(det_rec, 4),
            det_f1=round(det_f1, 4),
        )
