"""
Phase 5: Dataset Audit & Hard-Case Mining
========================================
Mines hard cases from evaluation error analysis:
- High-confidence false positives -> background negative samples (guardrails, bridge shadows)
- Low-confidence true positives -> under-represented / nocturnal vehicle samples
- Class boundary misclassifications (truck vs bus, car vs van)
- Validates bounding box health (degenerate, inverted, out-of-bounds checks)
- Generates dataset health and hard-case mining audit report
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_OUT_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "baseline_outputs"
ANNOT_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "annotations"
VIDEOS_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "videos"
HARD_CASE_DIR = WORKSPACE_ROOT / "data" / "evaluation" / "hard_cases"

HARD_CASE_DIR.mkdir(parents=True, exist_ok=True)
NEGATIVES_DIR = HARD_CASE_DIR / "negatives_background"
LOW_CONF_DIR = HARD_CASE_DIR / "low_confidence_positives"
MISCLASS_DIR = HARD_CASE_DIR / "boundary_misclassifications"

for d in [NEGATIVES_DIR, LOW_CONF_DIR, MISCLASS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def audit_annotation_health() -> Dict[str, Any]:
    """
    Audits all ground-truth annotation files for geometry validity,
    coordinate clamping, and degenerate bounding boxes.
    """
    audit_results = {
        "files_checked": 0,
        "total_annotations": 0,
        "degenerate_boxes": 0,
        "out_of_bounds_boxes": 0,
        "inverted_boxes": 0,
        "class_histogram": {},
    }

    annot_files = list(ANNOT_DIR.glob("*_gt.json"))
    for af in annot_files:
        audit_results["files_checked"] += 1
        with open(af, "r") as f:
            data = json.load(f)

        w, h = data.get("width", 1280), data.get("height", 720)
        for frame in data.get("frames", []):
            for det in frame.get("detections", []):
                audit_results["total_annotations"] += 1
                cname = det["class_name"]
                audit_results["class_histogram"][cname] = (
                    audit_results["class_histogram"].get(cname, 0) + 1
                )

                x1, y1, x2, y2 = det["bbox"]
                if x2 <= x1 or y2 <= y1:
                    audit_results["inverted_boxes"] += 1
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 2 or box_h < 2:
                    audit_results["degenerate_boxes"] += 1
                if x1 < 0 or y1 < 0 or x2 > w + 10 or y2 > h + 10:
                    audit_results["out_of_bounds_boxes"] += 1

    return audit_results


def mine_hard_cases(max_samples: int = 50) -> Dict[str, Any]:
    """
    Mines hard negative, low-confidence positive, and misclassification cases
    from recorded baseline errors across the benchmark clips.
    """
    errors_dir = BASE_OUT_DIR / "errors"
    mined_summary = {
        "mined_negatives": 0,
        "mined_low_conf": 0,
        "mined_misclassifications": 0,
        "samples": [],
    }

    caps = {}

    for cdir in errors_dir.iterdir():
        if not cdir.is_dir():
            continue
        err_file = cdir / "errors.json"
        if not err_file.exists():
            continue
        with open(err_file, "r") as f:
            err_data = json.load(f)

        vname = cdir.name + ".mp4"
        vpath = VIDEOS_DIR / vname
        if not vpath.exists():
            continue
        if vname not in caps:
            caps[vname] = cv2.VideoCapture(str(vpath))
        cap = caps[vname]

        for err in err_data.get("errors", []):
            f_idx = err["frame_idx"]
            cat = err["error_type"]
            conf = err.get("confidence", 0.0)
            bbox = err["bbox"]

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            fh, fw = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            pad = 20
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(fw, x2 + pad)
            cy2 = min(fh, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            sample_info = None

            # 1. High-confidence False Positives (conf >= 0.50 on background)
            if cat == "FALSE_POSITIVE" and conf >= 0.50 and mined_summary["mined_negatives"] < max_samples:
                fname = f"neg_{mined_summary['mined_negatives']+1:03d}_{cdir.name}_f{f_idx}_conf{int(conf*100)}.jpg"
                cv2.imwrite(str(NEGATIVES_DIR / fname), crop)
                mined_summary["mined_negatives"] += 1
                sample_info = {
                    "type": "NEGATIVE_BACKGROUND",
                    "filename": fname,
                    "clip": cdir.name,
                    "frame": f_idx,
                    "confidence": conf,
                    "predicted_class": err.get("predicted_class"),
                }

            # 2. Low-confidence Positives or False Negatives (missed vehicles)
            elif cat == "FALSE_NEGATIVE" and mined_summary["mined_low_conf"] < max_samples:
                fname = f"low_conf_{mined_summary['mined_low_conf']+1:03d}_{cdir.name}_f{f_idx}.jpg"
                cv2.imwrite(str(LOW_CONF_DIR / fname), crop)
                mined_summary["mined_low_conf"] += 1
                sample_info = {
                    "type": "HARD_POSITIVE_FN",
                    "filename": fname,
                    "clip": cdir.name,
                    "frame": f_idx,
                    "gt_class": err.get("ground_truth_class"),
                }

            # 3. Boundary Misclassifications (truck vs bus, car vs truck)
            elif cat == "MISCLASSIFICATION" and mined_summary["mined_misclassifications"] < max_samples:
                fname = f"misclass_{mined_summary['mined_misclassifications']+1:03d}_{cdir.name}_f{f_idx}.jpg"
                cv2.imwrite(str(MISCLASS_DIR / fname), crop)
                mined_summary["mined_misclassifications"] += 1
                sample_info = {
                    "type": "BOUNDARY_MISCLASS",
                    "filename": fname,
                    "clip": cdir.name,
                    "frame": f_idx,
                    "gt_class": err.get("ground_truth_class"),
                    "pred_class": err.get("predicted_class"),
                    "confidence": conf,
                }

            if sample_info:
                mined_summary["samples"].append(sample_info)

    for c in caps.values():
        c.release()

    return mined_summary


def main():
    print("=== Phase 5: Dataset Health Audit & Hard-Case Mining ===")

    # 1. Annotation Health Audit
    health = audit_annotation_health()
    print(f"Annotations checked: {health['total_annotations']} across {health['files_checked']} files")
    print(f"Degenerate: {health['degenerate_boxes']} | Out of bounds: {health['out_of_bounds_boxes']} | Inverted: {health['inverted_boxes']}")
    print(f"Class histogram: {health['class_histogram']}")

    # 2. Hard Case Mining
    mined = mine_hard_cases(max_samples=30)
    print(f"Mined negative background samples: {mined['mined_negatives']}")
    print(f"Mined hard positive/FN samples: {mined['mined_low_conf']}")
    print(f"Mined boundary misclassifications: {mined['mined_misclassifications']}")

    # Save summary
    out_payload = {
        "annotation_audit": health,
        "hard_case_mining": mined,
    }
    with open(HARD_CASE_DIR / "dataset_audit_report.json", "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Dataset Audit & Hard-Case Report saved to: {HARD_CASE_DIR / 'dataset_audit_report.json'}")


if __name__ == "__main__":
    main()
