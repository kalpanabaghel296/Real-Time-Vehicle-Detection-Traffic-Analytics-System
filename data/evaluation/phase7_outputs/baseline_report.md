# Baseline Benchmark Evaluation Report

**Generated:** 2026-10-01 23:24:02  
**Model:** `yolov8n.pt` (Confidence: 0.4, NMS IoU: 0.45)

## 1. Executive Summary Across 5 Diverse Clips

| Metric | Macro Average | Notes |
|---|---|---|
| **mAP@50** | **0.6127** | Primary detection accuracy at 0.50 IoU |
| **mAP@50-95** | **0.5534** | Stringent multi-threshold localization |
| **Precision** | **0.8198** | Vehicle detection precision |
| **Recall** | **0.6134** | Vehicle detection recall |
| **MOTA** | **0.4460** | Multiple Object Tracking Accuracy |
| **HOTA** | **0.6326** | Higher Order Tracking Accuracy |
| **IDF1** | **0.6582** | Identity F1 Preservation |
| **Total ID Switches** | **21** | Total identity switches across clips |
| **Counting Accuracy** | **72.4%** | 100 - Relative Error % |
| **Violation FAR/min** | **60.00** | False wrong-way alarms per minute |
| **Processing Speed** | **6.9 FPS** | End-to-end analytics throughput |

---

## 2. Per-Clip Detailed Breakdown

| Video Clip | Scenario | mAP50 | MOTA | HOTA | IDF1 | IDSW | GT Count | Pred Count | FAR/min | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| `day_highway.mp4` | Day Highway | 0.790 | 0.371 | 0.678 | 0.729 | 6 | 8 | 9 | 45.00 | 7.6 |
| `dense_traffic.mp4` | Dense Traffic | 0.444 | 0.736 | 0.806 | 0.849 | 0 | 5 | 5 | 60.00 | 6.7 |
| `night_traffic.mp4` | Night Traffic | 0.787 | 0.146 | 0.278 | 0.237 | 3 | 1 | 0 | 0.00 | 6.8 |
| `intersection.mp4` | Intersection | 0.385 | 0.623 | 0.747 | 0.768 | 2 | 7 | 6 | 135.00 | 6.9 |
| `wrong_way.mp4` | Wrong Way | 0.658 | 0.353 | 0.654 | 0.708 | 10 | 9 | 8 | 60.00 | 6.6 |

---
## 3. Class-Level AP@50 Breakdown

| Video Clip | Car AP50 | Truck AP50 | Bus AP50 | Motorcycle AP50 |
|---|---|---|---|---|
| `day_highway.mp4` | 0.836 | 0.323 | 1.000 | 1.000 |
| `dense_traffic.mp4` | 0.709 | 0.000 | 0.065 | 1.000 |
| `night_traffic.mp4` | 0.149 | 1.000 | 1.000 | 1.000 |
| `intersection.mp4` | 0.624 | 0.276 | 0.000 | 0.640 |
| `wrong_way.mp4` | 0.783 | 0.314 | 0.536 | 1.000 |

---
## 4. Size Breakdown (Small vs Medium vs Large)

| Video Clip | Small GT (P/R) | Med GT (P/R) | Large GT (P/R) |
|---|---|---|---|
| `day_highway.mp4` | 0 (0.00/0.00) | 849 (0.43/0.77) | 1230 (0.92/0.87) |
| `dense_traffic.mp4` | 294 (0.80/0.27) | 767 (0.87/0.84) | 106 (1.00/0.61) |
| `night_traffic.mp4` | 344 (0.00/0.00) | 351 (1.00/0.30) | 2 (0.00/0.00) |
| `intersection.mp4` | 121 (0.00/0.00) | 929 (0.97/0.64) | 230 (0.92/0.93) |
| `wrong_way.mp4` | 0 (0.00/0.00) | 856 (0.43/0.78) | 1338 (0.91/0.79) |
