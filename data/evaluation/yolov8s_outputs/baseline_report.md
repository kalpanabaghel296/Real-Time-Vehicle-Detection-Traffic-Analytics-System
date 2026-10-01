# Baseline Benchmark Evaluation Report

**Generated:** 2026-10-01 23:14:24  
**Model:** `yolov8s.pt` (Confidence: 0.35, NMS IoU: 0.5)

## 1. Executive Summary Across 5 Diverse Clips

| Metric | Macro Average | Notes |
|---|---|---|
| **mAP@50** | **0.7681** | Primary detection accuracy at 0.50 IoU |
| **mAP@50-95** | **0.6832** | Stringent multi-threshold localization |
| **Precision** | **0.8207** | Vehicle detection precision |
| **Recall** | **0.8312** | Vehicle detection recall |
| **MOTA** | **0.6731** | Multiple Object Tracking Accuracy |
| **HOTA** | **0.7641** | Higher Order Tracking Accuracy |
| **IDF1** | **0.7941** | Identity F1 Preservation |
| **Total ID Switches** | **45** | Total identity switches across clips |
| **Counting Accuracy** | **92.7%** | 100 - Relative Error % |
| **Violation FAR/min** | **75.00** | False wrong-way alarms per minute |
| **Processing Speed** | **1.8 FPS** | End-to-end analytics throughput |

---

## 2. Per-Clip Detailed Breakdown

| Video Clip | Scenario | mAP50 | MOTA | HOTA | IDF1 | IDSW | GT Count | Pred Count | FAR/min | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| `day_highway.mp4` | Day Highway | 0.950 | 0.639 | 0.791 | 0.840 | 9 | 8 | 8 | 75.00 | 1.8 |
| `dense_traffic.mp4` | Dense Traffic | 0.654 | 0.883 | 0.917 | 0.939 | 0 | 5 | 5 | 60.00 | 1.5 |
| `night_traffic.mp4` | Night Traffic | 0.898 | 0.512 | 0.544 | 0.535 | 7 | 1 | 1 | 0.00 | 1.8 |
| `intersection.mp4` | Intersection | 0.656 | 0.743 | 0.809 | 0.847 | 23 | 7 | 6 | 180.00 | 1.7 |
| `wrong_way.mp4` | Wrong Way | 0.681 | 0.588 | 0.759 | 0.810 | 6 | 9 | 7 | 60.00 | 2.0 |

---
## 3. Class-Level AP@50 Breakdown

| Video Clip | Car AP50 | Truck AP50 | Bus AP50 | Motorcycle AP50 |
|---|---|---|---|---|
| `day_highway.mp4` | 0.909 | 0.892 | 1.000 | 1.000 |
| `dense_traffic.mp4` | 0.832 | 0.786 | 0.000 | 1.000 |
| `night_traffic.mp4` | 0.594 | 1.000 | 1.000 | 1.000 |
| `intersection.mp4` | 0.783 | 0.308 | 0.667 | 0.866 |
| `wrong_way.mp4` | 0.851 | 0.875 | 0.000 | 1.000 |

---
## 4. Size Breakdown (Small vs Medium vs Large)

| Video Clip | Small GT (P/R) | Med GT (P/R) | Large GT (P/R) |
|---|---|---|---|
| `day_highway.mp4` | 0 (0.00/0.00) | 849 (0.57/0.93) | 1230 (0.90/0.96) |
| `dense_traffic.mp4` | 294 (0.87/0.71) | 767 (0.92/0.96) | 106 (1.00/0.63) |
| `night_traffic.mp4` | 344 (0.80/0.61) | 351 (0.91/0.65) | 2 (0.00/0.00) |
| `intersection.mp4` | 121 (0.59/0.30) | 929 (0.90/0.85) | 230 (0.94/0.96) |
| `wrong_way.mp4` | 0 (0.00/0.00) | 856 (0.57/0.93) | 1338 (0.89/0.85) |
