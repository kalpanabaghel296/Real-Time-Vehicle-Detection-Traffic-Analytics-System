# Baseline Benchmark Evaluation Report

**Generated:** 2026-10-01 16:13:05  
**Model:** `yolov8n.pt` (Confidence: 0.35, NMS IoU: 0.5)

## 1. Executive Summary Across 5 Diverse Clips

| Metric | Macro Average | Notes |
|---|---|---|
| **mAP@50** | **0.6439** | Primary detection accuracy at 0.50 IoU |
| **mAP@50-95** | **0.5731** | Multi-threshold localization quality |
| **Precision** | **0.7888** | Vehicle detection precision |
| **Recall** | **0.6690** | Vehicle detection recall |
| **MOTA** | **0.4311** | Multiple Object Tracking Accuracy |
| **HOTA** | **0.6397** | Higher Order Tracking Accuracy |
| **IDF1** | **0.6630** | Identity F1 Preservation |
| **Total ID Switches** | **29** | Total identity switches across clips |
| **Counting Accuracy** | **74.6%** | 100 - Relative Error % |
| **Violation FAR/min** | **60.00** | False wrong-way alarms per minute |
| **Processing Speed** | **5.3 FPS** | End-to-end analytics throughput |

---

## 2. Per-Clip Detailed Breakdown

| Video Clip | Scenario | mAP50 | MOTA | HOTA | IDF1 | IDSW | GT Count | Pred Count | FAR/min | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
| `day_highway.mp4` | Day Highway | 0.819 | 0.288 | 0.669 | 0.712 | 7 | 8 | 9 | 60.00 | 4.2 |
| `dense_traffic.mp4` | Dense Traffic | 0.498 | 0.769 | 0.832 | 0.865 | 3 | 5 | 5 | 60.00 | 5.0 |
| `night_traffic.mp4` | Night Traffic | 0.799 | 0.181 | 0.296 | 0.276 | 3 | 1 | 0 | 0.00 | 5.5 |
| `intersection.mp4` | Intersection | 0.406 | 0.648 | 0.764 | 0.786 | 4 | 7 | 6 | 135.00 | 6.0 |
| `wrong_way.mp4` | Wrong Way | 0.698 | 0.269 | 0.638 | 0.677 | 12 | 9 | 9 | 45.00 | 5.8 |

---
## 3. Class-Level AP@50 Breakdown

| Video Clip | Car AP50 | Truck AP50 | Bus AP50 | Motorcycle AP50 |
|---|---|---|---|---|
| `day_highway.mp4` | 0.865 | 0.411 | 1.000 | 1.000 |
| `dense_traffic.mp4` | 0.769 | 0.078 | 0.146 | 1.000 |
| `night_traffic.mp4` | 0.195 | 1.000 | 1.000 | 1.000 |
| `intersection.mp4` | 0.679 | 0.257 | 0.000 | 0.686 |
| `wrong_way.mp4` | 0.809 | 0.405 | 0.576 | 1.000 |

---
## 4. Size Breakdown (Small vs Medium vs Large)

| Video Clip | Small GT (P/R) | Med GT (P/R) | Large GT (P/R) |
|---|---|---|---|
| `day_highway.mp4` | 0 (0.00/0.00) | 849 (0.38/0.86) | 1230 (0.90/0.90) |
| `dense_traffic.mp4` | 294 (0.76/0.43) | 767 (0.86/0.88) | 106 (1.00/0.62) |
| `night_traffic.mp4` | 344 (1.00/0.00) | 351 (1.00/0.38) | 2 (0.00/0.00) |
| `intersection.mp4` | 121 (0.67/0.10) | 929 (0.94/0.70) | 230 (0.91/0.93) |
| `wrong_way.mp4` | 0 (0.00/0.00) | 856 (0.38/0.87) | 1338 (0.89/0.82) |

---
## 5. Detection Error Breakdown

| Video Clip | False Positives | False Negatives | Misclassifications | Localization Errs | Duplicates |
|---|---|---|---|---|---|
| `day_highway.mp4` | 1104 | 221 | 42 | 154 | 38 |
| `dense_traffic.mp4` | 35 | 262 | 46 | 65 | 9 |
| `night_traffic.mp4` | 0 | 561 | 0 | 0 | 0 |
| `intersection.mp4` | 18 | 389 | 23 | 22 | 12 |
| `wrong_way.mp4` | 1122 | 322 | 53 | 143 | 47 |
