"""
Traffic Analytics Evaluation Subpackage
=======================================
Provides rigorous, standardized evaluation metrics and error analysis:
- detection_evaluator: mAP@50, mAP@50-95, PR curves, per-class AP, size breakdown
- tracking_evaluator: HOTA, MOTA, IDF1, ID switches, track fragmentation, MT/ML
- counting_evaluator: Line crossing accuracy, GT vs Pred, Precision, Recall, F1
- violation_evaluator: Wrong-way detection accuracy, False Alarm Rate, latency
- error_analysis: Systematic categorization of detection and classification errors
"""

from src.evaluation.detection_evaluator import DetectionEvaluator, DetectionMetrics
from src.evaluation.tracking_evaluator import TrackingEvaluator, TrackingMetrics
from src.evaluation.counting_evaluator import CountingEvaluator, CountingMetrics
from src.evaluation.violation_evaluator import ViolationEvaluator, ViolationMetrics
from src.evaluation.error_analysis import ErrorAnalyzer, DetectionError

__all__ = [
    "DetectionEvaluator",
    "DetectionMetrics",
    "TrackingEvaluator",
    "TrackingMetrics",
    "CountingEvaluator",
    "CountingMetrics",
    "ViolationEvaluator",
    "ViolationMetrics",
    "ErrorAnalyzer",
    "DetectionError",
]
