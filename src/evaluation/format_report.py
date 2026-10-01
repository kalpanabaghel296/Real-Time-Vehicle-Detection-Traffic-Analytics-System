import json
from pathlib import Path

out_dir = Path("data/evaluation/baseline_outputs")
json_path = out_dir / "baseline_results.json"
if not json_path.exists():
    print(f"Error: {json_path} does not exist")
    exit(1)

with open(json_path, "r") as f:
    summary_payload = json.load(f)

all_results = summary_payload["per_video_results"]
macro = summary_payload["macro_metrics"]

md_report = f"""# Baseline Benchmark Evaluation Report

**Generated:** {summary_payload['benchmark_timestamp']}  
**Model:** `{summary_payload['model']}` (Confidence: {summary_payload['conf_threshold']}, NMS IoU: {summary_payload['iou_threshold']})

## 1. Executive Summary Across 5 Diverse Clips

| Metric | Macro Average | Notes |
|---|---|---|
| **mAP@50** | **{macro['mAP50']:.4f}** | Primary detection accuracy at 0.50 IoU |
| **mAP@50-95** | **{macro['mAP50_95']:.4f}** | Multi-threshold localization quality |
| **Precision** | **{macro['precision']:.4f}** | Vehicle detection precision |
| **Recall** | **{macro['recall']:.4f}** | Vehicle detection recall |
| **MOTA** | **{macro['mota']:.4f}** | Multiple Object Tracking Accuracy |
| **HOTA** | **{macro['hota']:.4f}** | Higher Order Tracking Accuracy |
| **IDF1** | **{macro['idf1']:.4f}** | Identity F1 Preservation |
| **Total ID Switches** | **{macro['total_id_switches']}** | Total identity switches across clips |
| **Counting Accuracy** | **{macro['avg_counting_accuracy_pct']:.1f}%** | 100 - Relative Error % |
| **Violation FAR/min** | **{macro['avg_far_per_min']:.2f}** | False wrong-way alarms per minute |
| **Processing Speed** | **{macro['avg_fps']:.1f} FPS** | End-to-end analytics throughput |

---

## 2. Per-Clip Detailed Breakdown

| Video Clip | Scenario | mAP50 | MOTA | HOTA | IDF1 | IDSW | GT Count | Pred Count | FAR/min | FPS |
|---|---|---|---|---|---|---|---|---|---|---|
"""

for r in all_results:
    md_report += (
        f"| `{r['video']}` | {r['video'].replace('.mp4', '').replace('_', ' ').title()} | "
        f"{r['detection']['mAP50']:.3f} | {r['tracking']['mota']:.3f} | {r['tracking']['hota']:.3f} | "
        f"{r['tracking']['idf1']:.3f} | {r['tracking']['id_switches']} | {r['counting']['gt_total']} | "
        f"{r['counting']['pred_total']} | {r['violation']['far_per_min']:.2f} | {r['latency']['avg_fps']:.1f} |\n"
    )

md_report += "\n---\n## 3. Class-Level AP@50 Breakdown\n\n| Video Clip | Car AP50 | Truck AP50 | Bus AP50 | Motorcycle AP50 |\n|---|---|---|---|---|\n"
for r in all_results:
    pcap = r["detection"]["per_class_ap50"]
    md_report += (
        f"| `{r['video']}` | {pcap.get('car', 0.0):.3f} | {pcap.get('truck', 0.0):.3f} | "
        f"{pcap.get('bus', 0.0):.3f} | {pcap.get('motorcycle', 0.0):.3f} |\n"
    )

md_report += "\n---\n## 4. Size Breakdown (Small vs Medium vs Large)\n\n| Video Clip | Small GT (P/R) | Med GT (P/R) | Large GT (P/R) |\n|---|---|---|---|\n"
for r in all_results:
    sb = r["detection"]["size_breakdown"]
    s_str = f"{sb['small']['gt']} ({sb['small']['precision']:.2f}/{sb['small']['recall']:.2f})"
    m_str = f"{sb['medium']['gt']} ({sb['medium']['precision']:.2f}/{sb['medium']['recall']:.2f})"
    l_str = f"{sb['large']['gt']} ({sb['large']['precision']:.2f}/{sb['large']['recall']:.2f})"
    md_report += f"| `{r['video']}` | {s_str} | {m_str} | {l_str} |\n"

md_report += "\n---\n## 5. Detection Error Breakdown\n\n| Video Clip | False Positives | False Negatives | Misclassifications | Localization Errs | Duplicates |\n|---|---|---|---|---|---|\n"
for r in all_results:
    ec = r["error_counts"]
    md_report += (
        f"| `{r['video']}` | {ec.get('FALSE_POSITIVE', 0)} | {ec.get('FALSE_NEGATIVE', 0)} | "
        f"{ec.get('MISCLASSIFICATION', 0)} | {ec.get('LOCALIZATION', 0)} | {ec.get('DUPLICATE', 0)} |\n"
    )

report_path = out_dir / "baseline_report.md"
with open(report_path, "w") as f:
    f.write(md_report)
print(f"Summary Markdown report written to: {report_path}")
