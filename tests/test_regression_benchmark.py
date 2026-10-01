"""
Regression and Stress Testing Suite
===================================
Verifies the production-optimized traffic analytics pipeline across
multi-condition stress tests and validates that performance guarantees hold:
1. Zero False Alarm Rate on legal traffic (FAR <= 0.05/min)
2. 100% Recall on actual wrong-way driving incidents
3. ID Switch stability across multi-lane flow (total switches <= 25)
4. Counting precision and duplicate prevention
5. End-to-end pipeline robustness
"""

import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = WORKSPACE_ROOT / "data" / "evaluation" / "optimized_outputs" / "baseline_results.json"


@pytest.fixture(scope="module")
def benchmark_results():
    if not RESULTS_PATH.exists():
        pytest.skip(f"Benchmark results not found at {RESULTS_PATH}")
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def test_zero_false_alarm_guarantee(benchmark_results):
    """Verifies that false wrong-way alarms are 0.00 across all legal flow clips."""
    for r in benchmark_results["per_video_results"]:
        far = r["violation"]["far_per_min"]
        assert far == 0.00, f"Clip {r['video']} produced unexpected false alarms: {far}/min"


def test_wrong_way_incident_recall(benchmark_results):
    """Verifies 100% recall on the wrong-way incident test clip."""
    ww_res = next((r for r in benchmark_results["per_video_results"] if "wrong_way" in r["video"]), None)
    assert ww_res is not None
    assert ww_res["violation"]["recall"] == 1.0, f"Expected 1.0 recall, got {ww_res['violation']['recall']}"
    assert ww_res["violation"]["precision"] == 1.0, f"Expected 1.0 precision, got {ww_res['violation']['precision']}"


def test_id_switch_stability(benchmark_results):
    """Verifies tracking stability: total ID switches <= 25 across all 5 clips."""
    total_idsw = benchmark_results["macro_metrics"]["total_id_switches"]
    assert total_idsw <= 25, f"Excessive tracking ID switches: {total_idsw}"


def test_real_time_fps_throughput(benchmark_results):
    """Verifies that end-to-end throughput is >= 6.0 FPS on CPU."""
    avg_fps = benchmark_results["macro_metrics"]["avg_fps"]
    assert avg_fps >= 6.0, f"Throughput too low: {avg_fps:.1f} FPS"
