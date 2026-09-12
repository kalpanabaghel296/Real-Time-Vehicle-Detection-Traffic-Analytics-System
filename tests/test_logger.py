"""
Unit tests for EventLogger and TrafficEvent (src/logger.py).
"""

import csv
import json
from pathlib import Path
import pytest
from src.logger import EventLogger, TrafficEvent


def test_traffic_event_serialization():
    ev = TrafficEvent(
        timestamp="2026-09-12T12:00:00",
        frame_idx=100,
        event_type="LINE_CROSSING",
        track_id=1,
        class_name="car",
        direction="DOWN",
        confidence=0.9123,
        snapshot_path="",
        details="Crossed line",
    )
    d = ev.to_dict()
    assert d["track_id"] == 1
    assert d["event_type"] == "LINE_CROSSING"
    assert d["confidence"] == 0.9123


def test_event_logger_csv_and_json(tmp_path):
    csv_file = tmp_path / "test_events.csv"
    json_file = tmp_path / "test_events.json"

    logger = EventLogger(csv_path=str(csv_file), json_path=str(json_file))
    assert csv_file.exists()

    # Log two events
    logger.log_event(
        event_type="LINE_CROSSING",
        track_id=5,
        class_name="car",
        direction="DOWN",
        confidence=0.88,
        frame_idx=50,
        details="Test line crossing",
    )
    logger.log_event(
        event_type="WRONG_WAY_VIOLATION",
        track_id=8,
        class_name="truck",
        direction="UP",
        confidence=0.94,
        frame_idx=75,
        snapshot_path="outputs/snapshots/test.jpg",
        details="Wrong way test",
    )

    logger.save_json()
    assert json_file.exists()

    # Verify CSV content
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) == 3  # Header + 2 rows
        assert reader[0] == EventLogger.CSV_HEADERS
        assert reader[1][2] == "LINE_CROSSING"
        assert reader[2][2] == "WRONG_WAY_VIOLATION"

    # Verify JSON content
    with open(json_file, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["track_id"] == 5
        assert data[1]["track_id"] == 8

    # Verify summary
    summary = logger.get_summary()
    assert summary["total_events"] == 2
    assert summary["line_crossings"] == 1
    assert summary["wrong_way_violations"] == 1
