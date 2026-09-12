"""
Structured Event Logging Module
===============================
Records traffic analytics events (line crossings and wrong-way violations)
into persistent, structured CSV and JSON audit logs.
"""

import csv
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig


@dataclass
class TrafficEvent:
    """
    Standardized traffic event record.

    Attributes:
        timestamp: ISO 8601 formatted timestamp string.
        frame_idx: Video frame number where the event occurred.
        event_type: Category of event ('LINE_CROSSING' or 'WRONG_WAY_VIOLATION').
        track_id: Persistent vehicle tracking ID.
        class_name: Vehicle class ('car', 'truck', 'bus', 'motorcycle').
        direction: Movement direction at event time ('DOWN', 'UP', 'LEFT', 'RIGHT').
        confidence: Model detection confidence [0.0 - 1.0].
        snapshot_path: File path to evidence image (if applicable, else empty string).
        details: Additional descriptive context.
    """

    timestamp: str
    frame_idx: int
    event_type: str
    track_id: int
    class_name: str
    direction: str
    confidence: float
    snapshot_path: str = ""
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EventLogger:
    """
    Manages in-memory event collection and disk persistence to CSV and JSON.
    """

    CSV_HEADERS = [
        "timestamp",
        "frame_idx",
        "event_type",
        "track_id",
        "class_name",
        "direction",
        "confidence",
        "snapshot_path",
        "details",
    ]

    def __init__(
        self,
        config: Optional[TrafficConfig] = None,
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
        clear_existing: bool = False,
    ):
        self.config = config or TrafficConfig()

        logs_dir = Path(self.config.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = Path(csv_path) if csv_path else logs_dir / "events.csv"
        self.json_path = Path(json_path) if json_path else logs_dir / "events.json"

        self.events: List[TrafficEvent] = []
        if clear_existing:
            if self.csv_path.exists():
                self.csv_path.unlink()
            if self.json_path.exists():
                self.json_path.unlink()
        self._init_csv()

    def _init_csv(self) -> None:
        """Initializes CSV file with header row if it does not already exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def log_event(
        self,
        event_type: str,
        track_id: int,
        class_name: str,
        direction: str,
        confidence: float,
        frame_idx: int,
        snapshot_path: str = "",
        details: str = "",
        timestamp: Optional[str] = None,
    ) -> TrafficEvent:
        """
        Records a new traffic event and immediately appends it to disk logs.
        """
        ts = timestamp or datetime.now().isoformat(timespec="seconds")
        event = TrafficEvent(
            timestamp=ts,
            frame_idx=frame_idx,
            event_type=event_type,
            track_id=track_id,
            class_name=class_name,
            direction=direction or "UNKNOWN",
            confidence=round(float(confidence), 4),
            snapshot_path=str(snapshot_path),
            details=details,
        )

        self.events.append(event)
        self._append_to_csv(event)
        return event

    def _append_to_csv(self, event: TrafficEvent) -> None:
        """Appends a single event row to the CSV file."""
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                event.timestamp,
                event.frame_idx,
                event.event_type,
                event.track_id,
                event.class_name,
                event.direction,
                event.confidence,
                event.snapshot_path,
                event.details,
            ])

    def save_json(self) -> None:
        """Saves all logged events as a structured JSON file."""
        data = [e.to_dict() for e in self.events]
        with open(self.json_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical counts of logged events."""
        line_crossings = sum(1 for e in self.events if e.event_type == "LINE_CROSSING")
        violations = sum(1 for e in self.events if e.event_type == "WRONG_WAY_VIOLATION")

        return {
            "total_events": len(self.events),
            "line_crossings": line_crossings,
            "wrong_way_violations": violations,
            "csv_path": str(self.csv_path.resolve()),
            "json_path": str(self.json_path.resolve()),
        }
