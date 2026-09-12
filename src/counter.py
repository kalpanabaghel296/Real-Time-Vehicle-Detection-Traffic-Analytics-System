"""
Vehicle Counting Module
=======================
Implements robust line-based vehicle counting with:
- Configurable virtual counting line
- 2D cross-product line segment intersection algorithm
- Track ID deduplication (guarantees zero double-counting)
- Real-time class-wise breakdown (Cars, Trucks, Buses, Motorcycles)
- Visual HUD overlay
"""

from pathlib import Path
import sys
from typing import List, Tuple, Dict, Set, Any, Optional
import cv2
import numpy as np

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.tracker import TrackedVehicle
from src.utils import do_segments_intersect


class VehicleCounter:
    """
    Manages virtual line crossing detection and vehicle tallying.
    """

    def __init__(
        self,
        config: Optional[TrafficConfig] = None,
        counting_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        counting_direction: str = "ANY",
    ):
        """
        Args:
            config: TrafficConfig instance.
            counting_line: Optional override for line coordinates ((x1, y1), (x2, y2)).
            counting_direction: Allowed direction to count ("ANY", "DOWN", "UP", "LEFT", "RIGHT").
        """
        self.config = config or TrafficConfig()
        self.counting_line = counting_line or self.config.counting_line
        self.counting_direction = counting_direction or self.config.counting_direction

        # Persistent counting registries
        self.total_count: int = 0
        self.counts_by_class: Dict[str, int] = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorcycle": 0,
        }

        # Set of Track IDs already counted - ensures O(1) duplicate prevention
        self.counted_ids: Set[int] = set()

        # Audit log of line crossing events
        self.crossing_events: List[Dict[str, Any]] = []

    def _ensure_line_coordinates(self, frame_shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculates default horizontal line across the middle-lower portion of the frame
        if no explicit line coordinates were configured.
        """
        if self.counting_line is not None:
            return self.counting_line

        h, w = frame_shape[:2]
        # Default: Horizontal line at 35% of frame height across full roadway width
        line_y = int(h * 0.35)
        self.counting_line = ((0, line_y), (w, line_y))
        return self.counting_line

    def update(
        self,
        tracks: List[TrackedVehicle],
        frame_shape: Tuple[int, int],
        frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates active tracks for counting line crossing in the current frame.

        Args:
            tracks: Active TrackedVehicle instances from the tracker.
            frame_shape: (height, width) of the video frame.
            frame_idx: Current frame index.

        Returns:
            List of new crossing events detected in this frame.
        """
        line_p1, line_p2 = self._ensure_line_coordinates(frame_shape)
        new_events: List[Dict[str, Any]] = []

        for veh in tracks:
            # 1. Duplicate Prevention: If already counted, skip immediately
            if veh.track_id in self.counted_ids:
                continue

            # 2. Minimum Trajectory Check: Need previous and current centroid
            if veh.previous_centroid is None:
                continue

            p1 = veh.previous_centroid
            p2 = veh.centroid

            # 3. Geometric Line Intersection Check
            # Checks if vehicle step segment (p1 -> p2) intersects counting line (line_p1 -> line_p2)
            if do_segments_intersect(p1, p2, line_p1, line_p2):
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # 4. Optional Direction Filter Check
                if self.counting_direction == "DOWN" and dy <= 0:
                    continue
                if self.counting_direction == "UP" and dy >= 0:
                    continue
                if self.counting_direction == "RIGHT" and dx <= 0:
                    continue
                if self.counting_direction == "LEFT" and dx >= 0:
                    continue

                # 5. Register Valid Crossing
                self.total_count += 1
                veh_class = veh.class_name.lower()
                self.counts_by_class[veh_class] = self.counts_by_class.get(veh_class, 0) + 1
                self.counted_ids.add(veh.track_id)
                veh.counted = True

                event = {
                    "frame_idx": frame_idx,
                    "track_id": veh.track_id,
                    "class_name": veh.class_name,
                    "confidence": veh.confidence,
                    "centroid": veh.centroid,
                    "displacement": (dx, dy),
                }
                self.crossing_events.append(event)
                new_events.append(event)

        return new_events

    def draw_hud(
        self,
        frame: np.ndarray,
        active_track_count: int = 0,
    ) -> np.ndarray:
        """
        Renders the virtual counting line and statistics HUD card onto the frame.

        Args:
            frame: Original BGR image.
            active_track_count: Number of active tracks in current frame.

        Returns:
            Annotated image with line and HUD overlay.
        """
        annotated = frame.copy()
        line_p1, line_p2 = self._ensure_line_coordinates(frame.shape[:2])

        # 1. Draw Virtual Counting Line
        line_color = (0, 255, 255)  # Bright Yellow
        cv2.line(annotated, line_p1, line_p2, line_color, 3, cv2.LINE_AA)

        # Label along the line
        mid_x = (line_p1[0] + line_p2[0]) // 2
        mid_y = (line_p1[1] + line_p2[1]) // 2
        cv2.putText(
            annotated,
            "--- COUNTING LINE ---",
            (max(10, mid_x - 120), max(20, mid_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_color,
            2,
            cv2.LINE_AA,
        )

        # 2. Render Semi-Transparent HUD Statistics Card (Top-Left)
        card_w = 230
        card_h = 160
        card_x = 15
        card_y = 45

        # Create overlay for alpha transparency
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (20, 20, 20),
            -1,
        )
        # 75% opacity dark background
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Draw card border
        cv2.rectangle(
            annotated,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (80, 80, 80),
            1,
        )

        # 3. Draw Text Statistics inside Card
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255)
        accent_color = (0, 255, 255)

        cv2.putText(annotated, "TRAFFIC METRICS", (card_x + 10, card_y + 22), font, 0.55, accent_color, 2)
        cv2.line(annotated, (card_x + 10, card_y + 28), (card_x + card_w - 10, card_y + 28), (100, 100, 100), 1)

        cv2.putText(annotated, f"Total Count: {self.total_count}", (card_x + 10, card_y + 48), font, 0.52, (50, 255, 50), 2)
        cv2.putText(annotated, f"Cars: {self.counts_by_class.get('car', 0)}", (card_x + 10, card_y + 70), font, 0.45, text_color, 1)
        cv2.putText(annotated, f"Trucks: {self.counts_by_class.get('truck', 0)}", (card_x + 10, card_y + 90), font, 0.45, text_color, 1)
        cv2.putText(annotated, f"Buses: {self.counts_by_class.get('bus', 0)}", (card_x + 10, card_y + 110), font, 0.45, text_color, 1)
        cv2.putText(annotated, f"Motorcycles: {self.counts_by_class.get('motorcycle', 0)}", (card_x + 10, card_y + 130), font, 0.45, text_color, 1)
        cv2.putText(annotated, f"Active Tracks: {active_track_count}", (card_x + 10, card_y + 150), font, 0.42, (180, 180, 180), 1)

        return annotated


def main():
    """CLI runner to test vehicle counting pipeline on a video stream."""
    import argparse
    from src.video_processor import VideoReader, VideoWriterHelper
    from src.tracker import VehicleTracker

    parser = argparse.ArgumentParser(description="Test vehicle counting pipeline on video.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/input/traffic.mp4",
        help="Path to input video (default: data/input/traffic.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/counting_test.mp4",
        help="Destination path for counted output video",
    )
    parser.add_argument(
        "--line-y",
        type=float,
        default=0.35,
        help="Normalized Y position for counting line [0.0 - 1.0]",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live preview window",
    )
    args = parser.parse_args()

    cfg = TrafficConfig(video_source=args.source)
    tracker = VehicleTracker(cfg)

    with VideoReader(args.source) as reader:
        meta = reader.metadata
        print(f"[x] Opened video: {meta.width}x{meta.height} @ {meta.fps} FPS ({meta.total_frames} frames)")

        # Configure counting line across frame width at configured Y
        line_y = int(meta.height * args.line_y)
        counting_line = ((0, line_y), (meta.width, line_y))
        counter = VehicleCounter(cfg, counting_line=counting_line)
        print(f"[x] Configured counting line at: {counting_line}")

        with VideoWriterHelper(
            output_path=args.output,
            fps=meta.fps,
            frame_size=(meta.width, meta.height),
        ) as writer:
            for idx, frame in reader.read_frames():
                # 1. Update Tracker
                active_tracks = tracker.update(frame, idx)

                # 2. Update Counter
                new_crossings = counter.update(active_tracks, frame.shape[:2], idx)
                for ev in new_crossings:
                    print(f"  [+] Frame {idx:03d} | Counted {ev['class_name']} (ID: {ev['track_id']}) | Total: {counter.total_count}")

                # 3. Render Tracking Annotations & Counting HUD
                annotated = tracker.draw_tracks(frame, active_tracks, draw_trajectory=True)
                annotated = counter.draw_hud(annotated, active_track_count=len(active_tracks))

                writer.write(annotated)

                if args.preview:
                    cv2.imshow("Vehicle Counting Test", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.preview:
                cv2.destroyAllWindows()

            print("=" * 60)
            print("[x] COUNTING SUMMARY:")
            print(f"    - Total Vehicles Counted: {counter.total_count}")
            print(f"    - Cars: {counter.counts_by_class.get('car', 0)}")
            print(f"    - Trucks: {counter.counts_by_class.get('truck', 0)}")
            print(f"    - Buses: {counter.counts_by_class.get('bus', 0)}")
            print(f"    - Motorcycles: {counter.counts_by_class.get('motorcycle', 0)}")
            print(f"    - Output video saved to: {args.output}")
            print("=" * 60)


if __name__ == "__main__":
    main()
