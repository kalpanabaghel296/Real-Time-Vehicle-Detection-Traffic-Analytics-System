"""
Synthetic Traffic Video Generator
==================================
Generates a realistic synthetic road scenario with moving vehicle bounding boxes
to enable deterministic, offline end-to-end verification of the analytics pipeline.
"""

from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_traffic_video(
    output_path: str = "data/sample/traffic_sample.mp4",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    duration_sec: int = 6,
) -> str:
    """
    Creates a synthetic video simulating a two-lane roadway with traffic.

    Lane 1 (Left): Standard flow moving DOWN
    Lane 2 (Right): Standard flow moving DOWN
    Violator: Vehicle traveling UP against flow (Wrong-Way)
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target_path), fourcc, fps, (width, height))
    total_frames = fps * duration_sec

    # Define vehicle simulation objects
    # vehicle = {name, color, w, h, start_frame, init_x, init_y, speed_y}
    vehicles = [
        # Car 1: Lane 1 moving DOWN
        {"name": "Car 1", "color": (220, 100, 50), "w": 50, "h": 80, "start": 0, "x": 180, "y": -80, "vy": 3.5},
        # Truck: Lane 2 moving DOWN
        {"name": "Truck", "color": (50, 160, 220), "w": 70, "h": 120, "start": 15, "x": 400, "y": -120, "vy": 2.5},
        # Car 2: Lane 1 moving DOWN
        {"name": "Car 2", "color": (180, 50, 200), "w": 48, "h": 76, "start": 45, "x": 200, "y": -80, "vy": 4.0},
        # Wrong-Way Car: Lane 2 traveling UP against flow
        {"name": "WrongWay", "color": (0, 0, 220), "w": 52, "h": 80, "start": 30, "x": 380, "y": height + 80, "vy": -3.5},
    ]

    for frame_idx in range(total_frames):
        # 1. Draw Background Scene (Road and Environment)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Grass borders
        frame[:, :80] = (40, 120, 40)       # Left grass
        frame[:, width - 80:] = (40, 120, 40) # Right grass

        # Road surface
        frame[:, 80:width - 80] = (60, 60, 60) # Dark asphalt

        # Road boundary solid white lines
        cv2.line(frame, (85, 0), (85, height), (240, 240, 240), 3)
        cv2.line(frame, (width - 85, 0), (width - 85, height), (240, 240, 240), 3)

        # Dashed yellow center dividing line
        dash_len = 25
        gap_len = 20
        y = 0
        while y < height:
            cv2.line(frame, (width // 2, y), (width // 2, y + dash_len), (0, 215, 255), 3)
            y += dash_len + gap_len

        # 2. Draw Moving Simulated Vehicles
        for v in vehicles:
            if frame_idx >= v["start"]:
                elapsed = frame_idx - v["start"]
                curr_y = int(v["y"] + v["vy"] * elapsed)
                curr_x = int(v["x"])

                # Draw vehicle body if within frame bounds
                if -v["h"] <= curr_y <= height + v["h"]:
                    # Main vehicle body
                    cv2.rectangle(
                        frame,
                        (curr_x - v["w"] // 2, curr_y - v["h"] // 2),
                        (curr_x + v["w"] // 2, curr_y + v["h"] // 2),
                        v["color"],
                        -1,
                    )
                    # Vehicle windshield / roof
                    cv2.rectangle(
                        frame,
                        (curr_x - v["w"] // 3, curr_y - v["h"] // 4),
                        (curr_x + v["w"] // 3, curr_y + v["h"] // 4),
                        (30, 30, 30),
                        -1,
                    )
                    # Headlights / Taillights
                    light_color = (0, 255, 255) if v["vy"] > 0 else (0, 0, 255)
                    light_y = curr_y + (v["h"] // 2 - 4 if v["vy"] > 0 else -v["h"] // 2 + 4)
                    cv2.circle(frame, (curr_x - v["w"] // 3, light_y), 4, light_color, -1)
                    cv2.circle(frame, (curr_x + v["w"] // 3, light_y), 4, light_color, -1)

        # Stamp synthetic watermark
        cv2.putText(
            frame,
            f"Synthetic Traffic Stream - Frame {frame_idx:03d}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    print(f"[x] Generated synthetic video at: {target_path.resolve()} ({total_frames} frames, {duration_sec}s)")
    return str(target_path.resolve())


if __name__ == "__main__":
    generate_synthetic_traffic_video()
