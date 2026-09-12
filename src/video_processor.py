"""
Video Input and Output Processing Module
========================================
Encapsulates OpenCV VideoCapture and VideoWriter with robust error handling,
metadata extraction, frame generator streaming, and codec fallbacks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple, Optional, Union
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Stores fundamental properties of a video stream."""

    source: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float

    def __str__(self) -> str:
        return (
            f"VideoMetadata(source='{self.source}', resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, frames={self.total_frames}, duration={self.duration_seconds:.2f}s)"
        )


class VideoReader:
    """
    Robust wrapper around cv2.VideoCapture.

    Supports:
    - Local video files (mp4, avi, mkv, etc.)
    - Live webcam feeds (specified as an integer index like 0 or '0')
    - RTSP/HTTP network video streams
    """

    def __init__(self, source: Union[str, int]):
        """
        Initializes the VideoReader.

        Args:
            source: File path (str) or webcam device index (int or str).
        """
        self.raw_source = source
        self.is_camera = False

        # Determine if source is an integer camera device ID
        if isinstance(source, int):
            self.capture_source = source
            self.is_camera = True
        elif isinstance(source, str) and source.strip().isdigit():
            self.capture_source = int(source.strip())
            self.is_camera = True
        else:
            self.capture_source = str(source)
            # Verify file exists if it's not a URL
            if not (str(source).startswith("rtsp://") or str(source).startswith("http://")):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(
                        f"[VideoReader Error] Video source path does not exist: {path.resolve()}"
                    )
                self.capture_source = str(path.resolve())

        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None
        self._open()

    def _open(self) -> None:
        """Opens the video source and verifies readiness."""
        self.cap = cv2.VideoCapture(self.capture_source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[VideoReader Error] Failed to open video source: '{self.raw_source}'. "
                f"Please verify camera permissions, file integrity, or video codec."
            )

        # Extract stream properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        # Handle edge cases: cameras or unindexed streams reporting 0 or negative FPS
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Safe default assumption for live streams

        if self.is_camera:
            total_frames = -1
            duration = -1.0
        else:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        self.metadata = VideoMetadata(
            source=str(self.raw_source),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
        )

    def read_frames(
        self, frame_skip: int = 0
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yields frames sequentially as a generator to conserve memory.

        Args:
            frame_skip: Number of frames to skip between yields (0 = process every frame).

        Yields:
            Tuple of (frame_index, frame_image_bgr)
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("[VideoReader Error] VideoCapture stream is not active.")

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # End of stream or unreadable frame
                break

            # If frame_skip > 0, only yield every (frame_skip + 1)-th frame
            if frame_skip > 0 and (frame_idx % (frame_skip + 1) != 0):
                frame_idx += 1
                continue

            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        """Releases the underlying OpenCV VideoCapture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriterHelper:
    """
    Robust wrapper around cv2.VideoWriter with automatic codec fallbacks.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = "mp4v",
    ):
        """
        Args:
            output_path: Target destination path (e.g. outputs/videos/result.mp4).
            fps: Frame rate of the output video.
            frame_size: (width, height) of frames.
            codec: 4-character codec code (default 'mp4v').
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        self.codec = codec

        self.writer: Optional[cv2.VideoWriter] = None
        self._init_writer()

    def _init_writer(self) -> None:
        """Attempts to initialize VideoWriter with fallback codec candidates."""
        # Preferred fourcc candidates for mp4 / avi containers
        codecs_to_try = [self.codec, "mp4v", "avc1", "XVID", "MJPG"]
        seen = set()

        for c in codecs_to_try:
            if c in seen:
                continue
            seen.add(c)
            fourcc = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size,
            )
            if writer.isOpened():
                self.writer = writer
                self.codec = c
                return

        raise RuntimeError(
            f"[VideoWriter Error] Failed to initialize cv2.VideoWriter for '{self.output_path}' "
            f"with tested codecs: {codecs_to_try}"
        )

    def write(self, frame: np.ndarray) -> None:
        """Writes a single BGR frame to the video file."""
        if self.writer is None or not self.writer.isOpened():
            raise RuntimeError("[VideoWriter Error] Writer is closed or uninitialized.")

        # Ensure frame dimensions match declared output size
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)

    def release(self) -> None:
        """Releases VideoWriter and finalizes file on disk."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self._ensure_browser_compatible()

    def _ensure_browser_compatible(self) -> None:
        """
        Transcodes output video to web-compatible H.264 (avc1) using ffmpeg if available.
        Modern web browsers (Chrome, Edge, Safari) cannot play raw OpenCV 'mp4v' streams.
        """
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin or not self.output_path.exists():
            return

        temp_h264 = self.output_path.with_name(f"{self.output_path.stem}_web.mp4")
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(self.output_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-crf", "22",
                str(temp_h264),
            ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and temp_h264.exists() and temp_h264.stat().st_size > 0:
                self.output_path.unlink()
                temp_h264.rename(self.output_path)
        except Exception:
            if temp_h264.exists():
                try:
                    temp_h264.unlink()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def main():
    """CLI runner to test video input/output pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Test VideoReader and VideoWriterHelper.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/sample/traffic_sample.mp4",
        help="Path to video file or webcam index (default: data/sample/traffic_sample.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/videos/test_stream.mp4",
        help="Destination path for output video",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display OpenCV imshow window during processing",
    )
    args = parser.parse_args()

    print(f"[*] Opening video source: {args.source}")
    try:
        with VideoReader(args.source) as reader:
            meta = reader.metadata
            print(f"[x] Successfully initialized video reader:")
            print(f"    - Resolution: {meta.width}x{meta.height}")
            print(f"    - FPS: {meta.fps:.2f}")
            print(f"    - Total Frames: {meta.total_frames}")
            print(f"    - Duration: {meta.duration_seconds:.2f}s")

            with VideoWriterHelper(
                output_path=args.output,
                fps=meta.fps,
                frame_size=(meta.width, meta.height),
            ) as writer:
                print(f"[*] Writing processed stream to: {args.output}")
                frame_count = 0
                for idx, frame in reader.read_frames():
                    # Stamp frame index on the test video
                    annotated = frame.copy()
                    cv2.putText(
                        annotated,
                        f"Milestone 2 Test Frame: {idx}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    writer.write(annotated)
                    frame_count += 1

                    if args.preview:
                        cv2.imshow("Milestone 2 Video Preview", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if args.preview:
                    cv2.destroyAllWindows()

                print(f"[x] Video pipeline test complete! Wrote {frame_count} frames to {args.output}")

    except Exception as e:
        print(f"[!] Video processing error: {e}")
        raise


if __name__ == "__main__":
    main()
