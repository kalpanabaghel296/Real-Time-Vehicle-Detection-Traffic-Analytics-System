"""
Unit tests for VideoReader and VideoWriterHelper (src/video_processor.py).
"""

from pathlib import Path
import pytest
import numpy as np
from src.video_processor import VideoReader, VideoWriterHelper
from data.sample.generate_sample_video import generate_synthetic_traffic_video


@pytest.fixture(scope="session")
def sample_video_path(tmp_path_factory):
    """Fixture that generates a temporary 2-second test video."""
    tmp_dir = tmp_path_factory.mktemp("video_data")
    video_file = tmp_dir / "test_traffic.mp4"
    generate_synthetic_traffic_video(
        output_path=str(video_file),
        width=320,
        height=240,
        fps=30,
        duration_sec=2,
    )
    return str(video_file)


def test_missing_video_raises_error():
    with pytest.raises(FileNotFoundError):
        VideoReader("non_existent_video_path.mp4")


def test_video_metadata_reading(sample_video_path):
    with VideoReader(sample_video_path) as reader:
        meta = reader.metadata
        assert meta is not None
        assert meta.width == 320
        assert meta.height == 240
        assert meta.fps == 30.0
        assert meta.total_frames == 60  # 30 fps * 2 sec
        assert pytest.approx(meta.duration_seconds, 0.1) == 2.0


def test_read_frames_generator(sample_video_path):
    with VideoReader(sample_video_path) as reader:
        frames = list(reader.read_frames(frame_skip=0))
        assert len(frames) == 60

        idx, frame = frames[0]
        assert idx == 0
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (240, 320, 3)


def test_frame_skipping(sample_video_path):
    with VideoReader(sample_video_path) as reader:
        # frame_skip=1 should yield every 2nd frame (indices 0, 2, 4, ...) -> 30 frames
        frames = list(reader.read_frames(frame_skip=1))
        assert len(frames) == 30
        assert frames[0][0] == 0
        assert frames[1][0] == 2


def test_video_writer_helper(tmp_path):
    out_file = tmp_path / "output_test.mp4"
    writer = VideoWriterHelper(
        output_path=out_file,
        fps=20.0,
        frame_size=(320, 240),
    )

    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(10):
        writer.write(dummy_frame)
    writer.release()

    assert out_file.exists()
    assert out_file.stat().st_size > 0
