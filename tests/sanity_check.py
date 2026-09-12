"""
Sanity check script to verify Python environment, GPU support, and core modules.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("TRAFFIC & VEHICLE ANALYTICS: ENVIRONMENT SANITY CHECK")
    print("=" * 60)

    # 1. Python runtime
    print(f"[x] Python Version: {sys.version.split()[0]}")

    # 2. Configuration loading
    try:
        from config.config import TrafficConfig

        cfg = TrafficConfig()
        cfg.ensure_directories()
        print(f"[x] TrafficConfig initialized successfully.")
        print(f"    - Default Model: {cfg.model_name}")
        print(f"    - Target Classes: {cfg.target_class_ids}")
        print(f"    - Input Resolution: {cfg.input_size}")
    except Exception as e:
        print(f"[!] Error loading TrafficConfig: {e}")
        return False

    # 3. Geometric utilities
    try:
        from src.utils import calculate_centroid, do_segments_intersect, calculate_iou

        cx, cy = calculate_centroid((100, 100, 200, 200))
        assert (cx, cy) == (150, 150)
        intersect = do_segments_intersect((0, 50), (100, 50), (50, 0), (50, 100))
        assert intersect is True
        iou = calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0
        print(f"[x] Geometry and Vector Math Utilities verified.")
    except Exception as e:
        print(f"[!] Error verifying src.utils: {e}")
        return False

    # 4. Dependency checks
    try:
        import numpy as np

        print(f"[x] NumPy Version: {np.__version__}")
    except ImportError as e:
        print(f"[!] NumPy missing: {e}")

    try:
        import cv2

        print(f"[x] OpenCV Version: {cv2.__version__}")
    except ImportError as e:
        print(f"[!] OpenCV missing: {e}")

    try:
        import torch

        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        print(f"[x] PyTorch Version: {torch.__version__}")
        print(f"    - CUDA Available: {cuda_avail}")
        print(f"    - GPU Device: {device_name}")
    except ImportError as e:
        print(f"[!] PyTorch missing: {e}")

    try:
        import ultralytics

        print(f"[x] Ultralytics Version: {ultralytics.__version__}")
    except ImportError as e:
        print(f"[!] Ultralytics missing: {e}")

    print("=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
