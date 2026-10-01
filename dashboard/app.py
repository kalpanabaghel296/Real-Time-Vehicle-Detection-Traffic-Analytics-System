"""
Streamlit Traffic & Vehicle Analytics Dashboard
===============================================
A lightweight interactive web UI for monitoring traffic video analytics,
inspecting class distributions, viewing wrong-way violation snapshots,
and analyzing real-time performance telemetry.
"""

import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.main import run_pipeline
from src.video_processor import VideoReader
from src.tracker import VehicleTracker
from src.counter import VehicleCounter
from src.violation import WrongWayDetector
from src.metrics import PerformanceMonitor
from src.logger import EventLogger
from src.visualizer import Visualizer


def load_events(csv_path: Path) -> pd.DataFrame:
    """Loads event log CSV safely into a pandas DataFrame."""
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_summary(summary_path: Path) -> dict:
    """Loads run summary telemetry JSON."""
    if summary_path.exists() and summary_path.stat().st_size > 0:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def main():
    st.set_page_config(
        page_title="Traffic & Vehicle Analytics",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚦 Real-Time Traffic & Vehicle Analytics System")
    st.caption("End-to-End Computer Vision Pipeline: Detection, ByteTrack Tracking, Counting, and Wrong-Way Violation Monitoring")

    # -------------------------------------------------------------------------
    # Sidebar Configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Pipeline Configuration")

    video_options = {
        "Real Highway CCTV (data/input/traffic.mp4)": "data/input/traffic.mp4",
        "Uploaded Highway Traffic (data/input/188613-883402208.mp4)": "data/input/188613-883402208.mp4",
        "Synthetic Roadway (data/sample/traffic_sample.mp4)": "data/sample/traffic_sample.mp4",
        "📷 Live Laptop Webcam (Camera 0)": "0",
        "📹 External USB Camera (Camera 1)": "1",
        "🌐 Custom RTSP / IP Camera Stream": "RTSP_CUSTOM",
    }
    selected_option = st.sidebar.selectbox("Select Video Source", list(video_options.keys()))
    video_source = video_options[selected_option]

    if video_source == "RTSP_CUSTOM":
        rtsp_input = st.sidebar.text_input("Enter RTSP Stream URL", "rtsp://192.168.1.100:554/stream")
        video_source = rtsp_input.strip()

    # File uploader override
    uploaded_file = st.sidebar.file_uploader("Or Upload Custom Video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        upload_path = Path("data/input") / uploaded_file.name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source = str(upload_path)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

    st.sidebar.subheader("🎯 Model & Highway Configuration")
    preset = st.sidebar.selectbox(
        "Detection Preset Profile",
        [
            "Standard Traffic (640px, Balanced)",
            "Dense / Small / High-Angle Vehicles (1280px, High Recall)",
            "High Precision (640px, Strict)",
        ],
        index=0,
    )
    if "1280px" in preset:
        imgsz = 1280
        default_conf = 0.25
    elif "Strict" in preset:
        imgsz = 640
        default_conf = 0.50
    else:
        imgsz = 640
        default_conf = 0.35

    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, default_conf, 0.05)

    direction_options = {
        "AUTO": "🔄 AUTO (Infer Majority Flow - Recommended)",
        "DOWN": "⬇️ DOWN (Traffic Towards Camera / Downward)",
        "UP": "⬆️ UP (Traffic Away From Camera / Upward)",
        "LEFT": "⬅️ LEFT (Traffic Moving Towards Left)",
        "RIGHT": "➡️ RIGHT (Traffic Moving Towards Right)",
    }
    allowed_direction = st.sidebar.radio(
        "Legal Traffic Flow Direction",
        options=list(direction_options.keys()),
        index=0,
        format_func=lambda k: direction_options[k],
        help="In 2D video coordinates, vehicles driving towards the camera move DOWN the screen, and vehicles driving away move UP. Select AUTO to infer the baseline flow automatically from majority traffic.",
    )

    line_orientation = st.sidebar.selectbox(
        "Counting Line Orientation",
        ["AUTO", "HORIZONTAL", "VERTICAL"],
        index=0,
        help="AUTO places line perpendicular to traffic flow. HORIZONTAL spans left-to-right (for UP/DOWN traffic), VERTICAL spans top-to-bottom (for cross-traffic).",
    )
    line_y_ratio = st.sidebar.slider("Counting Line Position (0.0 = Top/Left, 1.0 = Bottom/Right)", 0.10, 0.90, 0.35, 0.05)
    frame_skip = st.sidebar.selectbox("Frame Skipping", [0, 1, 2], index=0, format_func=lambda x: f"Process all frames (0)" if x == 0 else f"Skip {x} frame(s)")

    reset_logs = st.sidebar.checkbox("Reset audit logs for this run", value=True, help="Clear previous video counts and logs before running this video")
    ignore_median = st.sidebar.checkbox(
        "Filter Opposing Highway Median (Top-Right)",
        value=True,
        help="Filters out vehicles traveling on the separate opposing carriageway visible across the highway barrier in divided highway footage.",
    )
    st.sidebar.caption("💡 **Camera Perspective Tip**: Best results are achieved with standard roadside or overhead CCTV footage (30°-60° angle). High-altitude vertical drone footage experiences COCO domain shift and extreme downsampling.")

    is_live = str(video_source).isdigit() or str(video_source).startswith("rtsp://") or str(video_source).startswith("http://")

    # Output paths
    output_video_path = Path("outputs/videos/processed_video.mp4")
    csv_log_path = Path("outputs/logs/events.csv")
    snapshots_dir = Path("outputs/snapshots")
    summary_path = Path("outputs/logs/summary.json")

    # Pipeline Execution Trigger for file-based sources
    if not is_live:
        run_btn = st.sidebar.button("🚀 Run Analytics Pipeline", type="primary", width="stretch")
        if run_btn:
            with st.spinner("Executing Computer Vision Pipeline... Please wait."):
                run_pipeline(
                    source=video_source,
                    output=str(output_video_path),
                    allowed_direction=allowed_direction,
                    line_y=line_y_ratio,
                    conf_thresh=conf_threshold,
                    frame_skip=frame_skip,
                    preview=False,
                    reset_logs=reset_logs,
                    imgsz=imgsz,
                    line_orientation=line_orientation,
                    ignore_opposing_median=ignore_median,
                )
            st.sidebar.success("✅ Video Processing Complete!")
            st.rerun()
    else:
        st.sidebar.info("📷 Live Camera mode active. Controls are available in the 'Live Camera Stream' tab.")

    # -------------------------------------------------------------------------
    # Top KPI Metrics Row (Dynamic Telemetry)
    # -------------------------------------------------------------------------
    summary = load_summary(summary_path)
    events_df = load_events(csv_log_path)

    if summary:
        total_counted = summary.get("total_counted", 0)
        total_violations = summary.get("total_violations", 0)
        avg_fps = f"{summary.get('avg_fps', 0.0):.1f} FPS"
        avg_lat = f"{summary.get('avg_inference_ms', 0.0):.1f} ms"
        eff_dir = summary.get("effective_allowed_direction", summary.get("allowed_direction", "AUTO"))
    else:
        total_counted = len(events_df[events_df["event_type"] == "LINE_CROSSING"]) if not events_df.empty else 0
        total_violations = len(events_df[events_df["event_type"] == "WRONG_WAY_VIOLATION"]) if not events_df.empty else 0
        avg_fps = "-- FPS"
        avg_lat = "-- ms"
        eff_dir = allowed_direction

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("🚗 Total Vehicles Counted", total_counted)
    with kpi2:
        st.metric(
            "🚨 Wrong-Way Violations",
            total_violations,
            delta=f"{total_violations} alerts" if total_violations > 0 else "0 (Normal Flow)",
            delta_color="inverse" if total_violations > 0 else "normal",
            help=f"Legal Flow: {eff_dir}",
        )
    with kpi3:
        st.metric("⚡ Processing Speed", avg_fps, help="Actual measured pipeline throughput on host machine")
    with kpi4:
        st.metric("⏱️ Inference Latency", avg_lat, help="Actual measured YOLO forward pass latency per frame")

    st.divider()

    # -------------------------------------------------------------------------
    # Main Dashboard Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🎥 Video Stream & Analytics", "🚨 Violation Snapshots", "📋 Audit Event Log"])

    with tab1:
        if is_live:
            st.subheader("📹 Live Camera Detection & Tracking Feed")
            source_desc = f"Hardware Camera Index {video_source}" if str(video_source).isdigit() else f"Network Stream: {video_source}"
            st.caption(f"Active Source: **{source_desc}** | Orientation: **{line_orientation}** | Allowed Flow: **{allowed_direction}**")

            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                start_live = st.button("▶️ Start Live Stream", type="primary", width="stretch")
            with c2:
                stop_live = st.button("⏹️ Stop Stream", width="stretch")

            if "is_live_running" not in st.session_state:
                st.session_state.is_live_running = False

            if start_live:
                st.session_state.is_live_running = True
            if stop_live:
                st.session_state.is_live_running = False

            live_status_box = st.empty()
            col_live_video, col_live_stats = st.columns([3, 2])
            with col_live_video:
                live_frame_box = st.empty()
            with col_live_stats:
                live_metrics_box = st.empty()

            if st.session_state.is_live_running:
                live_status_box.success("🔴 Live Camera Active — Detecting & Tracking in Real-Time. Click 'Stop Stream' to halt.")
                cfg = TrafficConfig(
                    video_source=str(video_source),
                    confidence_threshold=conf_threshold,
                    allowed_direction=allowed_direction,
                    imgsz=imgsz,
                    line_orientation=line_orientation,
                    frame_skip=frame_skip,
                )
                tracker = VehicleTracker(cfg)
                counter = VehicleCounter(cfg, counting_direction="ANY")
                violation_detector = WrongWayDetector(cfg, allowed_direction=allowed_direction)
                monitor = PerformanceMonitor()
                logger = EventLogger(cfg, clear_existing=reset_logs)
                visualizer = Visualizer()

                try:
                    with VideoReader(int(video_source) if str(video_source).isdigit() else video_source) as reader:
                        meta = reader.metadata
                        norm_o = line_orientation.upper()
                        if norm_o == "VERTICAL" or (norm_o == "AUTO" and allowed_direction in ["LEFT", "RIGHT"]):
                            l_pos = int(meta.width * line_y_ratio)
                            counter.counting_line = ((l_pos, 0), (l_pos, meta.height))
                        else:
                            l_pos = int(meta.height * line_y_ratio)
                            counter.counting_line = ((0, l_pos), (meta.width, l_pos))

                        for f_idx, frame in reader.read_frames(frame_skip=frame_skip):
                            if not st.session_state.is_live_running:
                                break
                            monitor.start_frame()
                            monitor.mark_preprocessed()
                            tracks = tracker.update(frame, f_idx)
                            monitor.mark_inference_complete()

                            new_crossings = counter.update(tracks, frame.shape[:2], f_idx)
                            for ev in new_crossings:
                                logger.log_event(
                                    event_type="LINE_CROSSING",
                                    track_id=ev["track_id"],
                                    class_name=ev["class_name"],
                                    direction="CROSSING",
                                    confidence=ev["confidence"],
                                    frame_idx=f_idx,
                                    details="Live camera line crossing",
                                )

                            new_violations = violation_detector.update(frame, tracks, f_idx)
                            for v in new_violations:
                                logger.log_event(
                                    event_type="WRONG_WAY_VIOLATION",
                                    track_id=v["track_id"],
                                    class_name=v["class_name"],
                                    direction=v["direction"],
                                    confidence=v["confidence"],
                                    frame_idx=f_idx,
                                    snapshot_path=v.get("snapshot_path", ""),
                                    details="Live camera wrong-way violation",
                                )

                            monitor.mark_tracking_complete()
                            annotated = visualizer.render(
                                frame,
                                tracks,
                                counter=counter,
                                violation_detector=violation_detector,
                                monitor=monitor,
                            )
                            monitor.end_frame(f_idx)

                            # Stream annotated frame into web UI
                            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            live_frame_box.image(rgb, channels="RGB", width="stretch")

                            # Update live telemetry card
                            live_perf = monitor.get_summary()
                            live_metrics_box.markdown(
                                f"""
                                ### 📊 Live Stream Telemetry
                                - **Active Tracks:** `{len(tracks)}`
                                - **Total Counted:** `{counter.total_count}`
                                - **Violations:** `{len(violation_detector.violations)}`
                                - **Real-Time Speed:** `{live_perf['avg_fps']} FPS`
                                - **Inference Latency:** `{live_perf['avg_inference_ms']} ms`
                                """
                            )
                            time.sleep(0.01)
                except Exception as e:
                    live_status_box.error(f"Live stream interrupted: {e}")
                    st.session_state.is_live_running = False
            else:
                live_frame_box.info("Click '▶️ Start Live Stream' to activate live camera detection.")
        else:
            col_video, col_stats = st.columns([3, 2])

            with col_video:
                st.subheader("Annotated Video Playback")
                if output_video_path.exists() and output_video_path.stat().st_size > 0:
                    with open(output_video_path, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes, format="video/mp4")
                    st.download_button(
                        label="⬇️ Download Processed Video (MP4)",
                        data=video_bytes,
                        file_name="processed_traffic_video.mp4",
                        mime="video/mp4",
                    )
                else:
                    st.info("Click 'Run Analytics Pipeline' in the sidebar to generate the processed video.")

            with col_stats:
                st.subheader("Vehicle Class Breakdown")
                if not events_df.empty and "class_name" in events_df.columns:
                    crossing_df = events_df[events_df["event_type"] == "LINE_CROSSING"]
                    if not crossing_df.empty:
                        class_counts = crossing_df["class_name"].value_counts()
                        st.bar_chart(class_counts)
                        st.dataframe(
                            class_counts.reset_index().rename(columns={"index": "Class", "class_name": "Count"}),
                            width="stretch",
                        )
                    else:
                        st.info("No line crossings recorded yet.")
                else:
                    st.info("Class statistics will appear here after running the pipeline.")

    with tab2:
        st.subheader("Forensic Evidence Gallery (Wrong-Way Snapshots)")
        valid_snapshots = []
        if not events_df.empty:
            viol_events = events_df[events_df["event_type"] == "WRONG_WAY_VIOLATION"]
            for _, row in viol_events.iterrows():
                snap_path_str = str(row.get("snapshot_path", ""))
                if snap_path_str:
                    p = Path(snap_path_str)
                    if p.exists() and p.is_file() and p not in valid_snapshots:
                        valid_snapshots.append(p)
        elif snapshots_dir.exists():
            valid_snapshots = sorted(list(snapshots_dir.glob("violation_*.jpg")), reverse=True)

        if valid_snapshots:
            cols = st.columns(min(3, len(valid_snapshots)))
            for idx, snap_path in enumerate(valid_snapshots):
                col = cols[idx % len(cols)]
                with col:
                    img = Image.open(snap_path)
                    st.image(img, caption=f"Evidence: {snap_path.name}", width="stretch")
                    st.download_button(
                        label="Download Evidence",
                        data=open(snap_path, "rb").read(),
                        file_name=snap_path.name,
                        mime="image/jpeg",
                        key=f"dl_{idx}",
                    )
        else:
            st.info("No violation snapshots have been recorded for this video run.")

    with tab3:
        st.subheader("Structured Traffic Audit Events")
        if not events_df.empty:
            st.dataframe(events_df, width="stretch")
            csv_data = events_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Audit Log (CSV)",
                data=csv_data,
                file_name="traffic_events.csv",
                mime="text/csv",
            )
        else:
            st.info("Audit log is currently empty.")


if __name__ == "__main__":
    main()
