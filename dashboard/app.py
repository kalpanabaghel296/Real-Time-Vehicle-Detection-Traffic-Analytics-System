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
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import TrafficConfig
from src.main import run_pipeline


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
        "Synthetic Roadway (data/sample/traffic_sample.mp4)": "data/sample/traffic_sample.mp4",
    }
    selected_option = st.sidebar.selectbox("Select Video Source", list(video_options.keys()))
    video_source = video_options[selected_option]

    # File uploader override
    uploaded_file = st.sidebar.file_uploader("Or Upload Custom Video (MP4/AVI)", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        upload_path = Path("data/input") / uploaded_file.name
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source = str(upload_path)
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

    allowed_direction = st.sidebar.radio("Legal Traffic Flow Direction", ["UP", "DOWN", "LEFT", "RIGHT"], index=0)
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.10, 0.90, 0.35, 0.05)
    line_y_ratio = st.sidebar.slider("Counting Line Height (0.0 = Top, 1.0 = Bottom)", 0.10, 0.90, 0.35, 0.05)
    frame_skip = st.sidebar.selectbox("Frame Skipping", [0, 1, 2], index=0, format_func=lambda x: f"Process all frames (0)" if x == 0 else f"Skip {x} frame(s)")

    reset_logs = st.sidebar.checkbox("Reset audit logs for this run", value=True, help="Clear previous video counts and logs before running this video")
    st.sidebar.caption("💡 **Camera Perspective Tip**: Best results are achieved with standard roadside or overhead CCTV footage (30°-60° angle). High-altitude vertical drone footage experiences COCO domain shift and extreme downsampling.")

    run_btn = st.sidebar.button("🚀 Run Analytics Pipeline", type="primary", width="stretch")

    # Output paths
    output_video_path = Path("outputs/videos/processed_video.mp4")
    csv_log_path = Path("outputs/logs/events.csv")
    snapshots_dir = Path("outputs/snapshots")
    summary_path = Path("outputs/logs/summary.json")

    # Pipeline Execution Trigger
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
            )
        st.sidebar.success("✅ Video Processing Complete!")
        st.rerun()

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
    else:
        total_counted = len(events_df[events_df["event_type"] == "LINE_CROSSING"]) if not events_df.empty else 0
        total_violations = len(events_df[events_df["event_type"] == "WRONG_WAY_VIOLATION"]) if not events_df.empty else 0
        avg_fps = "-- FPS"
        avg_lat = "-- ms"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("🚗 Total Vehicles Counted", total_counted)
    with kpi2:
        st.metric(
            "🚨 Wrong-Way Violations",
            total_violations,
            delta=f"{total_violations} alerts" if total_violations > 0 else None,
            delta_color="inverse",
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
        col_video, col_stats = st.columns([3, 2])

        with col_video:
            st.subheader("Annotated Video Playback")
            if output_video_path.exists():
                st.video(str(output_video_path))
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
