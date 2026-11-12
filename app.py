"""
=============================================================================
AI-Based Emotion Detection System
Streamlit Main Dashboard Application
=============================================================================
"""

import base64
import csv
import datetime
import io
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import streamlit as st

# Safe pandas import with pure Python fallback for strict AppLocker environments
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

from emotion_detector import (
    EmotionDetector,
    EMOTIONS,
    EMOTION_DISPLAY_NAMES,
    EMOTION_ICONS,
    EMOTION_COLORS_BGR,
    EMOTION_COLORS_HEX,
)

# =============================================================================
# STREAMLIT CONFIGURATION & THEMING
# =============================================================================

st.set_page_config(
    page_title="AI-Based Emotion Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CACHED MODEL ENGINE INITIALIZATION
# =============================================================================

@st.cache_resource(show_spinner="Loading Deep Learning Emotion Detector...")
def load_emotion_detector() -> EmotionDetector:
    """Load and cache the EmotionDetector model and cascade in memory."""
    return EmotionDetector()


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "total_faces_analyzed" not in st.session_state:
    st.session_state.total_faces_analyzed = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def image_to_base64_thumbnail(img_bgr: np.ndarray, max_dim: int = 120) -> str:
    """Convert a BGR numpy image to a base64 encoded JPEG thumbnail string."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return ""
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def add_detection_to_history(
    source: str,
    top_emotion: str,
    confidence: float,
    scores: Dict[str, float],
    face_crop: np.ndarray,
    box: Tuple[int, int, int, int],
):
    """Log an emotion detection event into session history."""
    st.session_state.total_faces_analyzed += 1
    record = {
        "id": st.session_state.total_faces_analyzed,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "emotion_key": top_emotion,
        "emotion": EMOTION_DISPLAY_NAMES.get(top_emotion, top_emotion.capitalize()),
        "icon": EMOTION_ICONS.get(top_emotion, "👤"),
        "confidence": round(confidence, 2),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "thumbnail": image_to_base64_thumbnail(face_crop),
        "box": box,
    }
    st.session_state.history.append(record)


def get_history_rows() -> List[Dict[str, Any]]:
    """Convert history records to a flat list of dictionaries for tabular view & CSV."""
    rows = []
    for h in st.session_state.history:
        row = {
            "ID": h["id"],
            "Timestamp": h["timestamp"],
            "Source": h["source"],
            "Emotion": h["emotion"],
            "Confidence (%)": h["confidence"],
        }
        for emo in EMOTIONS:
            row[f"{EMOTION_DISPLAY_NAMES[emo]} (%)"] = h["scores"].get(emo, 0.0)
        rows.append(row)
    return rows


def generate_history_csv() -> bytes:
    """Generate RFC-4180 compliant CSV bytes from session history records."""
    rows = get_history_rows()
    if not rows:
        return b""
    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def apply_custom_theme(theme_choice: str):
    """Apply modern CSS styling tailored to the active theme."""
    is_dark = theme_choice == "🌙 Dark Theme"

    bg_gradient = (
        "linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%)"
        if is_dark
        else "linear-gradient(135deg, rgba(241, 245, 249, 0.9) 0%, rgba(226, 232, 240, 0.9) 100%)"
    )
    card_bg = (
        "linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%)"
        if is_dark
        else "linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)"
    )
    border_color = "rgba(255, 255, 255, 0.1)" if is_dark else "rgba(0, 0, 0, 0.08)"
    text_primary = "#f8fafc" if is_dark else "#0f172a"
    text_secondary = "#94a3b8" if is_dark else "#64748b"

    css = f"""
    <style>
    /* Global Page Styling */
    .app-header {{
        background: {bg_gradient};
        border: 1px solid {border_color};
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, {0.3 if is_dark else 0.05});
        backdrop-filter: blur(10px);
    }}
    .app-title {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }}
    .app-subtitle {{
        color: {text_secondary};
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.5;
    }}
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, {0.2 if is_dark else 0.04});
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 12px;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, {0.35 if is_dark else 0.08});
    }}
    .metric-title {{
        color: {text_secondary};
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .metric-val {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {text_primary};
    }}
    .glass-panel {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, {0.15 if is_dark else 0.03});
    }}
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
    }}
    .emotion-guide-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        height: 100%;
    }}
    .emotion-guide-card:hover {{
        border-color: #38bdf8;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# CHART BUILDERS
# =============================================================================

def plot_emotion_bar_chart(scores: Dict[str, float], title: str = "Emotion Probability Distribution", is_dark: bool = True) -> go.Figure:
    """Create an attractive Plotly horizontal bar chart for all 7 emotions."""
    labels = [EMOTION_DISPLAY_NAMES[e] for e in EMOTIONS]
    values = [scores.get(e, 0.0) for e in EMOTIONS]
    colors = [EMOTION_COLORS_HEX[e] for e in EMOTIONS]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="auto",
            hoverinfo="x+y",
        )
    )

    bg_color = "rgba(15, 23, 42, 0.7)" if is_dark else "#ffffff"
    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.08)"

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=font_color)),
        xaxis=dict(title="Confidence (%)", range=[0, 105], showgrid=True, gridcolor=grid_color, color=font_color),
        yaxis=dict(autorange="reversed", color=font_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color, family="sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig


def plot_session_pie_chart(history_records: List[Dict[str, Any]], is_dark: bool = True) -> go.Figure:
    """Create a donut chart of detected emotion frequencies in the session."""
    emotion_counts: Dict[str, int] = {}
    for h in history_records:
        emo = h["emotion"]
        emotion_counts[emo] = emotion_counts.get(emo, 0) + 1

    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())

    color_map = {EMOTION_DISPLAY_NAMES[k]: EMOTION_COLORS_HEX[k] for k in EMOTIONS}
    slice_colors = [color_map.get(e, "#94a3b8") for e in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=slice_colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            )
        ]
    )

    bg_color = "rgba(15, 23, 42, 0.7)" if is_dark else "#ffffff"
    font_color = "#f8fafc" if is_dark else "#0f172a"

    fig.update_layout(
        title=dict(text="Session Emotion Distribution", font=dict(size=16, color=font_color)),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color, family="sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def plot_session_bar_chart(history_records: List[Dict[str, Any]], is_dark: bool = True) -> go.Figure:
    """Create a column bar chart comparing detection counts per emotion."""
    counts = {EMOTION_DISPLAY_NAMES[e]: 0 for e in EMOTIONS}
    for h in history_records:
        emo = h["emotion"]
        if emo in counts:
            counts[emo] += 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [EMOTION_COLORS_HEX[e] for e in EMOTIONS]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=values,
            textposition="auto",
        )
    )

    bg_color = "rgba(15, 23, 42, 0.7)" if is_dark else "#ffffff"
    font_color = "#f8fafc" if is_dark else "#0f172a"
    grid_color = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.08)"

    fig.update_layout(
        title=dict(text="Emotion Frequency Count", font=dict(size=16, color=font_color)),
        xaxis=dict(title="Emotion", color=font_color),
        yaxis=dict(title="Total Detections", showgrid=True, gridcolor=grid_color, color=font_color),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color, family="sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        height=350,
    )
    return fig


# =============================================================================
# SIDEBAR NAVIGATION & THEME CONTROLS
# =============================================================================

def render_sidebar():
    """Render the sidebar menu and configuration controls."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 20px 0;">
                <span style="font-size: 3rem;">🧠</span>
                <h2 style="margin: 6px 0 0 0; font-weight: 800; font-size: 1.35rem;">EMOVIBE AI</h2>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Emotion Recognition System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("🧭 Navigation")
        page = st.radio(
            "Select Page",
            [
                "🏠 Overview & Guide",
                "📷 Live Webcam Detection",
                "🖼️ Image Upload & Analysis",
                "📊 Emotion Analytics Dashboard",
                "ℹ️ Architecture & Theory",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.subheader("🎨 Appearance")
        theme_mode = st.radio(
            "Select Theme",
            ["🌙 Dark Theme", "☀️ Light Theme"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.subheader("⚙️ System Status")
        st.markdown(
            """
            <div style="font-size: 0.85rem; line-height: 1.8;">
                <div>🟢 <b>Model:</b> Mini-Xception CNN</div>
                <div>📁 <b>Dataset:</b> FER-2013 (7 Classes)</div>
                <div>⚡ <b>Backend:</b> PyTorch / OpenCV</div>
                <div>🎯 <b>Input Size:</b> 64x64 Grayscale</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("AI-Based Emotion Detection System v1.0.0")

        return page, theme_mode


# =============================================================================
# PAGE 1: 🏠 OVERVIEW & GUIDE
# =============================================================================

def render_overview_page(detector: EmotionDetector, is_dark: bool):
    """Render the application overview, statistics, and emotion reference guide."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">AI-Based Emotion Detection System</div>
            <div class="app-subtitle">
                Real-time human facial expression recognition powered by deep convolutional neural networks and OpenCV.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Metric Highlights
    total_det = len(st.session_state.history)
    most_freq = "None"
    avg_conf = 0.0
    if total_det > 0:
        counts: Dict[str, int] = {}
        conf_sum = 0.0
        for h in st.session_state.history:
            emo = h["emotion"]
            counts[emo] = counts.get(emo, 0) + 1
            conf_sum += h["confidence"]
        most_freq = max(counts, key=counts.get)
        avg_conf = conf_sum / total_det

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Detections</div>
                <div class="metric-val">{total_det}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Top Emotion</div>
                <div class="metric-val">{most_freq}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Avg Confidence</div>
                <div class="metric-val">{avg_conf:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Classes Recognized</div>
                <div class="metric-val">7 Emotions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎭 Supported Emotion Categories")
    st.write("The system detects and differentiates between seven fundamental human facial expressions:")

    emotion_info = [
        {"name": "Happy", "icon": "😄", "key": "happy", "desc": "Smiling mouth, cheek raising, crow's feet wrinkles around eyes."},
        {"name": "Sad", "icon": "😢", "key": "sad", "desc": "Drooping upper eyelids, downward mouth corners, inner brow pulled up."},
        {"name": "Angry", "icon": "😠", "key": "angry", "desc": "Eyebrows pulled down and together, glaring eyes, narrowed lips."},
        {"name": "Fear", "icon": "😨", "key": "fear", "desc": "Raised upper eyelids, eyebrows drawn up and together, lips stretched."},
        {"name": "Surprise", "icon": "😲", "key": "surprise", "desc": "Eyebrows raised and curved, wide open eyes, dropped jaw."},
        {"name": "Neutral", "icon": "😐", "key": "neutral", "desc": "Relaxed facial muscles, natural resting expression without tension."},
        {"name": "Disgust", "icon": "🤢", "key": "disgust", "desc": "Wrinkled nose bridge, raised upper lip, narrowed eyes."},
    ]

    cols = st.columns(7)
    for i, emo in enumerate(emotion_info):
        with cols[i]:
            hex_color = EMOTION_COLORS_HEX[emo["key"]]
            st.markdown(
                f"""
                <div class="emotion-guide-card" style="border-top: 4px solid {hex_color};">
                    <div style="font-size: 2.2rem; margin-bottom: 6px;">{emo['icon']}</div>
                    <div style="font-weight: 700; color: {hex_color}; font-size: 1.05rem;">{emo['name']}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 6px; line-height: 1.3;">
                        {emo['desc']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🚀 Quick Start Guide")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="glass-panel">
                <h4>📷 Mode 1: Live Webcam Detection</h4>
                <ol style="color: #94a3b8; font-size: 0.95rem; line-height: 1.7;">
                    <li>Navigate to <b>Live Webcam Detection</b> in the sidebar.</li>
                    <li>Choose either <b>Real-Time Live Stream</b> (continuous video) or <b>Snapshot Camera</b>.</li>
                    <li>Click <b>Start Live Camera</b> to begin continuous real-time recognition.</li>
                    <li>The system tracks bounding boxes and displays emotion percentages dynamically.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="glass-panel">
                <h4>🖼️ Mode 2: Image Upload Analysis</h4>
                <ol style="color: #94a3b8; font-size: 0.95rem; line-height: 1.7;">
                    <li>Navigate to <b>Image Upload & Analysis</b> in the sidebar.</li>
                    <li>Upload any <b>JPG, JPEG, or PNG</b> image, or select a pre-loaded sample.</li>
                    <li>The system locates all faces and extracts 64x64 facial crops.</li>
                    <li>View annotated images, confidence meters, and full 7-emotion probability charts.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: 📷 LIVE WEBCAM DETECTION
# =============================================================================

def render_webcam_page(detector: EmotionDetector, is_dark: bool):
    """Render the live webcam detection page with dual modes (Real-Time and Snapshot)."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">📷 Live Webcam Detection</div>
            <div class="app-subtitle">
                Access your computer webcam to detect faces and classify facial expressions in real-time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    webcam_tab1, webcam_tab2 = st.tabs(["🎥 Real-Time Continuous Stream", "📸 Browser Snapshot Camera"])

    # -------------------------------------------------------------------------
    # TAB 1: Real-Time Stream (OpenCV cv2.VideoCapture)
    # -------------------------------------------------------------------------
    with webcam_tab1:
        st.markdown(
            """
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Continuous video loop using OpenCV VideoCapture. Connects directly to your local camera hardware.
            </p>
            """,
            unsafe_allow_html=True,
        )

        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1.5, 1.5, 2])
        with c_ctrl1:
            cam_idx = st.selectbox("Select Camera Device", options=[0, 1, 2], format_func=lambda x: f"Camera {x} (Default)" if x == 0 else f"Camera {x}")
        with c_ctrl2:
            conf_threshold = st.slider("Min Confidence Filter (%)", min_value=10, max_value=90, value=30, step=5)
        with c_ctrl3:
            st.write("Controls")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_live = st.button("▶️ Start Camera", type="primary", use_container_width=True)
            with btn_col2:
                stop_live = st.button("⏹️ Stop Camera", use_container_width=True)

        if start_live:
            st.session_state.webcam_running = True
        if stop_live:
            st.session_state.webcam_running = False

        video_placeholder = st.empty()
        status_bar = st.empty()

        if st.session_state.webcam_running:
            try:
                cap = cv2.VideoCapture(cam_idx)
                if not cap.isOpened():
                    st.session_state.webcam_running = False
                    st.error(
                        f"❌ Could not access Camera {cam_idx}. The camera may be in use by another application "
                        "or permissions may be restricted. You can also try the 'Browser Snapshot Camera' tab!"
                    )
                else:
                    # Video stream loop
                    prev_time = time.time()
                    last_logged_time = 0.0

                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret:
                            status_bar.warning("⚠️ Failed to grab frame from webcam. Stream stopped.")
                            break

                        curr_time = time.time()
                        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
                        prev_time = curr_time

                        # Process frame with EmotionDetector
                        annotated_frame, detections = detector.process_image(frame, draw_annotations=True)

                        # Filter and log detections periodically (once every 1.5s to avoid flooding history)
                        if detections and (curr_time - last_logged_time > 1.5):
                            for d in detections:
                                if d["confidence"] >= conf_threshold:
                                    add_detection_to_history(
                                        source="Live Webcam",
                                        top_emotion=d["emotion"],
                                        confidence=d["confidence"],
                                        scores=d["scores"],
                                        face_crop=d["face_crop"],
                                        box=d["box"],
                                    )
                            last_logged_time = curr_time

                        # Add live FPS watermark
                        cv2.putText(
                            annotated_frame,
                            f"FPS: {fps:.1f} | Faces: {len(detections)}",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 200),
                            2,
                            cv2.LINE_AA,
                        )

                        # Convert BGR to RGB for Streamlit rendering
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        # Display live status metadata
                        if detections:
                            top_d = max(detections, key=lambda x: x["confidence"])
                            badge_color = top_d["color_hex"]
                            status_bar.markdown(
                                f"""
                                <div style="display: flex; gap: 16px; align-items: center; padding: 8px 14px; background: rgba(30, 41, 59, 0.5); border-radius: 10px;">
                                    <span>🟢 <b>Streaming</b> ({fps:.1f} FPS)</span>
                                    <span>👥 <b>Detected Faces:</b> {len(detections)}</span>
                                    <span>🎭 <b>Top Emotion:</b> <span class="badge-pill" style="background-color: {badge_color};">{top_d['icon']} {top_d['display_name']} ({top_d['confidence']:.1f}%)</span></span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            status_bar.markdown("🟢 *Streaming active. Looking for human faces in front of camera...*")

                    cap.release()
                    video_placeholder.empty()
                    status_bar.info("Camera stopped. Ready to start again.")

            except Exception as e:
                st.session_state.webcam_running = False
                st.error(f"Error accessing camera: {e}")
        else:
            video_placeholder.info("Click **'▶️ Start Camera'** to launch the live video emotion detection stream.")

    # -------------------------------------------------------------------------
    # TAB 2: Browser Snapshot Camera (HTML5 st.camera_input)
    # -------------------------------------------------------------------------
    with webcam_tab2:
        st.markdown(
            """
            <p style="color: #94a3b8; font-size: 0.95rem;">
                Capture a high-resolution snapshot directly through your browser's camera.
                Works in all web browsers and remote environments without device driver locks.
            </p>
            """,
            unsafe_allow_html=True,
        )

        camera_photo = st.camera_input("Capture Face Snapshot")
        if camera_photo is not None:
            # Read image bytes
            bytes_data = camera_photo.getvalue()
            pil_image = Image.open(io.BytesIO(bytes_data))
            bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing facial expressions..."):
                annotated_img, detections = detector.process_image(bgr_image, draw_annotations=True)

            if not detections:
                st.warning("⚠️ No face was detected in this snapshot. Please look directly at the camera and ensure good lighting.")
            else:
                st.success(f"✅ Successfully detected {len(detections)} face(s) in snapshot!")

                # Log all detections
                for d in detections:
                    add_detection_to_history(
                        source="Camera Snapshot",
                        top_emotion=d["emotion"],
                        confidence=d["confidence"],
                        scores=d["scores"],
                        face_crop=d["face_crop"],
                        box=d["box"],
                    )

                c1, c2 = st.columns([1.2, 1])
                with c1:
                    st.subheader("Annotated Snapshot")
                    rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(rgb_annotated, channels="RGB", use_container_width=True)

                with c2:
                    st.subheader("Emotion Analysis Breakdown")
                    for idx, d in enumerate(detections):
                        hex_c = d["color_hex"]
                        st.markdown(
                            f"""
                            <div class="glass-panel" style="border-left: 5px solid {hex_c};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-weight: 700; font-size: 1.1rem;">Face #{idx + 1}</span>
                                    <span class="badge-pill" style="background-color: {hex_c};">
                                        {d['icon']} {d['display_name']} ({d['confidence']:.1f}%)
                                    </span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        fig = plot_emotion_bar_chart(d["scores"], title=f"Face #{idx + 1} Probabilities", is_dark=is_dark)
                        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 3: 🖼️ IMAGE UPLOAD & ANALYSIS
# =============================================================================

def render_image_upload_page(detector: EmotionDetector, is_dark: bool):
    """Render the image upload detection page with sample image presets."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🖼️ Image Upload & Analysis</div>
            <div class="app-subtitle">
                Upload portrait photos or group pictures (JPG, JPEG, PNG) to detect all faces and classify expressions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input1, col_input2 = st.columns([2, 1.2])

    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload an image file (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Select a clear portrait or group photo with visible human faces.",
        )

    with col_input2:
        st.markdown("**Or Test With a Preloaded Sample:**")
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sample_images")
        sample_options = ["None"]
        if os.path.exists(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            sample_options.extend(sample_files)

        selected_sample = st.selectbox(
            "Choose Sample Image",
            options=sample_options,
            format_func=lambda x: "Select sample..." if x == "None" else x.replace("_", " ").replace(".jpg", "").capitalize(),
        )

    # Resolve target image
    target_bgr: Optional[np.ndarray] = None
    image_name = "Uploaded Image"

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        target_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image_name = uploaded_file.name
    elif selected_sample != "None":
        sample_path = os.path.join(sample_dir, selected_sample)
        target_bgr = cv2.imread(sample_path)
        image_name = selected_sample

    if target_bgr is not None:
        st.markdown("---")
        with st.spinner("Processing image through Deep Learning Emotion Pipeline..."):
            annotated_img, detections = detector.process_image(target_bgr, draw_annotations=True)

        if not detections:
            st.error(
                "❌ **No human faces detected in this image.**\n\n"
                "**Tips for best accuracy:**\n"
                "- Ensure the face is facing forward toward the camera.\n"
                "- Avoid strong motion blur, extreme angles, or heavy occlusions.\n"
                "- Ensure sufficient lighting across the facial features."
            )
            # Display original image anyway
            st.image(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB), caption="Original Image (No face detected)", use_container_width=True)
        else:
            st.success(f"🎉 Successfully detected **{len(detections)} face(s)** in `{image_name}`!")

            # Log detections
            for d in detections:
                add_detection_to_history(
                    source=f"Upload ({image_name})",
                    top_emotion=d["emotion"],
                    confidence=d["confidence"],
                    scores=d["scores"],
                    face_crop=d["face_crop"],
                    box=d["box"],
                )

            # Two-column layout: Annotated Image on left, Detailed Breakdown on right
            c_left, c_right = st.columns([1.3, 1.2])

            with c_left:
                st.subheader("🖼️ Detection Overview")
                rgb_annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(rgb_annotated, channels="RGB", use_container_width=True)

            with c_right:
                st.subheader("📊 Individual Face Results")

                for idx, d in enumerate(detections):
                    hex_color = d["color_hex"]
                    st.markdown(
                        f"""
                        <div class="glass-panel" style="border-left: 6px solid {hex_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span style="font-weight: 700; font-size: 1.15rem;">Face #{idx + 1}</span>
                                <span class="badge-pill" style="background-color: {hex_color};">
                                    {d['icon']} {d['display_name']} ({d['confidence']:.1f}%)
                                </span>
                            </div>
                            <div style="font-size: 0.85rem; color: #94a3b8;">
                                Bounding Box: <code>x={d['box'][0]}, y={d['box'][1]}, w={d['box'][2]}, h={d['box'][3]}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Sub-columns: face crop thumbnail + Plotly probabilities
                    sub_c1, sub_c2 = st.columns([1, 2.5])
                    with sub_c1:
                        crop_rgb = cv2.cvtColor(d["face_crop"], cv2.COLOR_BGR2RGB)
                        st.image(crop_rgb, caption=f"Crop #{idx + 1}", use_container_width=True)
                    with sub_c2:
                        fig = plot_emotion_bar_chart(
                            d["scores"],
                            title=f"Face #{idx + 1} Emotion Confidence",
                            is_dark=is_dark,
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Please upload an image file or choose one of the pre-loaded sample images above to begin detection.")


# =============================================================================
# PAGE 4: 📊 EMOTION ANALYTICS DASHBOARD
# =============================================================================

def render_analytics_page(is_dark: bool):
    """Render the session analytics dashboard with metrics, charts, and CSV export."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">📊 Emotion Analysis Dashboard</div>
            <div class="app-subtitle">
                Aggregate metrics, emotional distribution breakdowns, and session detection logs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history_records = st.session_state.history

    if not history_records:
        st.info(
            "ℹ️ **No emotion detections recorded in this session yet.**\n\n"
            "Run detections via **Live Webcam** or **Image Upload** to view analytical charts and metrics."
        )
        return

    # Key Performance Metric Cards
    total_faces = len(history_records)
    counts: Dict[str, int] = {}
    conf_sum = 0.0
    highest_conf = 0.0

    for h in history_records:
        emo = h["emotion"]
        counts[emo] = counts.get(emo, 0) + 1
        conf_sum += h["confidence"]
        if h["confidence"] > highest_conf:
            highest_conf = h["confidence"]

    most_freq_emotion = max(counts, key=counts.get) if counts else "None"
    avg_confidence = (conf_sum / total_faces) if total_faces > 0 else 0.0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Faces Detected</div>
                <div class="metric-val">{total_faces}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dominant Emotion</div>
                <div class="metric-val">{most_freq_emotion}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Average Confidence</div>
                <div class="metric-val">{avg_confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Peak Confidence</div>
                <div class="metric-val">{highest_conf:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Section
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        pie_fig = plot_session_pie_chart(history_records, is_dark=is_dark)
        st.plotly_chart(pie_fig, use_container_width=True)

    with ch_col2:
        bar_fig = plot_session_bar_chart(history_records, is_dark=is_dark)
        st.plotly_chart(bar_fig, use_container_width=True)

    # Session Timeline / Distribution Table
    st.markdown("---")
    st.subheader("📋 Detection History Log")

    # Action buttons: Download CSV & Clear History
    act_col1, act_col2, act_col3 = st.columns([1.5, 1.5, 3])
    with act_col1:
        csv_bytes = generate_history_csv()
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_bytes,
            file_name=f"emotion_detection_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with act_col2:
        if st.button("🗑️ Clear Session History", use_container_width=True):
            st.session_state.history = []
            st.session_state.total_faces_analyzed = 0
            st.rerun()

    # Formatted display table
    table_rows = get_history_rows()
    if HAS_PANDAS:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=320)
    else:
        st.dataframe(table_rows, use_container_width=True, height=320)


# =============================================================================
# PAGE 5: ℹ️ ARCHITECTURE & THEORY
# =============================================================================

def render_theory_page():
    """Render educational deep-dive explaining CNNs, FER-2013, and the system pipeline."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">ℹ️ System Architecture & Theory</div>
            <div class="app-subtitle">
                Comprehensive explanation of deep convolutional neural networks, facial emotion recognition, and the system pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns([1.5, 1])

    with t_col1:
        st.markdown(
            """
            ### 1. How Facial Emotion Recognition Works
            Facial Emotion Recognition (FER) is an end-to-end computer vision and deep learning workflow that analyzes facial muscle movements to infer psychological affective states.

            The processing pipeline operates in five distinct phases:
            1. **Face Localization:**
               An input image is converted to grayscale, and contrast is normalized using histogram equalization. An OpenCV Haar Feature-based Cascade Classifier scans the image pyramid to localize candidate bounding boxes around human faces.
            2. **Face Extraction & Aspect Ratio Preservation:**
               Each detected bounding box is expanded with a 10% padding margin to preserve forehead and chin boundary cues. The bounding box is made square to eliminate geometric distortion.
            3. **Input Standardization:**
               The cropped face is resized to a standardized **64x64 pixel matrix** and normalized to zero-centered floating point values in the range `[-1.0, +1.0]`.
            4. **Deep Feature Extraction (Mini-Xception):**
               The normalized tensor passes through 4 residual blocks featuring **Depthwise Separable Convolutions**. Spatial features (corners of lips, eyebrow furrows, eyelid contours) are extracted hierarchically without parameter bloat.
            5. **Classification & Softmax Distribution:**
               Global Average Pooling aggregates feature maps into a 128-dimensional latent vector, followed by a 1x1 convolution mapping to 7 emotion logits and a **Softmax layer** calculating the exact probability distribution across all 7 emotions.
            """
        )

    with t_col2:
        st.markdown(
            """
            <div class="glass-panel">
                <h4>📐 Technical Specifications</h4>
                <ul style="color: #94a3b8; font-size: 0.9rem; line-height: 1.8;">
                    <li><b>Architecture:</b> Mini-Xception Residual CNN</li>
                    <li><b>Number of Parameters:</b> ~60,000 (Lightweight)</li>
                    <li><b>Inference Latency:</b> &lt; 5 ms per face</li>
                    <li><b>Input Dimensions:</b> 64 &times; 64 &times; 1 (Grayscale)</li>
                    <li><b>Output Classes:</b> 7 Categories</li>
                    <li><b>Activation:</b> ReLU & Softmax</li>
                    <li><b>Training Dataset:</b> FER-2013 (35,887 images)</li>
                    <li><b>Deep Learning Engine:</b> PyTorch 2.15+ / OpenCV 5.0</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🔬 Deep Learning Architecture: Mini-Xception")

    st.markdown(
        """
        Conventional deep networks like VGG-16 or ResNet-50 contain tens of millions of parameters, which makes them unsuitable for real-time webcam inference at 30+ FPS on standard CPU hardware.

        The **Mini-Xception** architecture solves this problem by combining two modern deep learning innovations:
        - **Depthwise Separable Convolutions:** Decouples spatial cross-channel filtering into a depthwise convolution (applying 1 filter per input channel) followed by a pointwise convolution (1x1 linear combination across channels). This reduces computational cost by **8-9x** with virtually no loss in classification accuracy.
        - **Residual Connections:** Shortcut identity skip-connections allow gradient flow directly across layers, mitigating the vanishing gradient problem and accelerating model convergence.
        - **Batch Normalization:** Stabilizes hidden layer distributions, enabling faster inference and regularizing against overfitting.
        """
    )

    st.markdown("---")
    st.subheader("📚 The FER-2013 Dataset")
    st.markdown(
        """
        The model was trained on the landmark **FER-2013 dataset**, introduced at the ICML 2013 Challenges in Representation Learning:
        - **Total Images:** 35,887 labeled face images.
        - **Image Composition:** 28,709 training images, 3,589 validation images, and 3,589 test images.
        - **Diversity:** High variation in subject age, ethnicity, facial hair, glasses, head pose, and lighting conditions, making the detector resilient across diverse real-world webcam conditions.
        """
    )


# =============================================================================
# MAIN APPLICATION DISPATCHER
# =============================================================================

def main():
    """Main application loop."""
    # Render Sidebar and retrieve page + theme selection
    selected_page, theme_choice = render_sidebar()

    # Apply responsive CSS styling based on selected theme
    is_dark = theme_choice == "🌙 Dark Theme"
    apply_custom_theme(theme_choice)

    # Initialize / Retrieve cached EmotionDetector engine
    try:
        detector = load_emotion_detector()
    except Exception as e:
        st.error(f"Failed to load Emotion Detection Engine: {e}")
        st.stop()

    # Route to selected page
    if selected_page == "🏠 Overview & Guide":
        render_overview_page(detector, is_dark)
    elif selected_page == "📷 Live Webcam Detection":
        render_webcam_page(detector, is_dark)
    elif selected_page == "🖼️ Image Upload & Analysis":
        render_image_upload_page(detector, is_dark)
    elif selected_page == "📊 Emotion Analytics Dashboard":
        render_analytics_page(is_dark)
    elif selected_page == "ℹ️ Architecture & Theory":
        render_theory_page()


if __name__ == "__main__":
    main()
