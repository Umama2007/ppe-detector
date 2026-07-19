"""
app.py - Streamlit web interface for SafeGuard AI (YOLOv11 PPE Detection).
Fully redesigned with premium SaaS dashboard aesthetics (Vercel/OpenAI style).
"""

import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ----------------- Session State Theme -----------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ----------------- Configuration & CSS Injection -----------------
st.set_page_config(
    page_title="SafeGuard AI | Enterprise Safety Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Colors definitions for Light/Dark themes
if st.session_state.theme == "light":
    bg_color = "#F8FAFC"
    card_bg = "#FFFFFF"
    text_color = "#0F172A"
    sub_text = "#64748B"
    border_color = "#E2E8F0"
    card_shadow = "rgba(15, 23, 42, 0.04)"
    hero_grad = "linear-gradient(135deg, #FFF0F2 0%, #FFF5F2 100%)"
    glow_color = "rgba(255, 90, 95, 0.06)"
else:
    bg_color = "#0B0F19"
    card_bg = "#111827"
    text_color = "#F9FAFB"
    sub_text = "#9CA3AF"
    border_color = "#1F2937"
    card_shadow = "rgba(0, 0, 0, 0.4)"
    hero_grad = "linear-gradient(135deg, #1E1B29 0%, #111827 100%)"
    glow_color = "rgba(255, 90, 95, 0.15)"

# SVGs for custom dashboard indicators
HELMET_SVG = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="css-i6dzq1"><path d="M2 10s3-3 10-3 10 3 10 3M21.5 10v4a1 1 0 0 1-1 1h-17a1 1 0 0 1-1-1v-4M10 7V3a2 2 0 0 1 4 0v4"></path></svg>'
VEST_SVG = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="css-i6dzq1"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4zM3 6h18M16 10a4 4 0 0 1-8 0"></path></svg>'
PEOPLE_SVG = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="css-i6dzq1"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>'
ALERT_SVG = '<svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round" class="css-i6dzq1"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0zM12 9v4M12 17h.01"></path></svg>'

# Inject Custom SaaS CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    
    /* Hide Default Streamlit Style Elements */
    #MainMenu, footer, header, [data-testid="stHeader"], [data-testid="stDeployButton"] {{
        display: none !important;
        height: 0 !important;
        width: 0 !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg_color};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {border_color};
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #FF5A5F;
    }}
    
    /* Layout Header Container */
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        border-bottom: 1px solid {border_color};
        margin-bottom: 2rem;
    }}
    
    .header-logo {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .header-title-text {{
        font-weight: 800;
        font-size: 1.5rem;
        background: linear-gradient(90deg, #FF5A5F 0%, #FF8A65 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }}
    
    .header-right {{
        display: flex;
        align-items: center;
        gap: 20px;
    }}
    
    .clock {{
        font-size: 0.95rem;
        font-weight: 600;
        color: {sub_text};
        background: {border_color};
        padding: 6px 14px;
        border-radius: 20px;
    }}
    
    .avatar {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #FF5A5F 0%, #FF8A65 100%);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        border-radius: 50%;
        box-shadow: 0 4px 10px rgba(255, 90, 95, 0.3);
    }}
    
    /* Hero Section Card */
    .hero-card {{
        background: {hero_grad};
        border: 1px solid {border_color};
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        display: grid;
        grid-template-columns: 1.5fr 1fr;
        align-items: center;
        gap: 20px;
        box-shadow: 0 10px 40px {card_shadow};
        position: relative;
        overflow: hidden;
    }}
    
    .hero-card::before {{
        content: '';
        position: absolute;
        width: 150px;
        height: 150px;
        background: #FF5A5F;
        filter: blur(120px);
        top: -50px;
        right: -50px;
        border-radius: 50%;
        opacity: 0.2;
    }}
    
    .hero-badge {{
        background: rgba(255, 90, 95, 0.1);
        color: #FF5A5F;
        padding: 6px 14px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 90, 95, 0.2);
    }}
    
    .hero-heading {{
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }}
    
    .hero-desc {{
        color: {sub_text};
        font-size: 1.1rem;
        line-height: 1.5;
        max-width: 550px;
    }}
    
    .hero-right-illustration {{
        display: flex;
        justify-content: center;
        align-items: center;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    /* Sidebar Overrides & Redesigns */
    [data-testid="stSidebar"] {{
        background-color: {card_bg} !important;
        border-right: 1px solid {border_color} !important;
    }}
    
    .sidebar-card {{
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }}
    
    .sidebar-card-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        color: {sub_text};
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }}
    
    .sidebar-model-path {{
        font-size: 0.9rem;
        font-weight: 600;
        word-break: break-all;
        color: {text_color};
        font-family: monospace;
    }}
    
    /* Hardware status block */
    .hw-status-container {{
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.05);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    
    .hw-title {{
        font-size: 0.9rem;
        font-weight: 700;
        color: #10B981;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .hw-badge-glow {{
        width: 10px;
        height: 10px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.4; }}
    }}
    
    /* Streamlit Custom Tab styling */
    div[data-baseweb="tab-list"] {{
        gap: 15px;
        background-color: {card_bg};
        padding: 8px 12px;
        border-radius: 100px;
        border: 1px solid {border_color};
        display: inline-flex;
        box-shadow: 0 4px 15px {card_shadow};
        margin-bottom: 2rem;
    }}
    
    button[data-baseweb="tab"] {{
        border-radius: 100px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: {sub_text} !important;
        background-color: transparent !important;
        transition: all 0.25s ease !important;
    }}
    
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #FF5A5F 0%, #FF8A65 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(255, 90, 95, 0.3) !important;
    }}
    
    /* Drag & Drop Card styling */
    div[data-testid="stFileUploader"] {{
        background-color: {card_bg} !important;
        border: 2px dashed #FF5A5F !important;
        border-radius: 20px !important;
        padding: 40px 20px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px {card_shadow} !important;
    }}
    
    div[data-testid="stFileUploader"]:hover {{
        border-color: #FF8A65 !important;
        box-shadow: 0 15px 40px {glow_color} !important;
        transform: translateY(-2px);
    }}
    
    div[data-testid="stFileUploader"] label {{
        display: none !important;
    }}
    
    div[data-testid="stFileUploader"] section > div {{
        display: none !important;
    }}
    
    div[data-testid="stFileUploader"] section::before {{
        content: "⬆\\n\\nDrop File Here to Analyze\\n(PNG, JPG, JPEG, WEBP)";
        white-space: pre-wrap;
        font-size: 1.15rem;
        font-weight: 700;
        color: {text_color};
        margin-bottom: 20px;
        display: block;
        line-height: 1.5;
    }}
    
    div[data-testid="stFileUploader"] button {{
        background: linear-gradient(135deg, #FF5A5F 0%, #FF8A65 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 100px !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(255, 90, 95, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    
    div[data-testid="stFileUploader"] button:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(255, 90, 95, 0.4) !important;
    }}
    
    /* Result Frame Containers */
    .frame-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 10px 35px {card_shadow};
        transition: transform 0.3s ease;
    }}
    
    .frame-card:hover {{
        transform: scale(1.01);
    }}
    
    /* Compliance indicator lists */
    .summary-item {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 18px;
        border-radius: 12px;
        background: {bg_color};
        border: 1px solid {border_color};
        margin-bottom: 8px;
        font-weight: 600;
    }}
    
    .status-badge {{
        padding: 4px 12px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    
    .status-badge-ok {{
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
    }}
    
    .status-badge-err {{
        background: rgba(239, 68, 68, 0.12);
        color: #EF4444;
    }}
    
    /* Custom Slider Customizations */
    div[data-testid="stSlider"] [data-baseweb="slider"] {{
        height: 6px !important;
        background: linear-gradient(90deg, #FF5A5F 0%, #FF8A65 100%) !important;
        border-radius: 10px !important;
    }}
    
    div[data-testid="stSlider"] [role="slider"] {{
        background-color: #FFFFFF !important;
        border: 2px solid #FF5A5F !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
        width: 18px !important;
        height: 18px !important;
    }}
    
    /* Analytics & Alert Widgets */
    .analytics-card {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px {card_shadow};
        position: relative;
        overflow: hidden;
        margin-bottom: 1.2rem;
    }}
    
    .analytics-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }}
    
    .analytics-title {{
        font-size: 0.9rem;
        font-weight: 700;
        color: {sub_text};
        text-transform: uppercase;
    }}
    
    .analytics-icon {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .analytics-val {{
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
    }}
    
    /* Timeline styles */
    .timeline-container {{
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px {card_shadow};
    }}
    
    .timeline-item {{
        display: flex;
        gap: 15px;
        position: relative;
        padding-bottom: 1.5rem;
    }}
    
    .timeline-item::after {{
        content: '';
        position: absolute;
        left: 20px;
        top: 24px;
        bottom: 0;
        width: 2px;
        background: {border_color};
    }}
    
    .timeline-item:last-child::after {{
        display: none;
    }}
    
    .timeline-time {{
        font-size: 0.85rem;
        font-weight: 700;
        color: {sub_text};
        width: 45px;
        padding-top: 2px;
    }}
    
    .timeline-indicator {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FF5A5F;
        border: 3px solid {card_bg};
        box-shadow: 0 0 0 3px #FF5A5F;
        position: relative;
        z-index: 2;
        margin-top: 6px;
    }}
    
    .timeline-content {{
        font-size: 0.95rem;
        font-weight: 600;
    }}
    
    .timeline-desc {{
        font-size: 0.85rem;
        color: {sub_text};
        margin-top: 2px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Hero Vector Officer SVG -----------------
HERO_VECTOR_SVG = """
<svg viewBox="0 0 200 200" width="80%" height="80%" style="animation: float 6s ease-in-out infinite;">
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FF8A65" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#FF5A5F" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="helm-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFD200"/>
      <stop offset="100%" stop-color="#FFAA00"/>
    </linearGradient>
    <linearGradient id="vest-grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF5A5F"/>
      <stop offset="100%" stop-color="#FF8A65"/>
    </linearGradient>
  </defs>
  <circle cx="100" cy="100" r="90" fill="url(#glow)"/>
  <path d="M100 20 L160 45 V100 C160 145 100 180 100 180 C100 180 40 145 40 100 V45 Z" fill="none" stroke="#FF5A5F" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  
  <!-- Safety Helmet -->
  <path d="M78 80 C78 55 122 55 122 80 Z" fill="url(#helm-grad)"/>
  <rect x="72" y="78" width="56" height="5" rx="2" fill="#FFAA00"/>
  
  <!-- Human Face -->
  <circle cx="100" cy="98" r="14" fill="#FFE0BD"/>
  <path d="M96 102 Q100 106 104 102" stroke="#D4A373" stroke-width="2" fill="none" stroke-linecap="round"/>
  
  <!-- Vest Body -->
  <path d="M75 120 L125 120 L135 160 L65 160 Z" fill="url(#vest-grad)"/>
  
  <!-- Silver Safety Strips -->
  <rect x="85" y="120" width="7" height="40" fill="#E2E8F0"/>
  <rect x="108" y="120" width="7" height="40" fill="#E2E8F0"/>
  <rect x="65" y="138" width="70" height="7" fill="#E2E8F0"/>
</svg>
"""

# ----------------- Utility Analytics Renderers -----------------
def render_header():
    """Renders visual header card with avatar & real-time clock scripting."""
    st.markdown(
        f"""
        <div class="header-container">
            <div class="header-logo">
                <span style="font-size: 2rem;">🛡️</span>
                <span class="header-title-text">SafeGuard AI</span>
            </div>
            <div class="header-right">
                <div class="clock" id="live-clock">--:--:-- AM</div>
                <div class="avatar">AI</div>
            </div>
        </div>
        
        <script>
        const runClock = () => {{
            const now = new Date();
            let hours = now.getHours();
            let minutes = now.getMinutes();
            let seconds = now.getSeconds();
            const ampm = hours >= 12 ? 'PM' : 'AM';
            hours = hours % 12;
            hours = hours ? hours : 12;
            minutes = minutes < 10 ? '0'+minutes : minutes;
            seconds = seconds < 10 ? '0'+seconds : seconds;
            const clockEl = document.getElementById('live-clock');
            if (clockEl) clockEl.innerText = hours + ':' + minutes + ':' + seconds + ' ' + ampm;
        }};
        setInterval(runClock, 1000);
        runClock();
        </script>
        """,
        unsafe_allow_html=True
    )


def render_hero_section():
    """Renders gradient SaaS landing hero."""
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-left">
                <span class="hero-badge">YOLOv11 Protection Core</span>
                <div class="hero-heading">Real-Time PPE Detection & Workplace Compliance</div>
                <div class="hero-desc">
                    Instantly identify helmets, safety vests, boots and compliance levels in production environments using advanced, low-latency computer vision intelligence.
                </div>
            </div>
            <div class="hero-right-illustration">
                {HERO_VECTOR_SVG}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_analytics_card(title, value, color, icon_svg):
    """Renders a single premium Vercel-style KPI card."""
    border_col = "#10B981" if color == "green" else "#EF4444" if color == "red" else "#3B82F6"
    icon_bg = "rgba(16, 185, 129, 0.1)" if color == "green" else "rgba(239, 68, 68, 0.1)" if color == "red" else "rgba(59, 130, 246, 0.1)"
    icon_col = "#10B981" if color == "green" else "#EF4444" if color == "red" else "#3B82F6"
    
    st.markdown(
        f"""
        <div class="analytics-card" style="border-top: 4px solid {border_col};">
            <div class="analytics-card-header">
                <div class="analytics-title">{title}</div>
                <div class="analytics-icon" style="background: {icon_bg}; color: {icon_col};">
                    {icon_svg}
                </div>
            </div>
            <div class="analytics-val">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_timeline_feed(timeline_events):
    """Renders visual checklist timeline of safety events."""
    st.markdown('<div class="timeline-container"><h4>📋 Inspection History Feed</h4>', unsafe_allow_html=True)
    for event in timeline_events:
        time_str, title, desc, icon_color = event
        st.markdown(
            f"""
            <div class="timeline-item">
                <div class="timeline-time">{time_str}</div>
                <div class="timeline-indicator" style="background: {icon_color}; box-shadow: 0 0 0 3px {icon_color}40;"></div>
                <div class="timeline-content">
                    <div>{title}</div>
                    <div class="timeline-desc">{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)


def type_breakdown(model, boxes) -> Counter:
    counts = Counter()
    for box in boxes:
        cls = int(box.cls[0])
        counts[model.names[cls]] += 1
    return counts


def find_default_weights() -> str:
    import glob
    import os
    paths = glob.glob("runs/detect/runs/train/*/weights/best.pt") + glob.glob("runs/train/*/weights/best.pt")
    if paths:
        paths.sort(key=os.path.getmtime, reverse=True)
        return str(Path(paths[0]))
    return "yolo11n.pt"


# ----------------- Main Core App Execution -----------------
def main():
    # Render layout elements
    render_header()
    render_hero_section()

    # Redesigned Configuration Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        
        # Theme Selector
        theme_opt = st.selectbox("🌓 Dashboard Theme", ["Light Mode", "Dark Mode"], index=0 if st.session_state.theme == "light" else 1)
        st.session_state.theme = "light" if theme_opt == "Light Mode" else "dark"
        
        # Model configuration card
        weights_path = st.text_input("Active Weight File (.pt)", value=find_default_weights())
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-title">Active Model weights</div>
                <div class="sidebar-model-path">{Path(weights_path).name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence Slider
        conf_thresh = st.slider("Min Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
        st.markdown(f"**Confidence Level:** `{int(conf_thresh * 100)}%`")

        st.markdown("---")
        
        # Glowing status card
        st.markdown(
            """
            <div class="hw-status-container">
                <div class="hw-title">
                    <span class="hw-badge-glow"></span>
                    CPU Engine
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #10B981; font-weight: 800; letter-spacing: 0.5px;">ONLINE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Load Model Checkpoint
    try:
        model = load_model(weights_path)
    except Exception as e:
        st.error(f"Could not load weights at path: {e}")
        return

    # Visualizer Tabs
    tab_image, tab_video, tab_webcam = st.tabs(["○ Image Analyzer", "○ Video Stream", "○ Live Webcam"])

    # ---------------- Image Analyzer tab ----------------
    with tab_image:
        uploaded_image = st.file_uploader("Upload Inspection Frame", type=["jpg", "jpeg", "png", "bmp", "webp"])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            
            # Run Model & capture inference speed
            start_time = time.time()
            results = model.predict(source=image, conf=conf_thresh, verbose=False)
            latency = (time.time() - start_time) * 1000
            
            r = results[0]
            annotated = r.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Page layout split (Detections vs Analytics)
            col_res, col_an = st.columns([2, 1])
            
            with col_res:
                st.markdown('<div class="frame-card">', unsafe_allow_html=True)
                col_frame1, col_frame2 = st.columns(2)
                with col_frame1:
                    st.image(image, caption="Original Capture", use_container_width=True)
                with col_frame2:
                    st.image(annotated_rgb, caption="SafeGuard AI Render", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Render compliance breakdown checklist below image
                st.markdown("<br><h4>📋 Shield Checklist Breakdown</h4>", unsafe_allow_html=True)
                counts = type_breakdown(model, r.boxes)
                
                classes = ["helmet", "Vest", "gloves", "shoes"]
                labels = ["Safety Helmet", "Reflective Vest", "Protective Gloves", "Steel-toe Shoes"]
                
                col_list1, col_list2 = st.columns(2)
                for idx, (cls_name, label) in enumerate(zip(classes, labels)):
                    detected = counts.get(cls_name, 0) > 0
                    badge_class = "status-badge-ok" if detected else "status-badge-err"
                    badge_text = "✔ Detected" if detected else "❌ Missing"
                    
                    target_col = col_list1 if idx < 2 else col_list2
                    with target_col:
                        st.markdown(
                            f"""
                            <div class="summary-item">
                                <span>{label}</span>
                                <span class="status-badge {badge_class}">{badge_text}</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            
            with col_an:
                people_count = counts.get("person", 0) # Fallback to 0 if class isn't tracked or present
                # If no person tracked directly, default to 1 if some items were seen
                if people_count == 0 and len(r.boxes) > 0:
                    people_count = 1
                
                # Calculate metrics values
                total_violations = 0
                if not counts.get("helmet", 0): total_violations += 1
                if not counts.get("Vest", 0): total_violations += 1
                
                helmet_compliance = "100%" if counts.get("helmet", 0) else "0%"
                vest_compliance = "100%" if counts.get("Vest", 0) else "0%"
                
                # KPIs card renders
                render_analytics_card("Total Monitored Workers", f"{people_count} Worker(s)", "blue", PEOPLE_SVG)
                render_analytics_card("Helmet Compliance", helmet_compliance, "green" if helmet_compliance == "100%" else "red", HELMET_SVG)
                render_analytics_card("Safety Vest Compliance", vest_compliance, "green" if vest_compliance == "100%" else "red", VEST_SVG)
                
                # Event Feed Logs
                events = []
                now_str = time.strftime("%H:%M")
                if not counts.get("helmet", 0):
                    events.append((now_str, "Safety Alert: Helmet Missing", "Worker identified without safety helmet compliance.", "#EF4444"))
                if not counts.get("Vest", 0):
                    events.append((now_str, "Safety Alert: Vest Missing", "Worker identified without protective vest compliance.", "#EF4444"))
                if len(events) == 0:
                    events.append((now_str, "Checklist Completed", "All protective gear detected successfully.", "#10B981"))
                
                render_timeline_feed(events)

    # ---------------- Video Stream tab ----------------
    with tab_video:
        uploaded_video = st.file_uploader("Upload Inspection Video Stream", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            col_v_res, col_v_an = st.columns([2, 1])
            
            with col_v_res:
                stframe = st.empty()
                stop_button = st.button("⏹️ Terminate Video Stream", key="stop_video")
            
            with col_v_an:
                perf_placeholder = st.empty()
                violations_timeline = st.empty()

            cap = cv2.VideoCapture(video_path)
            events_log = []
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                results = model.predict(source=frame, conf=conf_thresh, verbose=False)
                latency = (time.time() - start_time) * 1000
                
                r = results[0]
                annotated = r.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_rgb, use_container_width=True)

                counts = type_breakdown(model, r.boxes)
                total_detected = sum(counts.values())
                
                perf_placeholder.markdown(
                    f"""
                    <div class="analytics-card" style="border-top: 4px solid #3B82F6;">
                        <div class="analytics-title">STREAM PROCESSING ENGINE</div>
                        <div class="analytics-val">{latency:.1f} ms</div>
                        <div style="color: {sub_text}; font-size: 0.85rem; margin-top: 4px;">Inference Frame Processing Latency</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Check frame changes for timeline log alerts
                time_now = time.strftime("%H:%M")
                new_alerts = []
                if not counts.get("helmet", 0) and not any("Helmet" in x[1] for x in events_log[-2:]):
                    new_alerts.append((time_now, "Helmet violation", "Frame violation logged.", "#EF4444"))
                if not counts.get("Vest", 0) and not any("Vest" in x[1] for x in events_log[-2:]):
                    new_alerts.append((time_now, "Vest violation", "Frame violation logged.", "#EF4444"))
                
                if new_alerts:
                    events_log.extend(new_alerts)
                    if len(events_log) > 4:
                        events_log = events_log[-4:]
                
                # Display feed in timeline
                with violations_timeline:
                    render_timeline_feed(events_log or [(time_now, "Scanning Stream", "Active visual safety checkpoints.", "#3B82F6")])
                    
            cap.release()

    # ---------------- Live Webcam tab ----------------
    with tab_webcam:
        st.write("Inspecting workspace frame using webcam frame captures.")
        camera_image = st.camera_input("📸 Capture SafeGuard Frame")
        
        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            
            start_time = time.time()
            results = model.predict(source=image, conf=conf_thresh, verbose=False)
            latency = (time.time() - start_time) * 1000
            
            r = results[0]
            annotated = r.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            col_cam_res, col_cam_an = st.columns([2, 1])
            
            with col_cam_res:
                st.image(annotated_rgb, caption="Inspection capture result", use_container_width=True)
                
            with col_cam_an:
                counts = type_breakdown(model, r.boxes)
                people_count = counts.get("person", 0) or (1 if len(r.boxes) > 0 else 0)
                helmet_ok = "100%" if counts.get("helmet", 0) else "0%"
                vest_ok = "100%" if counts.get("Vest", 0) else "0%"
                
                render_analytics_card("Inspected People", f"{people_count} Person", "blue", PEOPLE_SVG)
                render_analytics_card("Helmet status", helmet_ok, "green" if helmet_ok == "100%" else "red", HELMET_SVG)
                render_analytics_card("Vest status", vest_ok, "green" if vest_ok == "100%" else "red", VEST_SVG)

    # ----------------- Dashboard Global Statistics Grid (Bottom) -----------------
    st.markdown("---")
    st.markdown("### 📈 Engine Statistics Summary")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown(
            f"""
            <div class="analytics-card">
                <div class="analytics-title">Inspection sessions</div>
                <div class="analytics-val" style="color: #3B82F6;">Active</div>
                <div style="font-size: 0.85rem; color: {sub_text}; margin-top: 4px;">Monitor status is operational</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_stat2:
        st.markdown(
            f"""
            <div class="analytics-card">
                <div class="analytics-title">Dataset size</div>
                <div class="analytics-val">427 Files</div>
                <div style="font-size: 0.85rem; color: {sub_text}; margin-top: 4px;">Roboflow validation images</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_stat3:
        st.markdown(
            f"""
            <div class="analytics-card">
                <div class="analytics-title">Baseline model</div>
                <div class="analytics-val">YOLO11n</div>
                <div style="font-size: 0.85rem; color: {sub_text}; margin-top: 4px;">2.58 Million parameters</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_stat4:
        st.markdown(
            f"""
            <div class="analytics-card">
                <div class="analytics-title">Violations detected</div>
                <div class="analytics-val" style="color: #EF4444;">Logged</div>
                <div style="font-size: 0.85rem; color: {sub_text}; margin-top: 4px;">Compliance checks feed is alive</div>
            </div>
            """,
            unsafe_allow_html=True
        )


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


if __name__ == "__main__":
    main()


