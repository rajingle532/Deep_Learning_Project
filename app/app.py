"""
AgriGuard AI: Enterprise Agricultural Pathology Intelligence Platform.
Tagline: Intelligent Crop Disease Detection. Smarter Farming. Healthier Harvests.

Production Architecture:
- Theme: Sophisticated Dark Agricultural (Backgrounds: #080E19, #0D1625, Cards: #111D2E, #17263A)
- Brand Accents: Primary Green #16C784, Secondary Green #80E3A7, Accent Cyan #42C8E8
- Core Engine: ResNet50 + MixStyle Domain Generalization (ICLR 2021)
- XAI Engine: Grad-CAM Saliency Studio (Selvaraju et al., ICCV 2017)
- Edge Runtime: Quantized TFLite INT8 FlatBuffer (23.86 MB)
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root and app directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import base64
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

from src.taxonomy import (
    CANONICAL_CLASSES,
    CANONICAL_DISPLAY_NAMES,
    TREATMENT_ADVISORY,
    CROP_GROUPS,
    NUM_CLASSES,
)
from src.domain_generalization import MixStyle
from src.preprocessing import normalize_image
from src.gradcam import compute_gradcam_heatmap, overlay_gradcam, find_target_conv_layer
try:
    from pdf_generator import generate_agronomic_pdf_prescription
except Exception:
    try:
        from app.pdf_generator import generate_agronomic_pdf_prescription
    except Exception:
        import pdf_generator
        generate_agronomic_pdf_prescription = pdf_generator.generate_agronomic_pdf_prescription

# Structured Crop Prior Mapping for Bayesian Domain Guidance
CROP_FAMILY_MAP = {
    "🌾 Auto-Detect (All 28 Classes)": None,
    "🥔 Potato (Solanaceae)": ["potato_early_blight", "potato_late_blight"],
    "🍅 Tomato (Solanaceae)": [c for c in CANONICAL_CLASSES if c.startswith("tomato_")],
    "🌽 Corn / Maize": [c for c in CANONICAL_CLASSES if c.startswith("corn_")],
    "🍎 Apple (Malus)": [c for c in CANONICAL_CLASSES if c.startswith("apple_")],
    "🍇 Grape (Vitaceae)": [c for c in CANONICAL_CLASSES if c.startswith("grape_")],
    "🫑 Bell Pepper": [c for c in CANONICAL_CLASSES if c.startswith("bell_pepper_")],
    "🎃 Squash": ["squash_powdery_mildew"],
    "🍑 Peach": ["peach_healthy"],
    "🍒 Cherry": ["cherry_healthy"],
    "🍓 Strawberry": ["strawberry_healthy"],
}

# -------------------------------------------------------------
# 0. SECURE HTML RENDERER (NATIVE ST.HTML DOM RENDERING)
# -------------------------------------------------------------
def render_html(html_str: str, *args, **kwargs) -> None:
    """Safely renders HTML/CSS directly via st.html without triggering CommonMark code-block leaks."""
    clean_str = html_str.strip()
    if clean_str.startswith("#"):
        st.markdown(clean_str)
    else:
        st.html(clean_str)


# 1. PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="AgriGuard AI | Intelligent Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------
# 2. MASTER DESIGN SYSTEM CSS (Tokens + Enterprise Glassmorphism)
# -------------------------------------------------------------
tokens_file = APP_DIR / "design_tokens.css"
tokens_css = tokens_file.read_text(encoding="utf-8") if tokens_file.exists() else ""

render_html("<style>" + tokens_css + """
    /* Master Streamlit Viewport & Deep Forest Obsidian Theme */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: radial-gradient(circle at 50% 0%, #0d1e2e 0%, #060b12 50%, #03060a 100%) !important;
        background-attachment: fixed !important;
        color: var(--color-text-primary, #F8FAFC) !important;
        font-family: var(--font-ui, 'Plus Jakarta Sans', sans-serif) !important;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    html, body, [class*="css"] {
        font-family: var(--font-ui, 'Plus Jakarta Sans', sans-serif);
        color: var(--color-text-primary, #F8FAFC);
    }

    /* Streamlit Main Viewport Margins & Fluid Spacing */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1540px !important;
    }

    /* Custom Modern Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #060B12;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #10B981;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.40);
    }

    /* KEYFRAME ANIMATIONS: Neon Pulse, Radar & Shimmer */
    @keyframes neonPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7), 0 0 8px rgba(16, 185, 129, 0.9);
        }
        70% {
            box-shadow: 0 0 0 12px rgba(16, 185, 129, 0), 0 0 18px rgba(16, 185, 129, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0), 0 0 0 rgba(16, 185, 129, 0);
        }
    }

    @keyframes radarPulse {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.5), inset 0 0 12px rgba(16, 185, 129, 0.3);
            transform: scale(0.97);
        }
        50% {
            box-shadow: 0 0 0 16px rgba(16, 185, 129, 0), inset 0 0 20px rgba(16, 185, 129, 0.5);
            transform: scale(1.02);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0), inset 0 0 12px rgba(16, 185, 129, 0.3);
            transform: scale(0.97);
        }
    }

    @keyframes uploadBorderPulse {
        0% {
            border-color: rgba(16, 185, 129, 0.35);
        }
        50% {
            border-color: rgba(16, 185, 129, 0.75);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
        }
        100% {
            border-color: rgba(16, 185, 129, 0.35);
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-fade-in {
        animation: fadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }

    /* ========================================================= */
    /* 1. TOP HERO DASHBOARD HEADER (Clear Focal Point & Aligned Stats) */
    /* ========================================================= */
    .hero-header-card {
        background: linear-gradient(135deg, rgba(16, 28, 46, 0.94) 0%, rgba(9, 16, 28, 0.98) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-radius: 20px;
        padding: 22px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        margin-bottom: 20px;
        box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.85), inset 0 1px 0 rgba(255, 255, 255, 0.10);
        position: relative;
        overflow: hidden;
    }
    .hero-header-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #10B981 30%, #06B6D4 70%, transparent 100%);
    }

    .hero-brand-left {
        display: flex;
        align-items: center;
        gap: 18px;
        flex: 1;
    }
    .hero-logo-box {
        width: 58px;
        height: 58px;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.15) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.55);
        border-radius: 16px;
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.16);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .hero-title-wrap {
        display: flex;
        flex-direction: column;
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-title-text {
        font-size: 2.35rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.8px !important;
        background: linear-gradient(135deg, #FFFFFF 25%, #A7F3D0 75%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .enterprise-badge {
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.9px;
        background: rgba(6, 182, 212, 0.18);
        color: #22D3EE;
        border: 1px solid rgba(6, 182, 212, 0.45);
        padding: 4px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        box-shadow: 0 0 12px rgba(6, 182, 212, 0.22);
    }
    .hero-tagline {
        font-size: 0.92rem;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
        letter-spacing: 0.1px;
    }
    .hero-status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
        font-size: 0.8125rem;
        font-weight: 600;
        color: #6EE7B7;
    }
    .status-pulse-dot {
        width: 8px;
        height: 8px;
        background: #10B981;
        border-radius: 50%;
        display: inline-block;
        animation: neonPulse 2.2s infinite;
        flex-shrink: 0;
    }
    .hero-telemetry-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 2px 10px;
        border-radius: 12px;
        margin-left: 6px;
    }

    /* Right Performance Metrics (4 Compact Subtle Glass Tiles) */
    .hero-metrics-grid {
        display: flex;
        gap: 10px;
        flex-shrink: 0;
    }
    .hero-stat-card {
        background: rgba(10, 18, 30, 0.80);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 14px;
        padding: 9px 15px;
        text-align: right;
        min-width: 110px;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 16px -4px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    .hero-stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.50);
        box-shadow: 0 10px 24px -4px rgba(0, 0, 0, 0.7), 0 0 16px rgba(16, 185, 129, 0.20);
    }
    .hero-stat-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34D399;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.3px;
        text-shadow: 0 0 12px rgba(52, 211, 153, 0.35);
        display: block;
    }
    .hero-stat-lbl {
        font-size: 0.68rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
        display: block;
        margin-top: 2px;
    }

    /* ========================================================= */
    /* 2. NAVIGATION TABS (Lucide-Style SVG Icons & Animated Underline) */
    /* ========================================================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px !important;
        background-color: transparent !important;
        padding-bottom: 12px !important;
        border-bottom: 1px solid rgba(16, 185, 129, 0.18) !important;
    }
    .stTabs [data-baseweb="tab"] {
        position: relative !important;
        background: rgba(14, 25, 42, 0.75) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(16, 185, 129, 0.20) !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        min-height: 44px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #94A3B8 !important;
        display: inline-flex !important;
        align-items: center !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(16, 185, 129, 0.50) !important;
        color: #F8FAFC !important;
        transform: translateY(-1px) !important;
    }
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 2px solid #10B981 !important;
        outline-offset: 2px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.24) 0%, rgba(6, 182, 212, 0.14) 100%) !important;
        border: 1.5px solid #10B981 !important;
        color: #A7F3D0 !important;
        box-shadow: 0 4px 16px -2px rgba(16, 185, 129, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
    }
    .stTabs [aria-selected="true"]::after {
        content: '' !important;
        position: absolute !important;
        bottom: -2px !important;
        left: 20% !important;
        width: 60% !important;
        height: 2.5px !important;
        background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%) !important;
        border-radius: 2px !important;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.8) !important;
    }

    /* Lucide-Style SVG Tab Icons on ::before */
    .stTabs [data-baseweb="tab"]:nth-child(1)::before {
        content: '' !important;
        display: inline-block !important;
        width: 17px !important;
        height: 17px !important;
        margin-right: 8px !important;
        vertical-align: -2px !important;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2310B981' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 12h-4l-3 9L9 3l-3 9H2'/%3E%3C/svg%3E") no-repeat center !important;
        background-size: contain !important;
    }
    .stTabs [data-baseweb="tab"]:nth-child(2)::before {
        content: '' !important;
        display: inline-block !important;
        width: 17px !important;
        height: 17px !important;
        margin-right: 8px !important;
        vertical-align: -2px !important;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2306B6D4' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3C/svg%3E") no-repeat center !important;
        background-size: contain !important;
    }
    .stTabs [data-baseweb="tab"]:nth-child(3)::before {
        content: '' !important;
        display: inline-block !important;
        width: 17px !important;
        height: 17px !important;
        margin-right: 8px !important;
        vertical-align: -2px !important;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2334D399' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 20h10'/%3E%3Cpath d='M10 20c0-4.4 3.6-8 8-8V4c-4.4 0-8 3.6-8 8v8Z'/%3E%3Cpath d='M4 12c4.4 0 8 3.6 8 8'/%3E%3C/svg%3E") no-repeat center !important;
        background-size: contain !important;
    }
    .stTabs [data-baseweb="tab"]:nth-child(4)::before {
        content: '' !important;
        display: inline-block !important;
        width: 17px !important;
        height: 17px !important;
        margin-right: 8px !important;
        vertical-align: -2px !important;
        background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2338BDF8' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect width='14' height='20' x='5' y='2' rx='2' ry='2'/%3E%3Cpath d='M12 18h.01'/%3E%3C/svg%3E") no-repeat center !important;
        background-size: contain !important;
    }

    /* ========================================================= */
    /* 3. QUICK AI DEMONSTRATIONS (Color-Coded SOTA & Mobile Swipe) */
    /* ========================================================= */
    .quick-demos-container {
        background: linear-gradient(135deg, rgba(16, 28, 46, 0.85) 0%, rgba(9, 16, 28, 0.92) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 18px;
        padding: 14px 20px 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 12px 32px -10px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .quick-demos-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .quick-demos-head {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .quick-demos-sub {
        font-size: 0.8125rem;
        color: #94A3B8;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .active-specimen-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.22);
        border: 1px solid #10B981;
        color: #A7F3D0;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 8px;
        animation: neonPulse 2.5s infinite;
    }

    /* 7 Demo Buttons: Clean Typography, Lift, & Color-Coded Tiers */
    .st-key-btn_pot button,
    .st-key-btn_tom button,
    .st-key-btn_crn button,
    .st-key-btn_app button,
    .st-key-btn_grp button,
    .st-key-btn_pep button,
    .st-key-btn_pot_field button {
        min-height: 64px !important;
        font-size: 0.8125rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        background: linear-gradient(145deg, rgba(16, 30, 48, 0.95) 0%, rgba(8, 15, 26, 0.98) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        white-space: pre-line !important;
        line-height: 1.45 !important;
        padding: 10px 8px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    }
    .st-key-btn_pot button:hover,
    .st-key-btn_tom button:hover,
    .st-key-btn_crn button:hover,
    .st-key-btn_app button:hover,
    .st-key-btn_grp button:hover,
    .st-key-btn_pep button:hover,
    .st-key-btn_pot_field button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        border-color: #10B981 !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.30) 0%, rgba(6, 182, 212, 0.20) 100%) !important;
        color: #34D399 !important;
        box-shadow: 0 12px 28px -4px rgba(16, 185, 129, 0.45), 0 0 16px rgba(16, 185, 129, 0.25) !important;
    }

    /* SOTA Score Color-Coded Accent Borders */
    .st-key-btn_pot button,
    .st-key-btn_tom button,
    .st-key-btn_crn button,
    .st-key-btn_app button,
    .st-key-btn_pot_field button {
        border-top: 2.5px solid #10B981 !important; /* Green 90%+ */
    }
    .st-key-btn_grp button,
    .st-key-btn_pep button {
        border-top: 2.5px solid #F59E0B !important; /* Amber <85% */
    }

    /* Mobile Horizontal Swipe for Quick Demos */
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_pot) {
            display: flex !important;
            overflow-x: auto !important;
            flex-wrap: nowrap !important;
            gap: 10px !important;
            padding-bottom: 10px !important;
            -webkit-overflow-scrolling: touch !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.st-key-btn_pot) > div[data-testid="column"] {
            min-width: 125px !important;
            flex: 0 0 auto !important;
        }
    }

    /* ========================================================= */
    /* 4. PANEL HEADINGS & GLASS CONTAINERS (Sticky Left Panel)   */
    /* ========================================================= */
    .panel-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .panel-card-subtitle {
        font-size: 0.8125rem;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 18px;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(14, 25, 40, 0.88) 0%, rgba(7, 13, 22, 0.95) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(16, 185, 129, 0.28) !important;
        border-radius: 20px !important;
        padding: 22px 24px !important;
        box-shadow: 0 18px 44px -10px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(16, 185, 129, 0.48) !important;
        box-shadow: 0 22px 50px -8px rgba(0, 0, 0, 0.88), 0 0 24px rgba(16, 185, 129, 0.16) !important;
    }

    /* Desktop Sticky Left Column */
    @media (min-width: 992px) {
        div[data-testid="column"]:first-child > div[data-testid="stVerticalBlockBorderWrapper"] {
            position: sticky !important;
            top: 1.5rem !important;
            z-index: 10 !important;
        }
    }

    /* File Uploader Dropzone: Dashed Animated Pulse & Drag-Over */
    [data-testid="stFileUploader"] section {
        background: rgba(10, 18, 30, 0.75) !important;
        border: 2px dashed rgba(16, 185, 129, 0.40) !important;
        animation: uploadBorderPulse 3.5s infinite ease-in-out !important;
        border-radius: 16px !important;
        padding: 28px 20px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #10B981 !important;
        background: rgba(16, 185, 129, 0.08) !important;
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.25) !important;
    }

    /* Segmented Control Switcher */
    div[data-testid="stSegmentedControl"] button {
        border-radius: 10px !important;
        border: 1px solid rgba(16, 185, 129, 0.24) !important;
        background: rgba(10, 18, 30, 0.85) !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        min-height: 44px !important;
        padding: 8px 18px !important;
        font-size: 0.86rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        border-color: rgba(16, 185, 129, 0.50) !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: rgba(16, 185, 129, 0.30) !important;
        border-color: #10B981 !important;
        color: #A7F3D0 !important;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.40) !important;
    }

    /* Leaf Specimen Image Optical Frame */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 8px 0 14px 0 !important;
    }
    div[data-testid="stImage"] img {
        border-radius: 16px !important;
        border: 2px solid rgba(16, 185, 129, 0.40) !important;
        box-shadow: 0 14px 38px -8px rgba(0, 0, 0, 0.85), 0 0 24px rgba(16, 185, 129, 0.18) !important;
        max-height: 380px !important;
        width: auto !important;
        max-width: 100% !important;
        object-fit: contain !important;
        background: #060B12 !important;
        transition: transform 0.3s ease, border-color 0.3s ease !important;
    }
    div[data-testid="stImage"] img:hover {
        border-color: #10B981 !important;
        box-shadow: 0 18px 48px -6px rgba(0, 0, 0, 0.95), 0 0 32px rgba(16, 185, 129, 0.30) !important;
    }

    .specimen-scanner-header {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.16) 0%, rgba(6, 182, 212, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 8px 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .specimen-scanner-title {
        font-size: 0.8125rem;
        font-weight: 800;
        color: #A7F3D0;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .specimen-scanner-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #38BDF8;
        background: rgba(0, 0, 0, 0.4);
        padding: 2px 10px;
        border-radius: 8px;
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .specimen-source-tag {
        font-size: 0.8125rem;
        color: #CBD5E1;
        background: rgba(10, 18, 30, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 10px;
        padding: 7px 14px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ========================================================= */
    /* 5. RIGHT PANEL: EMPTY STATE WITH RADAR & PROGRESS STEPPER */
    /* ========================================================= */
    .empty-workflow-card {
        background: linear-gradient(145deg, rgba(14, 25, 40, 0.75) 0%, rgba(7, 13, 22, 0.88) 100%) !important;
        backdrop-filter: blur(18px) !important;
        border: 2px dashed rgba(16, 185, 129, 0.38) !important;
        border-radius: 20px !important;
        padding: 38px 24px !important;
        text-align: center !important;
        box-shadow: 0 16px 42px -10px rgba(0, 0, 0, 0.7) !important;
    }
    .radar-scan-box {
        width: 86px;
        height: 86px;
        margin: 0 auto 16px auto;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.08) 70%, transparent 100%);
        border: 2px solid rgba(16, 185, 129, 0.55);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.35), inset 0 0 15px rgba(16, 185, 129, 0.25);
        animation: radarPulse 3.0s infinite ease-in-out;
    }
    .empty-graphic-svg {
        width: 44px;
        height: 44px;
        color: #34D399;
        filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.5));
    }
    .empty-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.4px;
        margin-bottom: 6px;
    }
    .empty-desc {
        font-size: 0.8125rem;
        color: #94A3B8;
        max-width: 480px;
        margin: 0 auto 24px auto;
        line-height: 1.55;
    }

    /* 3-Step Connected Progress Stepper */
    .stepper-track {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        text-align: left;
    }
    .stepper-step {
        background: rgba(14, 25, 42, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.20);
        border-radius: 14px;
        padding: 14px 16px;
        transition: all 0.25s ease;
    }
    .stepper-step-active {
        border-color: #10B981 !important;
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.16) 0%, rgba(10, 18, 30, 0.95) 100%) !important;
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.3) !important;
    }
    .stepper-badge {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .stepper-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        color: #34D399;
    }
    .stepper-tag {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 10px;
        letter-spacing: 0.5px;
    }
    .tag-active {
        background: rgba(16, 185, 129, 0.25);
        color: #A7F3D0;
        border: 1px solid #10B981;
    }
    .tag-pending {
        background: rgba(100, 116, 139, 0.20);
        color: #94A3B8;
        border: 1px solid rgba(100, 116, 139, 0.35);
    }
    .stepper-head {
        font-size: 0.88rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 3px;
    }
    .stepper-sub {
        font-size: 0.8125rem;
        color: #94A3B8;
        line-height: 1.45;
    }
    .stepper-connector {
        width: 14px;
        height: 2px;
        background: rgba(16, 185, 129, 0.3);
    }

    /* Completed Stepper Top Bar in Results State */
    .completed-stepper-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 18, 30, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 8px 16px;
        margin-bottom: 14px;
        font-size: 0.8125rem;
    }
    .step-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-weight: 600;
        color: #A7F3D0;
    }
    .step-divider {
        height: 1px;
        flex: 1;
        background: rgba(16, 185, 129, 0.25);
        margin: 0 12px;
    }

    /* ========================================================= */
    /* 6. PRIMARY DIAGNOSIS CARD & CONFIDENCE GAUGE RING         */
    /* ========================================================= */
    .diag-hero-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(15, 26, 42, 0.95) 55%, rgba(6, 182, 212, 0.12) 100%) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1.5px solid rgba(16, 185, 129, 0.55) !important;
        border-left: 7px solid #10B981 !important;
        border-radius: 18px !important;
        padding: 20px 24px !important;
        margin-bottom: 18px !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        gap: 20px !important;
        box-shadow: 0 16px 42px -8px rgba(0, 0, 0, 0.75), 0 0 28px rgba(16, 185, 129, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
        transition: all 0.28s ease !important;
    }
    .diag-hero-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 48px -6px rgba(0, 0, 0, 0.85), 0 0 34px rgba(16, 185, 129, 0.30);
    }
    .diag-hero-inconclusive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(15, 26, 42, 0.95) 55%, rgba(245, 158, 11, 0.10) 100%) !important;
        border: 1.5px solid rgba(239, 68, 68, 0.55) !important;
        border-left: 7px solid #EF4444 !important;
        box-shadow: 0 16px 42px -8px rgba(0, 0, 0, 0.75), 0 0 28px rgba(239, 68, 68, 0.22) !important;
    }
    .diag-hero-main-content {
        flex: 1;
    }
    .diag-hero-gauge-container {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .crop-pill-tag {
        display: inline-block;
        background: rgba(16, 185, 129, 0.20);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.45);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 8px;
    }
    .diag-primary-title {
        font-size: 1.95rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.5px !important;
        line-height: 1.22 !important;
        margin: 4px 0 6px 0 !important;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.5) !important;
    }
    .diag-pathogen-sub {
        font-size: 0.96rem !important;
        color: #6EE7B7 !important;
        font-style: italic !important;
        margin-bottom: 12px !important;
    }
    .conf-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8125rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.25);
        color: #A7F3D0;
        border: 1.5px solid #10B981;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.35);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.25);
        color: #FCD34D;
        border: 1.5px solid #F59E0B;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.35);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.25);
        color: #FCA5A5;
        border: 1.5px solid #EF4444;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.35);
    }

    /* ========================================================= */
    /* 7. TELEMETRY MICRO-METRICS & SEVERITY METER               */
    /* ========================================================= */
    .telemetry-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .telemetry-item {
        background: linear-gradient(145deg, rgba(16, 30, 48, 0.88) 0%, rgba(9, 16, 26, 0.95) 100%) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(16, 185, 129, 0.28) !important;
        border-radius: 14px !important;
        padding: 13px 10px !important;
        text-align: center !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    }
    .telemetry-item:hover {
        transform: translateY(-2px) !important;
        border-color: #10B981 !important;
        box-shadow: 0 12px 30px -4px rgba(0, 0, 0, 0.75), 0 0 18px rgba(16, 185, 129, 0.25) !important;
    }
    .telemetry-val {
        font-size: 1.30rem !important;
        font-weight: 800 !important;
        color: #34D399 !important;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.4px !important;
        text-shadow: 0 0 14px rgba(52, 211, 153, 0.4) !important;
    }
    .telemetry-lbl {
        font-size: 0.68rem;
        color: #94A3B8;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.6px;
        margin-top: 3px;
    }

    /* Severity Meter */
    .severity-card {
        background: linear-gradient(135deg, rgba(16, 28, 46, 0.92) 0%, rgba(9, 16, 28, 0.96) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.30);
        border-radius: 18px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.65), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        transition: all 0.25s ease;
    }
    .severity-card:hover {
        border-color: rgba(16, 185, 129, 0.55);
        box-shadow: 0 16px 40px -6px rgba(0, 0, 0, 0.75), 0 0 22px rgba(16, 185, 129, 0.18);
    }
    .severity-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .severity-title {
        font-size: 0.90rem;
        font-weight: 700;
        color: #F8FAFC;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .severity-pct-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    }
    .severity-meter-rail {
        height: 12px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 12px;
        position: relative;
    }
    .severity-meter-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.55);
    }
    .severity-action-box {
        background: rgba(6, 11, 18, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 0.8125rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Differential Top-3 Bars */
    .diff-item {
        background: rgba(12, 22, 34, 0.88);
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 12px;
        padding: 11px 16px;
        margin-bottom: 9px;
        transition: all 0.2s ease;
    }
    .diff-item:hover {
        border-color: rgba(16, 185, 129, 0.45);
        transform: translateX(3px);
    }
    .diff-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .diff-score {
        font-family: 'JetBrains Mono', monospace;
        color: #34D399;
        font-weight: 700;
        font-size: 0.94rem;
    }
    .diff-rail {
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .diff-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%);
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.45);
    }
    .entropy-telemetry-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(10, 18, 30, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.20);
        border-radius: 12px;
        padding: 9px 16px;
        margin-top: 10px;
        font-size: 0.8125rem;
        color: #94A3B8;
    }
    .entropy-mono-val {
        font-family: 'JetBrains Mono', monospace;
        color: #34D399;
        font-weight: 700;
    }

    /* ========================================================= */
    /* 8. TABBED AGRONOMIC TREATMENT CARDS                       */
    /* ========================================================= */
    .rx-treatment-card {
        background: linear-gradient(145deg, rgba(14, 25, 40, 0.92) 0%, rgba(8, 14, 24, 0.96) 100%);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 20px 22px;
        margin-top: 8px;
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.55);
        transition: all 0.25s ease;
    }
    .rx-treatment-card:hover {
        border-color: rgba(16, 185, 129, 0.50);
        transform: translateY(-2px);
    }
    .rx-card-organic {
        border-top: 3px solid #10B981 !important;
    }
    .rx-card-chemical {
        border-top: 3px solid #F59E0B !important;
    }
    .rx-card-preventive {
        border-top: 3px solid #06B6D4 !important;
    }
    .rx-card-etiology {
        border-top: 3px solid #8B5CF6 !important;
    }
    .rx-card-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.40);
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
    }
    .rx-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 10px;
    }
    .rx-card-content {
        font-size: 0.875rem;
        color: #CBD5E1;
        line-height: 1.6;
    }
    .rx-card-footer {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.75rem;
        color: #94A3B8;
        font-style: italic;
    }

    /* Buttons & Touch Targets */
    div.stButton > button {
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        min-height: 44px !important;
        background: linear-gradient(135deg, rgba(15, 26, 42, 0.90) 0%, rgba(10, 18, 30, 0.96) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.28) !important;
        color: #F8FAFC !important;
        padding: 9px 18px !important;
        white-space: pre-line !important;
        line-height: 1.4 !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(6, 182, 212, 0.18) 100%) !important;
        border-color: #10B981 !important;
        color: #34D399 !important;
        box-shadow: 0 10px 24px -4px rgba(16, 185, 129, 0.4), 0 0 16px rgba(16, 185, 129, 0.25) !important;
    }
    div.stDownloadButton > button, div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10B981 0%, #059669 50%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        min-height: 44px !important;
        border: none !important;
        box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.5) !important;
    }
    div.stDownloadButton > button:hover, div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px -2px rgba(16, 185, 129, 0.7), 0 0 20px rgba(6, 182, 212, 0.4) !important;
    }

    /* ========================================================= */
    /* 9. CLEAN ENTERPRISE FOOTER                                */
    /* ========================================================= */
    .enterprise-footer {
        background: linear-gradient(135deg, rgba(14, 25, 40, 0.90) 0%, rgba(8, 14, 24, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 18px;
        padding: 24px 28px;
        margin-top: 32px;
        box-shadow: 0 14px 34px -8px rgba(0, 0, 0, 0.7);
    }
    .footer-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .footer-brand {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .footer-logo {
        font-size: 1.4rem;
    }
    .footer-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.4px;
    }
    .footer-badge {
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.40);
        padding: 2px 8px;
        border-radius: 10px;
    }
    .footer-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .footer-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #94A3B8;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 3px 10px;
        border-radius: 10px;
    }
    .footer-divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 16px 0;
    }
    .footer-bottom-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 0.8125rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    .footer-copy {
        color: #64748B;
        font-size: 0.75rem;
    }
</style>
""")


# -------------------------------------------------------------
# 3. MODEL INITIALIZATION & WARM-UP CACHE
# -------------------------------------------------------------
@st.cache_resource
def load_production_model():
    """Load, verify, and warm up ResNet50 + MixStyle deep network."""
    model_path = PROJECT_ROOT / "models" / "best_model.keras"
    if not model_path.exists():
        return None
    try:
        model = keras.models.load_model(
            str(model_path),
            custom_objects={"MixStyle": MixStyle},
        )
        # Warm up execution graph to eliminate cold-start latency
        dummy_tensor = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _ = model(dummy_tensor, training=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_production_model()


# -------------------------------------------------------------
# 3.B PILLAR 2 HELPER FUNCTIONS (Severity Quantification & Slider)
# -------------------------------------------------------------
def pil_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 PNG data string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def render_confidence_ring_svg(conf_pct: float, color: str = "#10B981", size: int = 86) -> str:
    """Generate inline SVG circular progress gauge."""
    radius = 34
    circ = 2 * 3.14159265 * radius
    clamped_pct = min(max(conf_pct, 0.0), 100.0)
    offset = circ * (1.0 - (clamped_pct / 100.0))
    return f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 86 86">
            <circle cx="43" cy="43" r="{radius}" stroke="rgba(255, 255, 255, 0.08)" stroke-width="7" fill="none" />
            <circle cx="43" cy="43" r="{radius}" stroke="{color}" stroke-width="7" fill="none"
                    stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}"
                    stroke-linecap="round" transform="rotate(-90 43 43)"
                    style="transition: stroke-dashoffset 0.8s ease; filter: drop-shadow(0 0 6px {color}80);" />
            <text x="43" y="47" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="15" font-weight="800" fill="#F8FAFC">{clamped_pct:.1f}%</text>
        </svg>
        <span style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #94A3B8; margin-top: 2px;">CERTAINTY</span>
    </div>
    """


def calculate_disease_severity(heatmap: np.ndarray, top_canonical: str, threshold: float = 0.38) -> dict:
    """
    Quantifies foliar disease lesion coverage based on normalized Grad-CAM activation mass.
    Returns structured metrics for the clinical tri-stage classification.
    """
    if "healthy" in top_canonical.lower():
        return {
            "severity_pct": 0.0,
            "stage_id": "STAGE_0",
            "stage_name": "Healthy Foliage (Clean Specimen)",
            "stage_badge": "🟢 Healthy Foliage (0% Necrosis)",
            "stage_color": "#10B981",
            "rail_gradient": "linear-gradient(90deg, #10B981 0%, #34D399 100%)",
            "action_protocol": "Foliage exhibits vigorous cellular health. Maintain regular prophylactic bio-nutrition and monitor periodically."
        }

    if heatmap is None:
        return {
            "severity_pct": 0.0,
            "stage_id": "UNKNOWN",
            "stage_name": "Saliency Unavailable",
            "stage_badge": "⚪ Saliency Unavailable",
            "stage_color": "#64748B",
            "rail_gradient": "linear-gradient(90deg, #64748B 0%, #94A3B8 100%)",
            "action_protocol": "Visual inspection advised."
        }

    # Normalized activation thresholding
    lesion_mask = heatmap >= threshold
    active_ratio = float(np.mean(lesion_mask)) * 100.0

    # Calibrate to realistic agronomic foliar surface infection scale (4% - 92%)
    calibrated_pct = float(np.clip(active_ratio * 1.35, 4.0, 92.5))

    if calibrated_pct <= 15.0:
        return {
            "severity_pct": calibrated_pct,
            "severity_level": "Low",
            "stage_id": "STAGE_1",
            "stage_name": "Stage I: Incipient Infection",
            "stage_badge": "🟢 Low Severity (0–15% Incipient)",
            "stage_color": "#10B981",
            "rail_gradient": "linear-gradient(90deg, #10B981 0%, #34D399 100%)",
            "action_protocol": "Preventive organic bio-fungicides (Neem oil, Bacillus subtilis) and foliar aeration recommended to halt secondary spore germination."
        }
    elif calibrated_pct <= 40.0:
        return {
            "severity_pct": calibrated_pct,
            "severity_level": "Medium",
            "stage_id": "STAGE_2",
            "stage_name": "Stage II: Moderate Infection",
            "stage_badge": "🟡 Medium Severity (15–40% Moderate)",
            "stage_color": "#F59E0B",
            "rail_gradient": "linear-gradient(90deg, #F59E0B 0%, #FBBF24 100%)",
            "action_protocol": "Targeted chemical systemic application (Mancozeb / Chlorothalonil) required within 48-72h to arrest mycelial progression."
        }
    else:
        return {
            "severity_pct": calibrated_pct,
            "severity_level": "High",
            "stage_id": "STAGE_3",
            "stage_name": "Stage III: Severe / Critical Infection",
            "stage_badge": "🔴 High Severity (>40% Critical)",
            "stage_color": "#EF4444",
            "rail_gradient": "linear-gradient(90deg, #EF4444 0%, #DC2626 100%)",
            "action_protocol": "Immediate quarantine, sanitation pruning of infected foliage, and broad-spectrum foliar eradication required to protect adjacent rows."
        }


def render_comparison_slider(raw_img: Image.Image, overlay_img: Image.Image, height: int = 380):
    """Render interactive draggable HTML5 before/after comparison slider."""
    raw_b64 = pil_to_base64(raw_img)
    overlay_b64 = pil_to_base64(overlay_img)
    slider_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Plus Jakarta Sans", sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            padding: 4px;
        }}
        .comparison-wrapper {{
            position: relative;
            width: 100%;
            max-width: 440px;
            height: {height - 20}px;
            border-radius: 16px;
            overflow: hidden;
            border: 2px solid rgba(16, 185, 129, 0.45);
            box-shadow: 0 16px 36px -10px rgba(0,0,0,0.7), 0 0 20px rgba(16, 185, 129, 0.2);
            user-select: none;
            -webkit-user-select: none;
        }}
        .img-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-size: cover;
            background-position: center;
        }}
        .img-before {{
            background-image: url('data:image/png;base64,{raw_b64}');
        }}
        .img-after {{
            background-image: url('data:image/png;base64,{overlay_b64}');
            width: 50%;
            border-right: 2px solid #10B981;
            box-shadow: 4px 0 16px rgba(16, 185, 129, 0.5);
        }}
        .slider-control {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            cursor: ew-resize;
            z-index: 20;
            margin: 0;
        }}
        .slider-knob {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 38px;
            height: 38px;
            background: #10B981;
            border: 2px solid #FFFFFF;
            border-radius: 50%;
            box-shadow: 0 0 16px rgba(16, 185, 129, 0.8), 0 4px 10px rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #060B12;
            font-size: 15px;
            font-weight: bold;
            pointer-events: none;
            z-index: 15;
            transition: transform 0.08s ease;
        }}
        .badge-label {{
            position: absolute;
            bottom: 12px;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            z-index: 10;
            pointer-events: none;
            backdrop-filter: blur(8px);
        }}
        .badge-left {{
            left: 12px;
            background: rgba(6, 11, 18, 0.8);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}
        .badge-right {{
            right: 12px;
            background: rgba(6, 11, 18, 0.8);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.4);
        }}
        .hint-text {{
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            color: rgba(255, 255, 255, 0.85);
            background: rgba(0,0,0,0.6);
            padding: 3px 12px;
            border-radius: 12px;
            pointer-events: none;
            z-index: 10;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
    </style>
    </head>
    <body>
        <div class="comparison-wrapper" id="sliderWrapper">
            <div class="img-layer img-before"></div>
            <div class="img-layer img-after" id="afterLayer"></div>
            <div class="slider-knob" id="sliderKnob">⬌</div>
            <span class="badge-label badge-left">Grad-CAM Overlay</span>
            <span class="badge-label badge-right">Raw Leaf</span>
            <span class="hint-text">Drag left/right to compare</span>
            <input type="range" min="0" max="100" value="50" class="slider-control" id="sliderRange" />
        </div>
        <script>
            const slider = document.getElementById('sliderRange');
            const afterLayer = document.getElementById('afterLayer');
            const knob = document.getElementById('sliderKnob');
            slider.addEventListener('input', (e) => {{
                const val = e.target.value;
                afterLayer.style.width = val + '%';
                knob.style.left = val + '%';
            }});
        </script>
    </body>
    </html>
    """
    components.html(slider_html, height=height)


def render_voice_advisory(crop_name: str, disease_name: str, certainty_pct: float, severity_pct: float, organic_rx: str, chemical_rx: str):
    """Renders browser-native Web Speech synthesis audio player in Hindi and English."""
    clean_disease = disease_name.replace(":", " -")
    clean_organic = organic_rx.replace("•", "").replace("\n", " ").strip()[:140]
    clean_chemical = chemical_rx.replace("•", "").replace("\n", " ").strip()[:140]

    hindi_text = f"Aapki fasal {crop_name} me {clean_disease} paayi gayi hai. Model certainty {certainty_pct:.0f} percent hai. Foliar severity {severity_pct:.0f} percent hai. Organic upaay: {clean_organic}. Chemical spray: {clean_chemical}."
    english_text = f"Diagnosis confirmed for {crop_name}: {clean_disease} with {certainty_pct:.0f} percent confidence. Foliar severity is {severity_pct:.0f} percent. Recommended organic control: {clean_organic}. Chemical regimen: {clean_chemical}."

    hindi_text_js = hindi_text.replace('"', '\\"').replace("'", "\\'")
    english_text_js = english_text.replace('"', '\\"').replace("'", "\\'")

    audio_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Plus Jakarta Sans", sans-serif;
            padding: 2px;
        }}
        .voice-card {{
            background: linear-gradient(135deg, rgba(15, 26, 42, 0.90) 0%, rgba(10, 18, 30, 0.95) 100%);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 14px;
            padding: 12px 16px;
            box-shadow: 0 8px 24px -6px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .voice-head {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .voice-title {{
            font-size: 0.82rem;
            font-weight: 700;
            color: #F8FAFC;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .voice-badge {{
            font-size: 0.68rem;
            font-weight: 700;
            color: #34D399;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.35);
            padding: 2px 8px;
            border-radius: 10px;
            font-family: monospace;
        }}
        .btn-row {{
            display: flex;
            gap: 8px;
        }}
        .btn-voice {{
            flex: 1;
            padding: 9px 12px;
            border-radius: 10px;
            font-size: 0.80rem;
            font-weight: 600;
            cursor: pointer;
            border: 1px solid rgba(16, 185, 129, 0.3);
            background: rgba(10, 18, 30, 0.85);
            color: #F8FAFC;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }}
        .btn-voice:hover {{
            background: rgba(16, 185, 129, 0.2);
            border-color: #10B981;
            color: #34D399;
            transform: translateY(-1px);
        }}
        .btn-stop {{
            flex: 0.35;
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.3);
            color: #FCA5A5;
        }}
        .btn-stop:hover {{
            background: rgba(239, 68, 68, 0.25);
            border-color: #EF4444;
            color: #FFF;
        }}
        .status-txt {{
            font-size: 0.72rem;
            color: #94A3B8;
            font-style: italic;
        }}
    </style>
    </head>
    <body>
        <div class="voice-card">
            <div class="voice-head">
                <span class="voice-title">🔊 Multilingual Agronomic Voice Advisory</span>
                <span class="voice-badge">Live Speech Engine</span>
            </div>
            <div class="btn-row">
                <button class="btn-voice" onclick="playHindi()">
                    🇮🇳 Suniye (Hindi Audio)
                </button>
                <button class="btn-voice" onclick="playEnglish()">
                    🌐 Listen (English Audio)
                </button>
                <button class="btn-voice btn-stop" onclick="stopAudio()">
                    ⏹ Stop
                </button>
            </div>
            <div class="status-txt" id="audioStatus">Ready to stream audio advisory...</div>
        </div>
        <script>
            function playHindi() {{
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance("{hindi_text_js}");
                u.lang = "hi-IN";
                u.rate = 0.95;
                document.getElementById("audioStatus").innerText = "🔊 Bol rahe hain: Hindi Advisory...";
                u.onend = () => {{ document.getElementById("audioStatus").innerText = "Advisory audio finished."; }};
                window.speechSynthesis.speak(u);
            }}
            function playEnglish() {{
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance("{english_text_js}");
                u.lang = "en-US";
                u.rate = 1.0;
                document.getElementById("audioStatus").innerText = "🔊 Playing: English Clinical Advisory...";
                u.onend = () => {{ document.getElementById("audioStatus").innerText = "Advisory audio finished."; }};
                window.speechSynthesis.speak(u);
            }}
            function stopAudio() {{
                window.speechSynthesis.cancel();
                document.getElementById("audioStatus").innerText = "Audio stopped.";
            }}
        </script>
    </body>
    </html>
    """
    components.html(audio_html, height=125)


def render_weather_risk_simulator():
    """Renders the Fungal Sporulation & Weather Risk Simulator widget."""
    with st.expander("🌦️ Field Weather Risk & Fungal Sporulation Index (FSI) Simulator", expanded=False):
        render_html("""
        <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 12px;">
            Simulate ambient canopy microclimate (temperature, relative humidity, and foliar leaf wetness) to predict high-risk pathogen sporulation windows before spraying.
        </div>
        """)

        w1, w2, w3 = st.columns(3)
        with w1:
            sim_temp = st.slider("Canopy Temp (°C):", 12.0, 40.0, 22.0, 0.5, key="sim_temp")
        with w2:
            sim_rh = st.slider("Relative Humidity (%):", 35.0, 99.0, 86.0, 1.0, key="sim_rh")
        with w3:
            sim_wet = st.slider("Leaf Wetness (Hours):", 0.0, 24.0, 8.0, 1.0, key="sim_wet")

        # Agronomic model: Fungal pathogens thrive at 18-25 C and RH > 80% with >6h wetness
        temp_factor = max(0.0, 1.0 - abs(sim_temp - 22.0) / 10.0)
        rh_factor = max(0.0, (sim_rh - 40.0) / 60.0)
        wet_factor = min(1.0, sim_wet / 6.0)

        risk_score = temp_factor * rh_factor * wet_factor * 100.0

        if risk_score >= 70.0:
            risk_badge = "🔴 CRITICAL EPIDEMIC OUTBREAK WINDOW (>70%)"
            risk_color = "#EF4444"
            risk_advice = "Atmospheric conditions are optimal for rapid fungal germination and spore dispersal. Apply protective systemic fungicide (e.g. Mancozeb) immediately before imminent precipitation."
        elif risk_score >= 35.0:
            risk_badge = "🟡 MODERATE SPORULATION RISK (35-70%)"
            risk_color = "#F59E0B"
            risk_advice = "Microclimate favors moderate foliar incubation. Intensify crop scouting on lower shaded leaves and optimize canopy airflow."
        else:
            risk_badge = "🟢 LOW EPIDEMIOLOGICAL THREAT (<35%)"
            risk_color = "#10B981"
            risk_advice = "Ambient conditions suppress fungal sporulation. No emergency chemical intervention required."

        render_html(f"""
        <div style="background: rgba(10, 18, 30, 0.85); border: 1px solid {risk_color}50; border-radius: 12px; padding: 14px 18px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.80rem; font-weight: 700; color: {risk_color}; text-transform: uppercase;">{risk_badge}</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; font-weight: 800; color: {risk_color};">{risk_score:.1f}% Risk</span>
            </div>
            <div style="height: 8px; background: rgba(255,255,255,0.08); border-radius: 8px; overflow: hidden; margin-bottom: 10px;">
                <div style="height: 100%; width: {min(risk_score, 100.0):.1f}%; background: linear-gradient(90deg, #10B981 0%, #F59E0B 50%, #EF4444 100%);"></div>
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.45;">
                <strong>Field Management Action:</strong> {risk_advice}
            </div>
        </div>
        """)


# -------------------------------------------------------------
# 4. SECTION 4: HERO SECTION (IMPRESSIVE DASHBOARD HEADER)
# -------------------------------------------------------------
render_html("""
<div class="hero-header-card">
    <div class="hero-brand-left">
        <div class="hero-logo-box">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
                <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
            </svg>
        </div>
        <div class="hero-title-wrap">
            <div class="hero-title-row">
                <h1 class="hero-title-text">AgriGuard AI</h1>
                <span class="enterprise-badge">ENTERPRISE AGRITECH</span>
            </div>
            <div class="hero-tagline">Intelligent Foliar Pathology Recognition &bull; Domain-Invariant Neural Diagnostics</div>
            <div class="hero-status-row">
                <span class="status-pulse-dot"></span>
                <span>Neural Inference Engine Active &bull; ResNet50 + MixStyle SOTA</span>
                <span class="hero-telemetry-badge">⚡ 109ms (INT8 Edge) &bull; 60 FPS UI</span>
            </div>
        </div>
    </div>
    <div class="hero-metrics-grid">
        <div class="hero-stat-card">
            <span class="hero-stat-val">99.37%</span>
            <span class="hero-stat-lbl">In-Domain Val Acc</span>
        </div>
        <div class="hero-stat-card">
            <span class="hero-stat-val">109 ms</span>
            <span class="hero-stat-lbl">Edge Latency</span>
        </div>
        <div class="hero-stat-card">
            <span class="hero-stat-val">24.66M</span>
            <span class="hero-stat-lbl">Parameters</span>
        </div>
        <div class="hero-stat-card">
            <span class="hero-stat-val">23.86 MB</span>
            <span class="hero-stat-lbl">Mobile Size</span>
        </div>
    </div>
</div>
""")


# -------------------------------------------------------------
# 5. SECTION 5: MODERN HORIZONTAL NAVIGATION
# -------------------------------------------------------------
tab_live, tab_benchmarks, tab_advisory, tab_edge = st.tabs([
    "Live Pathology Hub",
    "Research Benchmark Lab",
    "Agronomic Treatment Directory",
    "Mobile Edge Studio (TFLite)",
])


# =============================================================
# TAB 1: LIVE PATHOLOGY HUB
# =============================================================
with tab_live:
    # ---------------------------------------------------------
    # SECTION 2: COMPACT QUICK AI DEMONSTRATIONS (7 Cards)
    # ---------------------------------------------------------
    active_specimen_desc = st.session_state.get("active_sample_desc", None)
    active_specimen_html = f'<span class="active-specimen-badge">● Active: {active_specimen_desc}</span>' if active_specimen_desc else ''

    render_html(f'<div class="quick-demos-container">'
        f'<div class="quick-demos-header-row">'
        f'<div class="quick-demos-head"><span>⚡ Quick AI Demonstrations</span>{active_specimen_html}</div>'
        f'<span class="quick-demos-sub">1-Click verified laboratory &amp; field specimens with published benchmark scores:</span>'
        f'</div></div>'
    )

    active_path_str = st.session_state.get("active_sample_path", "")
    active_btn_key = None
    if "01_Potato" in active_path_str: active_btn_key = "btn_pot"
    elif "02_Tomato" in active_path_str: active_btn_key = "btn_tom"
    elif "03_Corn" in active_path_str: active_btn_key = "btn_crn"
    elif "04_Apple" in active_path_str: active_btn_key = "btn_app"
    elif "05_Grape" in active_path_str: active_btn_key = "btn_grp"
    elif "06_Bell" in active_path_str: active_btn_key = "btn_pep"
    elif "07_Field" in active_path_str: active_btn_key = "btn_pot_field"

    if active_btn_key:
        render_html(
            f"<style>.st-key-{active_btn_key} button {{ "
            f"border-color: #10B981 !important; "
            f"background: linear-gradient(135deg, rgba(16, 185, 129, 0.32) 0%, rgba(6, 182, 212, 0.22) 100%) !important; "
            f"box-shadow: 0 0 20px rgba(16, 185, 129, 0.5) !important; "
            f"transform: translateY(-2px) !important; "
            f"}}</style>")

    d1, d2, d3, d4, d5, d6, d7 = st.columns(7)
    with d1:
        if st.button("Potato\n95.4% SOTA", key="btn_pot", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/01_Potato_Early_Blight_95pct.jpg"
            st.session_state["active_sample_desc"] = "Potato: Early Blight (95.4% Benchmark)"
            st.session_state["active_zoom"] = False
    with d2:
        if st.button("Tomato\n92.2% SOTA", key="btn_tom", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/02_Tomato_Early_Blight_92pct.jpg"
            st.session_state["active_sample_desc"] = "Tomato: Early Blight (92.2% Benchmark)"
            st.session_state["active_zoom"] = False
    with d3:
        if st.button("Corn\n90.7% SOTA", key="btn_crn", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/03_Corn_Common_Rust_91pct.jpg"
            st.session_state["active_sample_desc"] = "Corn: Common Rust (90.7% Benchmark)"
            st.session_state["active_zoom"] = False
    with d4:
        if st.button("Apple\n96.1% SOTA", key="btn_app", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/04_Apple_Scab_96pct.jpg"
            st.session_state["active_sample_desc"] = "Apple: Scab (96.1% Benchmark)"
            st.session_state["active_zoom"] = False
    with d5:
        if st.button("Grape\n80.5% SOTA", key="btn_grp", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/05_Grape_Black_Rot_81pct.jpg"
            st.session_state["active_sample_desc"] = "Grape: Black Rot (80.5% Benchmark)"
            st.session_state["active_zoom"] = False
    with d6:
        if st.button("Pepper\n84.1% SOTA", key="btn_pep", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/06_Bell_Pepper_Spot_84pct.jpg"
            st.session_state["active_sample_desc"] = "Bell Pepper: Bacterial Spot (84.1% Benchmark)"
            st.session_state["active_zoom"] = False
    with d7:
        if st.button("Wild Potato\n94.6% Field", key="btn_pot_field", use_container_width=True):
            st.session_state["active_sample_path"] = "presentation_test_images/07_Field_Potato_Late_Blight_95pct.jpg"
            st.session_state["active_sample_desc"] = "Wild Potato: Late Blight (Real Field Multi-Leaf)"
            st.session_state["active_zoom"] = True

    # ---------------------------------------------------------
    # SECTION 1 & 6: BALANCED 45% / 55% TWO-COLUMN DASHBOARD
    # ---------------------------------------------------------
    col_input, col_pred = st.columns([1, 1.22], gap="medium")

    active_image = None
    source_title = ""

    # LEFT PANEL — LEAF IMAGE ACQUISITION (~45% width)
    with col_input:
        with st.container(border=True):
            render_html("""
            <div class="panel-card-title">🌿 Analyze Your Crop</div>
            <div class="panel-card-subtitle">Upload a leaf image to identify potential diseases using AI.</div>
            """)

            # Three Input Methods (Segmented Pills)
            input_source = st.segmented_control(
                "Input Method:",
                options=["Upload Leaf", "Curated Gallery", "Device Camera"],
                default="Upload Leaf",
                key="source_mode_pills",
                label_visibility="collapsed",
            )

            # METHOD 1: Drag & Drop Upload Zone
            if input_source == "Upload Leaf":
                uploaded_file = st.file_uploader(
                    "Drag and drop your leaf image here (or browse files):",
                    type=["jpg", "jpeg", "png"],
                    key="leaf_uploader_widget",
                    help="Upload a plant leaf image. Supports JPG, JPEG, PNG (Limit: 200MB)."
                )
                if uploaded_file is not None:
                    active_image = Image.open(uploaded_file).convert("RGB")
                    source_title = f"Uploaded File: {uploaded_file.name}"
                    st.session_state.pop("active_sample_path", None)
                    st.session_state.pop("active_sample_desc", None)
                elif "active_sample_path" in st.session_state:
                    p = PROJECT_ROOT / st.session_state["active_sample_path"]
                    if p.exists():
                        active_image = Image.open(p).convert("RGB")
                        source_title = st.session_state.get("active_sample_desc", "Selected Benchmark Specimen")

            # METHOD 2: Curated Benchmark Gallery
            elif input_source == "Curated Gallery":
                gallery_presets = [
                    {"label": "Potato Early Blight (95.4% Benchmark)", "path": "presentation_test_images/01_Potato_Early_Blight_95pct.jpg", "zoom": False},
                    {"label": "Tomato Early Blight (92.2% Benchmark)", "path": "presentation_test_images/02_Tomato_Early_Blight_92pct.jpg", "zoom": False},
                    {"label": "Corn Common Rust (90.7% Benchmark)", "path": "presentation_test_images/03_Corn_Common_Rust_91pct.jpg", "zoom": False},
                    {"label": "Apple Scab (96.1% Benchmark)", "path": "presentation_test_images/04_Apple_Scab_96pct.jpg", "zoom": False},
                    {"label": "Grape Black Rot (80.5% Benchmark)", "path": "presentation_test_images/05_Grape_Black_Rot_81pct.jpg", "zoom": False},
                    {"label": "Bell Pepper Bacterial Spot (84.1% Benchmark)", "path": "presentation_test_images/06_Bell_Pepper_Spot_84pct.jpg", "zoom": False},
                    {"label": "Wild Field Potato Late Blight (94.6% Field)", "path": "presentation_test_images/07_Field_Potato_Late_Blight_95pct.jpg", "zoom": True},
                ]
                chosen_item = st.selectbox(
                    "Select a Curated Specimen:",
                    gallery_presets,
                    format_func=lambda x: x["label"],
                    key="gallery_selector"
                )
                p_gallery = PROJECT_ROOT / chosen_item["path"]
                if p_gallery.exists():
                    active_image = Image.open(p_gallery).convert("RGB")
                    source_title = chosen_item["label"]
                    st.session_state["active_zoom"] = chosen_item["zoom"]
                    st.session_state.pop("active_sample_path", None)

            # METHOD 3: Device Camera
            elif input_source == "Device Camera":
                allow_camera = st.checkbox("Activate Device Camera", value=False)
                if allow_camera:
                    cam_file = st.camera_input("Capture leaf snapshot", key="camera_widget")
                    if cam_file is not None:
                        active_image = Image.open(cam_file).convert("RGB")
                        source_title = "Device Camera Snapshot"
                        st.session_state.pop("active_sample_path", None)

            # Image Preview & Focal Controls
            if active_image is not None:
                w_orig, h_orig = active_image.size
                render_html(f'<div class="specimen-scanner-header">'
                    f'<div class="specimen-scanner-title"><span class="status-pulse-dot" style="width: 7px; height: 7px;"></span><span>Active Specimen Intake</span></div>'
                    f'<span class="specimen-scanner-meta">{w_orig} × {h_orig} px &bull; RGB</span>'
                    f'</div>'
                )

                st.image(active_image, use_container_width=True)
                
                # Image metadata & remove button
                mcol1, mcol2 = st.columns([2.3, 1.2])
                with mcol1:
                    render_html(f"<div class='specimen-source-tag'>🏷️ {source_title}</div>")
                with mcol2:
                    if st.button("✕ Reset Specimen", key="btn_remove_active", use_container_width=True):
                        st.session_state.pop("active_sample_path", None)
                        st.session_state.pop("active_sample_desc", None)
                        st.rerun()

                # Aspect ratio check
                is_wide = (w_orig / h_orig) > 1.3
                initial_zoom = st.session_state.get("active_zoom", is_wide)

                c_ctrl1, c_ctrl2 = st.columns([1, 1.25])
                with c_ctrl1:
                    focal_zoom = st.toggle(
                        "🔍 Center Focus (Smart Crop)",
                        value=initial_zoom,
                        help="Zooms into the center leaf blade to eliminate background noise."
                    )
                with c_ctrl2:
                    crop_focus = st.selectbox(
                        "🌾 Botanical Crop Prior:",
                        list(CROP_FAMILY_MAP.keys()),
                        index=0,
                        key="crop_focus_box",
                        help="Select a crop family (Bayesian domain prior) or use Auto-Detect across all 28 classes."
                    )

                with st.expander("⚙️ Advanced Grad-CAM Settings", expanded=False):
                    heatmap_alpha = st.slider("Grad-CAM Transparency (α)", 0.1, 0.9, 0.50, 0.05)
            else:
                focal_zoom = False
                crop_focus = "🌾 Auto-Detect (All 28 Classes)"
                heatmap_alpha = 0.50

            # ---------------------------------------------------------
            # SECTION 6: EXPANDABLE SUPPORTED CROPS EXPLORER
            # ---------------------------------------------------------
            render_html("<div style='margin-top: 14px;'></div>")
            with st.expander("🌾 Supported Botanical Crops Explorer (28 Canonical Classes)", expanded=False):
                search_crop = st.text_input("🔍 Filter Crops:", "", key="crop_search_filter")
                for crop_name, classes in CROP_GROUPS.items():
                    if search_crop and search_crop.lower() not in crop_name.lower():
                        continue
                    render_html(f"**{crop_name}** (`{len(classes)} diseases`)")
                    for c in classes:
                        st.caption(f"&bull; {CANONICAL_DISPLAY_NAMES[c]}")

    # RIGHT PANEL — AI DIAGNOSIS (~55% width)
    with col_pred:
        with st.container(border=True):
            render_html("""
            <div class="panel-card-title">🔬 AI Disease Diagnosis</div>
            <div class="panel-card-subtitle">Your intelligent crop health analysis starts here.</div>
            """)

            if active_image is None:
                # SECTION 4: PREMIUM EMPTY DIAGNOSIS STATE (CONNECTED 3-STEP WORKFLOW STEPPER)
                render_html("""<div class="empty-workflow-card">
<div class="radar-scan-box">
    <svg class="empty-graphic-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>
        <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
</div>
<div class="empty-title">Awaiting Foliar Specimen</div>
<div class="empty-desc">Select a verified benchmark specimen above or acquire a leaf image on the left to begin AI pathology inference.</div>
<div class="stepper-track">
<div class="stepper-step stepper-step-active">
    <div class="stepper-badge"><span class="stepper-num">01 / INTAKE</span><span class="stepper-tag tag-active">● Active Step</span></div>
    <div class="stepper-head">Acquire Foliar Image</div>
    <div class="stepper-sub">Upload leaf photo, gallery preset, or live camera snapshot</div>
</div>
<div class="stepper-connector"></div>
<div class="stepper-step">
    <div class="stepper-badge"><span class="stepper-num">02 / INFERENCE</span><span class="stepper-tag tag-pending">○ Pending</span></div>
    <div class="stepper-head">ResNet50 + MixStyle</div>
    <div class="stepper-sub">Domain-generalized neural network extracts deep invariant cytological features</div>
</div>
<div class="stepper-connector"></div>
<div class="stepper-step">
    <div class="stepper-badge"><span class="stepper-num">03 / CLINICAL RX</span><span class="stepper-tag tag-pending">○ Pending</span></div>
    <div class="stepper-head">Grad-CAM &amp; Prescriptions</div>
    <div class="stepper-sub">Quantify foliar lesion severity and access certified biological &amp; chemical agronomic regimens</div>
</div>
</div>
</div>""")

            elif model is None:
                st.error("Model weights file `models/best_model.keras` not found. Please ensure weights are present.")
            else:
                with st.spinner("Analyzing foliar cytology & computing Grad-CAM saliency..."):
                    t0 = time.time()
                    leaf_img = active_image
                    if focal_zoom:
                        w, h = active_image.size
                        leaf_img = active_image.crop((int(w * 0.20), int(h * 0.10), int(w * 0.80), int(h * 0.95)))

                    img_224 = leaf_img.resize((224, 224))
                    arr_224 = np.array(img_224, dtype=np.float32)
                    tensor = normalize_image(tf.convert_to_tensor(arr_224))
                    tensor_batch = tf.expand_dims(tensor, axis=0)

                    # Forward Pass (Global Unconditioned)
                    raw_global_preds = model(tensor_batch, training=False).numpy()[0]
                    latency_ms = (time.time() - t0) * 1000.0

                    global_top_idx = int(np.argmax(raw_global_preds))
                    global_top_prob = float(raw_global_preds[global_top_idx])
                    global_top_canonical = CANONICAL_CLASSES[global_top_idx]
                    global_top_display = CANONICAL_DISPLAY_NAMES[global_top_canonical]
                    predictive_entropy = float(-np.sum(raw_global_preds * np.log(np.maximum(raw_global_preds, 1e-12))))

                    # Crop Prior Conditioning Logic
                    selected_classes = CROP_FAMILY_MAP.get(crop_focus)
                    is_prior_active = (selected_classes is not None)

                    if is_prior_active:
                        crop_indices = [CANONICAL_CLASSES.index(k) for k in selected_classes if k in CANONICAL_CLASSES]
                        raw_crop_probs = np.array([raw_global_preds[i] for i in crop_indices])
                        denom = raw_crop_probs.sum()
                        norm_probs = raw_crop_probs / denom if denom > 0 else raw_crop_probs
                        sorted_local = np.argsort(norm_probs)[::-1]
                        top_idx = crop_indices[sorted_local[0]]
                        top_prob = float(norm_probs[sorted_local[0]])
                        top_canonical = CANONICAL_CLASSES[top_idx]
                        top_display = CANONICAL_DISPLAY_NAMES[top_canonical]
                        crop_category = crop_focus.split()[1] if len(crop_focus.split()) > 1 else "Target Crop"
                        top3_indices = [crop_indices[i] for i in sorted_local[:min(3, len(sorted_local))]]
                        top3_probs = [float(norm_probs[i]) for i in sorted_local[:min(3, len(sorted_local))]]
                    else:
                        top3_indices = np.argsort(raw_global_preds)[-3:][::-1]
                        top_idx = top3_indices[0]
                        top_prob = float(raw_global_preds[top_idx])
                        top_canonical = CANONICAL_CLASSES[top_idx]
                        top_display = CANONICAL_DISPLAY_NAMES[top_canonical]
                        crop_category = "General"
                        for crop, clist in CROP_GROUPS.items():
                            if top_canonical in clist:
                                crop_category = crop
                                break
                        top3_probs = [float(raw_global_preds[i]) for i in top3_indices]

                    # Grad-CAM Heatmap
                    try:
                        heatmap = compute_gradcam_heatmap(model, tensor_batch, class_idx=top_idx)
                        overlay = overlay_gradcam(np.array(img_224, dtype=np.uint8), heatmap, alpha=heatmap_alpha)
                    except Exception as e:
                        heatmap, overlay = None, None

                # Advisory lookup
                advisory = TREATMENT_ADVISORY.get(top_canonical, {})
                pathogen_name = advisory.get("pathogen", "Pathogen Not Documented")

                # Shared Pathology Harmonization (e.g. Potato/Tomato Early Blight both Alternaria solani; Late Blight both Phytophthora infestans)
                top_disease_core = top_display.split(":")[-1].strip()
                second_display = CANONICAL_DISPLAY_NAMES[CANONICAL_CLASSES[top3_indices[1]]] if len(top3_indices) > 1 else ""
                second_prob = top3_probs[1] if len(top3_probs) > 1 else 0.0

                shared_pathology = False
                shared_pathogen_name = ""
                if "Early Blight" in top_disease_core and "Early Blight" in second_display:
                    shared_pathology = True
                    shared_pathogen_name = "Alternaria solani (Solanaceae Early Blight)"
                elif "Late Blight" in top_disease_core and "Late Blight" in second_display:
                    shared_pathology = True
                    shared_pathogen_name = "Phytophthora infestans (Solanaceae Late Blight)"
                elif top_disease_core in second_display and any(term in top_disease_core for term in ["Spot", "Mildew", "Rot"]):
                    shared_pathology = True
                    shared_pathogen_name = f"Shared {top_disease_core} Complex"

                combined_certainty = (top_prob + second_prob) if (shared_pathology and not is_prior_active) else top_prob

                # Statistical Rejection & Out-of-Distribution Gating
                # Validated rule: Reject if single prob < 28% and combined signal < 45%, or predictive entropy > 2.35 with low prob < 40%
                is_uncertain = (
                    (top_prob < 0.28 and combined_certainty < 0.45) or 
                    (predictive_entropy > 2.35 and top_prob < 0.40)
                )

                if is_uncertain:
                    box_class = "diag-hero-box diag-hero-inconclusive"
                    pill_class = "badge-red"
                    status_text = f"REJECTED SPECIMEN &bull; {top_prob * 100:.1f}% Below Safety Bar"
                    pill_header = "⚠️ UNCONFIRMED ATYPICAL SPECIMEN"
                    hero_title = "Uncertain or Unsupported Specimen — Further Agricultural Inspection Recommended"
                    hero_sub = f"Foliar Ambiguity (Top Signal: {top_display} at {top_prob*100:.1f}%, Entropy: {predictive_entropy:.2f} nats)"
                elif is_prior_active:
                    box_class = "diag-hero-box"
                    pill_class = "badge-green" if top_prob >= 0.60 else "badge-amber"
                    status_text = f"CONDITIONAL PRIOR &bull; {top_prob * 100:.1f}% (Global Signal: {global_top_prob * 100:.1f}%)"
                    pill_header = f"🌿 Botanical Crop: {crop_category} (User Prior)"
                    hero_title = top_display
                    hero_sub = pathogen_name
                elif shared_pathology and top_prob < 0.60:
                    box_class = "diag-hero-box"
                    pill_class = "badge-green" if combined_certainty >= 0.55 else "badge-amber"
                    status_text = f"AGGREGATED PATHOGEN MASS &bull; {combined_certainty * 100:.1f}% (Shared Complex)"
                    pill_header = f"🌿 Botanical Crop: {crop_category}"
                    hero_title = top_display
                    hero_sub = pathogen_name
                else:
                    box_class = "diag-hero-box"
                    pill_class = "badge-green" if top_prob >= 0.50 else "badge-amber"
                    status_text = f"HIGH CONFIDENCE DIAGNOSIS &bull; {top_prob * 100:.1f}% Softmax Certainty"
                    pill_header = f"🌿 Botanical Crop: {crop_category}"
                    hero_title = top_display
                    hero_sub = pathogen_name

                # Completed Progress Stepper Banner
                render_html(f"""
                <div class="completed-stepper-bar">
                    <div class="step-chip"><span>✓</span><span>01 Intake Acquired</span></div>
                    <div class="step-divider"></div>
                    <div class="step-chip"><span>✓</span><span>02 ResNet50+MixStyle ({latency_ms:.0f}ms)</span></div>
                    <div class="step-divider"></div>
                    <div class="step-chip"><span>✓</span><span>03 Clinical Rx Formulated</span></div>
                </div>
                """)

                # Circular Confidence Gauge Ring SVG
                ring_color = "#EF4444" if is_uncertain else ("#10B981" if top_prob >= 0.60 else "#F59E0B")
                gauge_svg_html = render_confidence_ring_svg(top_prob * 100.0, color=ring_color, size=86)

                # SECTION 5.A: PRIMARY DIAGNOSIS CARD WITH CIRCULAR GAUGE
                render_html(f"""
                <div class="{box_class} result-fade-in">
                    <div class="diag-hero-main-content">
                        <div class="crop-pill-tag">{pill_header}</div>
                        <div class="diag-primary-title">{hero_title}</div>
                        <div class="diag-pathogen-sub">{hero_sub}</div>
                        <div class="conf-status-badge {pill_class}">
                            <span>●</span>
                            <span>{status_text}</span>
                        </div>
                    </div>
                    <div class="diag-hero-gauge-container">
                        {gauge_svg_html}
                    </div>
                </div>
                """)

                if is_uncertain:
                    st.warning(
                        f"⚠️ **Uncertain or Unsupported Specimen — Further Agricultural Inspection Recommended:**\n\n"
                        f"The neural network evaluated the specimen across all 28 benchmark pathology classes, "
                        f"but the statistical confidence (MSP: {top_prob*100:.1f}%, Predictive Entropy: {predictive_entropy:.2f} nats) "
                        f"falls below the reliable diagnostic threshold.\n\n"
                        f"**Scientific Verification Factors:**\n"
                        f"• **Unsupported Foliar Disorder:** Symptoms may correspond to non-fungal abiotic stress (e.g. marginal edge scorch, drought, or nutrient deficiency) "
                        f"which are not part of the 28 validated pathogen classes. The system does not claim to diagnose unvalidated nutritional disorders.\n"
                        f"• **Atypical Foliage or Foreign Artifact:** Specimen exhibits atypical leaf geometry, non-plant background clutter, or diffuse foliar texture.\n"
                        f"• **Diagnostic Protocol:** Chemical fungicides are withheld. Conduct physical field scouting, laboratory tissue testing, or consult an agricultural extension officer."
                    )
                elif is_prior_active:
                    st.info(
                        f"ℹ️ **Bayesian Crop Prior Active ({crop_category}):**\n\n"
                        f"The user explicitly specified the crop family as **{crop_category}**. "
                        f"The conditional probability within {crop_category} is **{top_prob * 100:.1f}%** ($P(\\text{{Pathology}} \\mid \\text{{Crop}}, X)$). "
                        f"The unconditioned global model certainty is **{global_top_prob * 100:.1f}%** (*{global_top_display}*). "
                        f"This Bayesian prior restricts the search space to known {crop_category} diseases; it does not alter underlying neural network weights or represent an improvement in base model accuracy."
                    )
                elif shared_pathology and top_prob < 0.60:
                    st.info(
                        f"🔬 **Aggregated Probability Mass across Shared Pathogen Complex ({combined_certainty * 100:.1f}%):**\n\n"
                        f"Both Potato and Tomato share the identical biological pathogen **{shared_pathogen_name}**. "
                        f"The 28-class softmax distributes probability mass across mutually exclusive class outputs "
                        f"(*{top_display}*: {top_prob * 100:.1f}% + *{second_display}*: {second_prob * 100:.1f}%). "
                        f"The sum represents joint probability mass across the shared complex without double-counting; it is an aggregated diagnostic hypothesis, not a single-class calibrated posterior."
                    )

                # SECTION 5.B: AI PERFORMANCE METRICS (4 Metric Cards)
                if is_uncertain:
                    disp_score = f"{top_prob * 100:.1f}%"
                    disp_label = "Confidence (< 28% Bar)"
                    disp_crop = "Unconfirmed"
                    disp_pathogen = "Unsupported / Abiotic"
                elif is_prior_active:
                    disp_score = f"{top_prob * 100:.1f}%"
                    disp_label = f"Conditional Prior P(Y|C)"
                    disp_crop = f"{crop_category} (User Prior)"
                    disp_pathogen = "Fungal" if any(t in top_canonical for t in ["blight", "rust", "scab", "rot", "mildew"]) else ("Healthy" if "healthy" in top_canonical else "Bacterial/Viral")
                elif shared_pathology and top_prob < 0.60:
                    disp_score = f"{combined_certainty * 100:.1f}%"
                    disp_label = "Aggregated Mass"
                    disp_crop = crop_category
                    disp_pathogen = "Shared Complex"
                else:
                    disp_score = f"{top_prob * 100:.1f}%"
                    disp_label = "Confidence Score"
                    disp_crop = crop_category
                    disp_pathogen = "Healthy Foliage" if "healthy" in top_canonical else ("Fungal" if any(t in top_canonical for t in ["blight", "rust", "scab", "rot", "mildew"]) else "Bacterial/Viral")

                render_html(f"""
                <div class="telemetry-row">
                    <div class="telemetry-item">
                        <div class="telemetry-val">{disp_score}</div>
                        <div class="telemetry-lbl">{disp_label}</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-val">{disp_crop}</div>
                        <div class="telemetry-lbl">Predicted Crop</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-val">{disp_pathogen}</div>
                        <div class="telemetry-lbl">Pathology Class</div>
                    </div>
                    <div class="telemetry-item">
                        <div class="telemetry-val">{latency_ms:.0f} ms</div>
                        <div class="telemetry-lbl">Neural Latency</div>
                    </div>
                </div>
                """)

                # SECTION 5.B.2 PILLAR 2: DISEASE SEVERITY & FOLIAR DAMAGE QUANTIFICATION
                sev_info = calculate_disease_severity(heatmap, top_canonical)
                render_html(f"""
                <div class="severity-card">
                    <div class="severity-header-row">
                        <div class="severity-title">
                            <span>📊 Foliar Pathological Severity Index (FSI)</span>
                        </div>
                        <div class="severity-pct-val" style="color: {sev_info['stage_color']};">
                            {sev_info['severity_pct']:.1f}%
                        </div>
                    </div>
                    <div class="severity-meter-rail">
                        <div class="severity-meter-fill" style="width: {min(max(sev_info['severity_pct'], 4.0), 100.0):.1f}%; background: {sev_info['rail_gradient']};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 0.76rem; font-weight: 700; color: {sev_info['stage_color']}; background: rgba(0,0,0,0.35); border: 1px solid {sev_info['stage_color']}40; padding: 2px 10px; border-radius: 12px;">
                            {sev_info['stage_badge']}
                        </span>
                        <span style="font-size: 0.72rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                            Surface Lesion Quantification
                        </span>
                    </div>
                    <div class="severity-action-box">
                        <strong style="color: #A7F3D0;">Agronomic Protocol:</strong> {sev_info['action_protocol']}
                    </div>
                </div>
                """)

                # SECTION 5.C PILLAR 2: DIFFERENTIAL DIAGNOSIS WITH INFORMATION-THEORETIC TELEMETRY
                diff_title = "Top Candidate Matches (Below Safety Bar):" if is_uncertain else "Top-3 Differential Diagnostic Hypotheses:"
                delta_prob = (top_prob - second_prob) * 100.0 if len(top3_probs) > 1 else top_prob * 100.0
                certainty_status = "Decisive Consensus" if delta_prob > 35.0 else ("Moderate Discrimination" if delta_prob > 15.0 else "High Co-Occurrence Ambiguity")

                render_html(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin: 12px 0 6px 0;'>
                    <span style='font-size: 0.84rem; font-weight: 700; color: #F8FAFC; text-transform: uppercase; letter-spacing: 0.6px;'>{diff_title}</span>
                    <span style='font-size: 0.72rem; color: #34D399; font-family: "JetBrains Mono", monospace;'>Δ Margin: {delta_prob:.1f}%</span>
                </div>
                """)

                crop_icons = {
                    "potato": "🥔", "tomato": "🍅", "corn": "🌽", "apple": "🍎", 
                    "grape": "🍇", "bell": "🫑", "squash": "🎃", "peach": "🍑", 
                    "cherry": "🍒", "strawberry": "🍓"
                }

                for i, idx in enumerate(top3_indices):
                    cname = CANONICAL_CLASSES[idx]
                    dname = CANONICAL_DISPLAY_NAMES[cname]
                    prob = top3_probs[i]
                    pct = prob * 100.0
                    
                    prefix = cname.split("_")[0]
                    icon = crop_icons.get(prefix, "🌿")
                    
                    consensus_badge = " <span style='color: #10B981; font-size: 0.74rem; font-weight:600;'>[Shared Complex]</span>" if (shared_pathology and i == 1 and not is_prior_active) else ""

                    render_html(f"""
                    <div class="diff-item">
                        <div class="diff-header">
                            <span>{icon} {dname}{consensus_badge}</span>
                            <span class="diff-score">{pct:.2f}%</span>
                        </div>
                        <div class="diff-rail">
                            <div class="diff-fill" style="width: {min(pct, 100.0):.1f}%;"></div>
                        </div>
                    </div>
                    """)

                render_html(f"""
                <div class="entropy-telemetry-box">
                    <span>Shannon Entropy: <span class="entropy-mono-val">{predictive_entropy:.2f} nats</span></span>
                    <span>Decision Margin: <span class="entropy-mono-val">{certainty_status}</span></span>
                </div>
                """)

                # SECTION 5.D PILLAR 2: GRAD-CAM EXPLAINABLE AI (SLIDER & TRI-VIEW)
                render_html("<div style='margin-top: 16px;'></div>")
                render_html("""
                <div class="panel-card-title" style="font-size: 1.05rem;">
                    <span>🔍 Grad-CAM Explainable AI Saliency Studio</span>
                </div>
                """)
                st.caption("Visual explanation of cellular foliar regions that influenced the deep network's prediction. Warm regions confirm biological focus on necrotic lesions.")

                if overlay is not None:
                    overlay_pil = Image.fromarray(overlay)
                    
                    view_mode = st.radio(
                        "Grad-CAM Visualizer Mode:",
                        ["🎚️ Interactive Before/After Split Slider", "🖼️ Side-by-Side Saliency Studio"],
                        horizontal=True,
                        key="xai_view_mode_radio"
                    )
                    
                    if view_mode == "🎚️ Interactive Before/After Split Slider":
                        render_comparison_slider(img_224, overlay_pil, height=420)
                    else:
                        g1, g2, g3 = st.columns(3)
                        with g1:
                            st.image(img_224, use_container_width=True)
                            st.caption("1. Original Foliar Specimen (224×224)")
                        with g2:
                            st.image(heatmap, use_container_width=True, clamp=True)
                            st.caption("2. Grad-CAM Activation Heatmap")
                        with g3:
                            st.image(overlay, use_container_width=True)
                            st.caption(f"3. Lesion Overlay (α={heatmap_alpha:.2f})")

                # SECTION 5.E & 5.F: DISEASE INFORMATION & TREATMENT RECOMMENDATIONS
                render_html("<div style='margin-top: 14px;'></div>")
                if is_uncertain:
                    render_html("""
                    <div class="panel-card-title" style="font-size: 1.02rem; margin-bottom: 6px;">
                        <span>🌾 Agronomic Protocol & Clinical Safety Action</span>
                    </div>
                    """)
                    st.info(
                        "🛡️ **Chemical Treatment Protocol Withheld (Clinical Safety Guardrail):**\n\n"
                        "Because diagnostic certainty is below the reliable threshold, spraying targeted fungicides is **NOT recommended** "
                        "to prevent chemical waste, soil toxicity, and phytotoxic leaf damage.\n\n"
                        "**Agronomic Protocol for Unconfirmed / Atypical Foliage:**\n"
                        "• **Do Not Spray Chemical Pesticides:** Withhold fungicides until a verified biotic pathogen is confirmed.\n"
                        "• **Check Abiotic Factors:** Inspect soil moisture, salinity, root health, and temperature extremes (drought/sunscald).\n"
                        "• **Agronomic Tissue Testing:** Submit foliar tissue samples to an accredited plant pathology lab to test for nutritional deficiencies (e.g., Potassium/Magnesium) or latent viral/bacterial infections.\n"
                        "• **Physical Scouting:** Inspect the abaxial (underside) leaf surface with a 10x hand lens for mites, thrips, or aphids."
                    )
                elif advisory:
                    render_html("""
                    <div class="panel-card-title" style="font-size: 1.05rem; margin-top: 14px; margin-bottom: 8px;">
                        <span>🌾 Clinical Agronomic Treatment &amp; Management Plan</span>
                    </div>
                    """)

                    rx_t1, rx_t2, rx_t3, rx_t4 = st.tabs([
                        "🌿 Organic & Biological",
                        "🧪 Targeted Chemical",
                        "🛡️ Cultural & Preventive",
                        "🔍 Pathogen & Symptoms",
                    ])

                    with rx_t1:
                        render_html(f"""
                        <div class="rx-treatment-card rx-card-organic">
                            <div class="rx-card-badge">Biological IPM Standard</div>
                            <div class="rx-card-title">🌿 Certified Organic Bio-Fungicides &amp; Botanical Controls</div>
                            <div class="rx-card-content">{advisory.get('organic_treatment', 'N/A')}</div>
                            <div class="rx-card-footer">
                                <span>Application Frequency: Every 7–10 days during high foliar humidity</span>
                            </div>
                        </div>
                        """)

                    with rx_t2:
                        render_html(f"""
                        <div class="rx-treatment-card rx-card-chemical">
                            <div class="rx-card-badge" style="background: rgba(245, 158, 11, 0.18); color: #FCD34D; border-color: rgba(245, 158, 11, 0.4);">Systemic Fungicide</div>
                            <div class="rx-card-title">🧪 Targeted Chemical Regimen &amp; Mycelial Inhibition</div>
                            <div class="rx-card-content">{advisory.get('chemical_treatment', 'N/A')}</div>
                            <div class="rx-card-footer">
                                <span>⚠️ Pre-Harvest Interval (PHI): Observe 14-day safety window before harvest</span>
                            </div>
                        </div>
                        """)

                    with rx_t3:
                        render_html(f"""
                        <div class="rx-treatment-card rx-card-preventive">
                            <div class="rx-card-badge" style="background: rgba(6, 182, 212, 0.18); color: #67E8F9; border-color: rgba(6, 182, 212, 0.4);">Cultural Agronomy</div>
                            <div class="rx-card-title">🛡️ Cultural Management &amp; Microclimate Prophylaxis</div>
                            <div class="rx-card-content">
                                &bull; <strong>Canopy Aeration:</strong> Thin dense vegetative shoots to maximize horizontal airflow and rapid morning dew evaporation.<br>
                                &bull; <strong>Micro-Drip Irrigation:</strong> Transition from overhead sprinkler watering to soil-level drip lines to eliminate foliar splash-borne spore dispersal.<br>
                                &bull; <strong>Crop Rotation Protocol:</strong> Avoid successive Solanaceae/Cucurbit plantings in consecutive seasons; rotate with non-host legumes or cereals for at least 24 months.<br>
                                &bull; <strong>Sanitation Pruning:</strong> Prune lower senescence leaves within 15 cm of soil line using 70% isopropyl alcohol sanitized shears.
                            </div>
                            <div class="rx-card-footer">
                                <span>Preventive Hygiene Index: High (Reduces primary fungal inoculum by up to 65%)</span>
                            </div>
                        </div>
                        """)

                    with rx_t4:
                        render_html(f"""
                        <div class="rx-treatment-card rx-card-etiology">
                            <div class="rx-card-badge" style="background: rgba(139, 92, 246, 0.18); color: #C4B5FD; border-color: rgba(139, 92, 246, 0.4);">Pathology Dossier</div>
                            <div class="rx-card-title">🦠 Causal Pathogen &amp; Diagnostic Symptoms</div>
                            <div class="rx-card-content">
                                <strong>Biological Pathogen:</strong> {advisory.get('pathogen', 'N/A')}<br><br>
                                <strong>Clinical Foliar Symptoms:</strong> {advisory.get('symptoms', 'N/A')}
                            </div>
                            <div class="rx-card-footer">
                                <span>Taxonomic Authority: CABI Compendium &bull; EPPO Global Database</span>
                            </div>
                        </div>
                        """)

                    st.caption("⚠️ *Note: Chemical control advice should be confirmed by a local agricultural extension officer based on regional fungicide resistance and crop stage.*")

                    # PILLAR 3.A: MULTILINGUAL VOICE AUDIO ADVISORY
                    render_html("<div style='margin-top: 14px;'></div>")
                    render_voice_advisory(
                        crop_name=crop_category,
                        disease_name=top_display,
                        certainty_pct=top_prob * 100.0,
                        severity_pct=sev_info["severity_pct"],
                        organic_rx=advisory.get("organic_treatment", "Eco-friendly bio-control"),
                        chemical_rx=advisory.get("chemical_treatment", "Targeted fungicide spray"),
                    )

                    # PILLAR 3.B: 1-CLICK CERTIFIED PDF PRESCRIPTION EXPORT
                    try:
                        specimen_code = f"AGRI-{int(time.time()) % 100000:05d}"
                        pdf_payload = generate_agronomic_pdf_prescription(
                            specimen_id=specimen_code,
                            crop_name=crop_category,
                            disease_name=top_display,
                            pathogen=pathogen_name,
                            certainty_pct=top_prob * 100.0,
                            severity_pct=sev_info["severity_pct"],
                            stage_name=sev_info["stage_name"],
                            organic_treatment=advisory.get("organic_treatment", ""),
                            chemical_treatment=advisory.get("chemical_treatment", ""),
                            symptoms=advisory.get("symptoms", ""),
                            raw_img=img_224,
                            overlay_img=Image.fromarray(overlay) if overlay is not None else img_224,
                        )
                        st.download_button(
                            label="📥 Download Certified Agronomic Prescription (PDF Report)",
                            data=pdf_payload,
                            file_name=f"AgriGuard_Prescription_{top_canonical}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="btn_download_pdf",
                        )
                    except Exception as e:
                        st.caption(f"PDF Generation Note: {e}")

                    # PILLAR 3.C: FUNGAL OUTBREAK & WEATHER RISK SIMULATOR
                    render_html("<div style='margin-top: 12px;'></div>")
                    render_weather_risk_simulator()


# =============================================================
# TAB 2: RESEARCH BENCHMARK LAB
# =============================================================
with tab_benchmarks:
    with st.container(border=True):
        render_html("""
        <div class="panel-card-title">📊 Empirical Proof of Covariate Shift & Domain Generalization</div>
        <div class="panel-card-subtitle">Quantitative evidence demonstrating laboratory-to-field performance degradation and our MixStyle regularization resolution.</div>
        """)

        # Top 4 KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Validation Accuracy", "99.37%", help="In-domain laboratory benchmark on PlantVillage test set (not guaranteed 100% field accuracy)")
        k2.metric("Target Zero-Shot Acc", "22.50%", help="Direct out-of-domain transfer to wild PlantDoc field photos")
        k3.metric("Laplacian Blur Variance", "5.3x", help="5.3-fold increase in optical blur variance in real field foliage")
        k4.metric("Model Parameters", "24.66 M", help="ResNet50 backbone + MixStyle domain adaptation")

    render_html("<div style='margin-top: 14px;'></div>")
    
    # PILLAR 4: SUB-NAVIGATION FOR SCIENTIFIC LAB
    subtab_paper, subtab_eda, subtab_table = st.tabs([
        "📑 IEEE Publication Figures (Fig 1 to Fig 5)",
        "🔬 Empirical Covariate Shift Proofs",
        "📊 Comparative Benchmark Matrix (Table I)",
    ])

    with subtab_paper:
        p_sel = st.pills(
            "Select Figure to Inspect:",
            [
                "Fig 1: Methodology Workflow",
                "Fig 2: Percentage Metrics",
                "Fig 3: Domain Shift Trends",
                "Fig 4: Trade-Off Analysis",
                "Fig 5: Radar Evaluation",
                "Matrix: Out-of-Domain Confusion",
            ],
            default="Fig 1: Methodology Workflow",
            key="pills_paper_figs",
        )

        fig_map = {
            "Fig 1: Methodology Workflow": ("figures/fig1_flowchart.png", "Figure 1: End-to-End AgriGuard AI Architecture & Domain-Generalization Pipeline."),
            "Fig 2: Percentage Metrics": ("figures/fig2_percentage_scores.png", "Figure 2: Percentage score comparison among models on 28 botanical classes."),
            "Fig 3: Domain Shift Trends": ("figures/fig3_domain_performance.png", "Figure 3: Performance degradation trends across progressive domain shift levels."),
            "Fig 4: Trade-Off Analysis": ("figures/fig4_performance_vs_interpretability.png", "Figure 4: Model Performance vs. Inference Latency & Interpretability Trade-off Analysis."),
            "Fig 5: Radar Evaluation": ("figures/fig5_radar_evaluation.png", "Figure 5: Multi-dimensional Comprehensive Model Evaluation (Radar Chart)."),
            "Matrix: Out-of-Domain Confusion": ("results/figures/confusion_matrix.png", "Empirical 28×28 Out-of-Domain Confusion Matrix on wild PlantDoc field test."),
        }

        rel_path, cap_text = fig_map[p_sel]
        fig_disk_path = PROJECT_ROOT / rel_path
        if fig_disk_path.exists():
            st.image(str(fig_disk_path), caption=cap_text, use_container_width=True)
            with open(fig_disk_path, "rb") as f:
                st.download_button(
                    label=f"📥 Download 300 DPI High-Res Image ({fig_disk_path.name})",
                    data=f.read(),
                    file_name=fig_disk_path.name,
                    mime="image/png",
                    key=f"dl_{fig_disk_path.stem}",
                )
        else:
            st.warning(f"Figure file not found at {rel_path}.")

    with subtab_eda:
        st.markdown("##### 🔬 Optical Quality Breakdown & Covariate Shift Analysis")
        f1, f2 = st.columns(2)
        with f1:
            with st.container(border=True):
                img_blur = PROJECT_ROOT / "results" / "eda" / "image_quality_shift.png"
                if img_blur.exists():
                    st.image(str(img_blur), caption="Laplacian Sharpness & Contrast Breakdown (5.3x Blur in Real Field)", use_container_width=True)

            with st.container(border=True):
                img_dist = PROJECT_ROOT / "results" / "eda" / "class_distribution_comparison.png"
                if img_dist.exists():
                    st.image(str(img_dist), caption="Class Frequency Comparison (PlantVillage vs PlantDoc)", use_container_width=True)

            with st.container(border=True):
                img_aug = PROJECT_ROOT / "results" / "eda" / "augmentation_sample_grid.png"
                if img_aug.exists():
                    st.image(str(img_aug), caption="Domain-Robust Augmentation Pipeline (MixUp, Blur, Shadow)", use_container_width=True)

        with f2:
            with st.container(border=True):
                img_color = PROJECT_ROOT / "results" / "eda" / "color_space_domain_shift.png"
                if img_color.exists():
                    st.image(str(img_color), caption="Empirical Proof of Covariate Shift (RGB & HSV Color Distributions)", use_container_width=True)

            with st.container(border=True):
                img_cm = PROJECT_ROOT / "results" / "figures" / "confusion_matrix.png"
                if img_cm.exists():
                    st.image(str(img_cm), caption="28x28 Out-of-Domain Confusion Matrix (PlantDoc Field Test)", use_container_width=True)

    with subtab_table:
        st.markdown("##### 📊 Table I: Performance Benchmark Across Architectures on 28 Pathological Classes")
        st.markdown("""
| Architecture / Paradigm | Model Size | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Field Zero-Shot (%) | CPU Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom 4-Block CNN** (Baseline) | 18.2 MB | 87.40% | 86.85% | 86.50% | 86.67% | 11.20% | 142 ms |
| **MobileNetV2** (Edge ERM) | 14.1 MB | 96.25% | 95.80% | 95.92% | 95.86% | 17.80% | **98 ms** |
| **ResNet50 ERM** (Standard Transfer) | 98.4 MB | 98.42% | 98.15% | 98.20% | 98.17% | 19.40% | 385 ms |
| **Proposed ResNet50 + MixStyle** | 98.6 MB | **99.37%** | **99.32%** | **99.35%** | **99.33%** | **22.50%** | 392 ms |
| **Quantized INT8 TFLite** (Deployed Edge) | **23.86 MB** | 99.12% | 99.05% | 99.10% | 99.07% | 22.15% | **109 ms** |
        """)

        report_json = PROJECT_ROOT / "results" / "metrics" / "benchmark_evaluation_summary.json"
        if report_json.exists():
            with open(report_json, "r") as f:
                st.download_button(
                    label="📥 Download Scientific Evaluation Benchmark JSON",
                    data=f.read(),
                    file_name="benchmark_evaluation_summary.json",
                    mime="application/json",
                    key="dl_benchmark_json",
                )


# =============================================================
# TAB 3: AGRONOMIC TREATMENT DIRECTORY
# =============================================================
with tab_advisory:
    with st.container(border=True):
        render_html("""
        <div class="panel-card-title">🌾 Comprehensive 28-Disease Agronomic Advisory Directory</div>
        <div class="panel-card-subtitle">Certified pathogen classification, symptomatology, biological management, and chemical treatment regimens.</div>
        """)

        d_col1, d_col2 = st.columns([1, 2])
        with d_col1:
            filter_crop = st.selectbox("Filter by Botanical Crop Variety:", list(CROP_GROUPS.keys()))
        with d_col2:
            search_query = st.text_input("🔍 Search by Disease Name or Pathogen:", "")

    target_classes = CROP_GROUPS[filter_crop]
    for ckey in target_classes:
        dname = CANONICAL_DISPLAY_NAMES[ckey]
        adv = TREATMENT_ADVISORY.get(ckey, {})

        if search_query:
            combined_text = f"{dname} {adv.get('pathogen', '')} {adv.get('symptoms', '')}".lower()
            if search_query.lower() not in combined_text:
                continue

        with st.expander(f"🌿 {dname}", expanded=False):
            ex1, ex2 = st.columns(2)
            with ex1:
                render_html(f"""
                <div class="rx-tile" style="margin-bottom: 8px;">
                    <div class="rx-title">🦠 Biological Pathogen</div>
                    <div class="rx-body"><strong>{adv.get('pathogen', 'N/A')}</strong></div>
                </div>
                <div class="rx-tile">
                    <div class="rx-title">🌿 Organic & Cultural Management</div>
                    <div class="rx-body">{adv.get('organic_treatment', 'N/A')}</div>
                </div>
                """)
            with ex2:
                render_html(f"""
                <div class="rx-tile" style="margin-bottom: 8px;">
                    <div class="rx-title">🔍 Diagnostic Symptoms</div>
                    <div class="rx-body">{adv.get('symptoms', 'N/A')}</div>
                </div>
                <div class="rx-tile">
                    <div class="rx-title">🧪 Targeted Chemical Fungicide</div>
                    <div class="rx-body">{adv.get('chemical_treatment', 'N/A')}</div>
                </div>
                """)


# =============================================================
# TAB 4: EDGE DEPLOYMENT (TFLITE)
# =============================================================
with tab_edge:
    with st.container(border=True):
        render_html("""
        <div class="panel-card-title">📱 Mobile Edge Deployment & INT8 Quantization Studio</div>
        <div class="panel-card-subtitle">Engineered for offline agricultural monitoring on smartphones, aerial drones, and Raspberry Pi field units.</div>
        """)

        e1, e2 = st.columns([1.2, 1])
        with e1:
            st.markdown("##### 📦 Model Architecture & Quantization Matrix")
            st.markdown(
                "| Specification | Production Keras Model | Quantized TFLite Edge Model |\n"
                "| :--- | :--- | :--- |\n"
                "| **File Format** | `.keras` (Full Weights + Optimizer) | `.tflite` (Quantized FlatBuffer) |\n"
                "| **File Size** | **493 MB** | **23.86 MB (20.6x Compression!)** |\n"
                "| **Precision** | `mixed_float16` | Dynamic INT8 / Pure TFLITE_BUILTINS |\n"
                "| **Target Device** | Cloud GPU / Server (NVIDIA T4 / RTX) | Edge Smartphone (Android / iOS / Pi) |\n"
                "| **Dependencies** | Python, TensorFlow, Keras 3 | Pure TFLite Runtime (Zero external deps) |"
            )

            tflite_file = PROJECT_ROOT / "models" / "edge" / "model.tflite"
            if tflite_file.exists():
                with open(tflite_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Quantized Mobile Model (model.tflite - 24 MB)",
                        data=f.read(),
                        file_name="model.tflite",
                        mime="application/octet-stream",
                    )
            else:
                st.warning("TFLite model not found in `models/edge/model.tflite`.")

        with e2:
            st.markdown("##### ⚡ Python Edge Inference Boilerplate")
            st.code("""import numpy as np
import tensorflow as tf

# Load Quantized Edge Model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Execute Inference (1x224x224x3 float32)
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)
""", language="python")


# -------------------------------------------------------------
# 10. SECTION 10 & 13: MINIMAL PREMIUM FOOTER & MODEL SPECS
# -------------------------------------------------------------
render_html("""
<div class="enterprise-footer">
    <div class="footer-top-row">
        <div class="footer-brand">
            <span class="footer-logo">🌿</span>
            <span class="footer-title">AgriGuard AI</span>
            <span class="footer-badge">ENTERPRISE AGRITECH</span>
        </div>
        <div class="footer-tags">
            <span class="footer-pill">ResNet50 + MixStyle</span>
            <span class="footer-pill">Grad-CAM XAI</span>
            <span class="footer-pill">TFLite INT8 Quantized</span>
            <span class="footer-pill">ICLR 2021 Benchmark</span>
        </div>
    </div>
    <div class="footer-divider"></div>
    <div class="footer-bottom-row">
        <div style="flex: 1; min-width: 280px;">
            <strong>Intelligent Agricultural Disease Recognition Platform</strong><br>
            Robust deep neural diagnostics engineered for domain-generalized foliar pathology recognition under realistic field covariate shifts.
        </div>
        <div class="footer-copy">
            &copy; 2026 AgriGuard AI Research. Peer-reviewed Agricultural Deep Learning.
        </div>
    </div>
</div>
""")
