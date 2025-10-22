import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/best.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Cut Nazwa Humaira_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# UI
# ==========================
# KONFIGURASI DASBOR
# ==========================
st.set_page_config(
    page_title="AI Flower Vision",
    page_icon="🌸",
    layout="wide",
)

# ==========================
# CSS STYLING PROFESIONAL + NAVBAR + BACKGROUND
# ==========================
st.markdown("""
    <style>
        /* ===== BODY & BACKGROUND ===== */
        body {
            background: linear-gradient(135deg, #eafaf1 0%, #ffffff 100%);
            font-family: "Poppins", sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #e6fff1 0%, #ffffff 100%);
        }

        /* ===== NAVBAR ===== */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #ffffff;
            padding: 14px 40px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .navbar-left img {
            width: 38px;
            height: 38px;
        }
        .navbar-left h2 {
            margin: 0;
            font-size: 20px;
            color: #00a86b;
            font-weight: 700;
        }
        .navbar-right a {
            margin-left: 25px;
            text-decoration: none;
            color: #222;
            font-weight: 500;
            transition: 0.3s;
        }
        .navbar-right a:hover {
            color: #00a86b;
        }
        .btn-nav {
            background-color: #00a86b;
            color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-weight: 600;
            text-decoration: none;
            transition: 0.3s;
        }
        .btn-nav:hover {
            background-color: #009660;
        }

        /* ===== HERO SECTION ===== */
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 80px 60px;
            background: linear-gradient(135deg, #ecfff4 0%, #ffffff 100%);
            border-radius: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
        .label-small {
            display: inline-block;
            background-color: #c9f7df;
            color: #007a4a;
            font-size: 14px;
            font-weight: 500;
            padding: 6px 14px;
            border-radius: 20px;
            margin-bottom: 10px;
        }
        .hero-text {
            max-width: 55%;
        }
        .hero-text h1 {
            font-size: 48px;
            font-weight: 800;
            color: #1a1a1a;
        }
        .hero-text span {
            color: #00a86b;
        }
        .hero-text p {
            font-size: 18px;
            color: #444;
            margin-top: 10px;
            line-height: 1.6;
        }
        .stats {
            display: flex;
            gap: 50px;
            margin-top: 20px;
        }
        .stat-box {
            text-align: left;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #00a86b;
        }

        /* ===== CTA BUTTON ===== */
        .cta-button {
            background-color: #00a86b;
            color: white;
            padding: 12px 28px;
            font-size: 18px;
            border: none;
            border-radius: 12px;
            margin-top: 35px;
            cursor: pointer;
            box-shadow: 0 3px 10px rgba(0, 168, 107, 0.3);
            transition: 0.3s ease;
        }
        .cta-button:hover {
            background-color: #008e5b;
            transform: translateY(-2px);
        }

        /* ===== GAMBAR KANAN ===== */
        .hero-img img {
            width: 420px;
            border-radius: 20px;
            box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# NAVBAR
# ==========================
st.markdown("""
    <div class="navbar">
        <div class="navbar-left">
            <img src="https://cdn-icons-png.flaticon.com/512/765/765564.png" alt="Logo">
            <h2>AI Flower Vision</h2>
        </div>
        <div class="navbar-right">
            <a href="#fitur">Fitur</a>
            <a href="#carakerja">Cara Kerja</a>
            <a class="btn-nav" href="#mulai">Mulai Sekarang</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='label-small'>Teknologi AI Terdepan untuk Klasifikasi Bunga</div>", unsafe_allow_html=True)
    st.markdown("<h1>Kenali <span>Setiap Bunga</span> dengan AI</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p>
        Platform revolusioner yang menggunakan kecerdasan buatan untuk mengidentifikasi spesies bunga, 
        mendeteksi objek, dan memberikan informasi detail tentang setiap bunga yang Anda temukan.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">500+</div>
                <div>Spesies Bunga</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">98%</div>
                <div>Akurasi AI</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">1200+</div>
                <div>Pengguna Aktif</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<button class='cta-button'>🌺 Mulai Petualangan AI</button>", unsafe_allow_html=True)

with col2:
    image = Image.open("sample_images/1cc501a2ea_jpg.rf.dc455624ba691a864edbf790e48543dd.jpg")
    st.image(image, use_container_width=True, caption="AI mendeteksi bunga di gambar ini 🌷")
