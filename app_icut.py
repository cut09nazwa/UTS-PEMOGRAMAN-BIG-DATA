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
# CSS STYLING (MIRIP CONTOH GAMBAR)
# ==========================
st.markdown("""
    <style>
        /* FONT */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        body {
            background-color: #f6fffb;
            font-family: 'Inter', sans-serif;
        }
        [data-testid="stAppViewContainer"] {
            background: #f6fffb;
        }

        /* HERO SECTION */
        .hero {
            padding: 80px 80px 60px 80px;
            background: linear-gradient(180deg, #f6fffb 0%, #ffffff 100%);
            border-radius: 20px;
        }

        /* LABEL ATAS */
        .label-small {
            display: inline-block;
            background-color: #c9f7df;
            color: #007a4a;
            font-size: 15px;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 25px;
            margin-bottom: 22px;
        }

        /* JUDUL UTAMA */
        .hero h1 {
            font-size: 56px;
            font-weight: 800;
            line-height: 1.2;
            color: #0f172a; /* hitam elegan */
            margin-bottom: 20px;
        }
        .hero h1 span {
            color: #009f6b; /* hijau toska */
        }

        /* PARAGRAF */
        .hero p {
            font-size: 18px;
            color: #334155;
            line-height: 1.6;
            margin-bottom: 45px;
        }

        /* STATISTIK */
        .stats {
            display: flex;
            gap: 60px;
            margin-bottom: 45px;
        }
        .stat-box {
            text-align: left;
        }
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            color: #009f6b;
        }
        .stat-label {
            color: #334155;
            font-size: 15px;
        }

        /* TOMBOL */
        .btn-primary {
            background-color: #009f6b;
            color: white !important;
            padding: 12px 26px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 600;
            text-decoration: none;
            margin-right: 14px;
            transition: 0.3s;
        }
        .btn-primary:hover {
            background-color: #008658;
        }
        .btn-outline {
            border: 2px solid #009f6b;
            color: #009f6b !important;
            background-color: transparent;
            padding: 12px 26px;
            border-radius: 12px;
            font-size: 17px;
            font-weight: 600;
            text-decoration: none;
            transition: 0.3s;
        }
        .btn-outline:hover {
            background-color: #e6fff2;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# NAVBAR
# ==========================
st.markdown("""
    <div class="navbar">
        <div class="navbar-left">
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
    st.markdown("<h1><span>Kenali</span> <span>Setiap</span> <span>Bunga</span> dengan AI</h1>", unsafe_allow_html=True)
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
