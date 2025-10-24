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
# CSS STYLING TAMPILAN PROFESIONAL + BACKGROUND WARNA
# ==========================
st.markdown("""
    <style>
        /* ===== BODY & BACKGROUND ===== */
        body {
            background: linear-gradient(135deg, #eafaf1 0%, #ffffff 100%);
            font-family: "Poppins", sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: linear-gradient(120deg, #f1fff7, #ffffff);
        }

        /* ===== NAVBAR ===== */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #ffffff;
            padding: 16px 80px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            border-bottom: 2px solid #00a86b20;
            border-radius: 0;
            margin: 0;
            position: sticky;
            top: 0;
            z-index: 100;
            margin-bottom: 30px;
        }

        .navbar a {
            margin-left: 28px;
            text-decoration: none;
            color: #222;
            font-weight: 500;
            transition: 0.3s;
            font-size: 17px;
        }

        .navbar a:hover {
            color: #00a86b;
        }

        .navbar-left {
            flex: 1;
        }

        .navbar-title {
            font-family: 'Pacifico', cursive;
            font-size: 30px;
            font-weight: 600;
            color: #009970;
            letter-spacing: 0.5px;
        }

        .navbar-button {
            background-color: #00a86b;
            color: white !important;
            padding: 10px 22px;
            border-radius: 14px;
            font-weight: 600;
            margin-left: 25px;
            box-shadow: 0px 3px 8px rgba(0, 153, 112, 0.25);
            transition: 0.3s ease;
        }

        .navbar-button:hover {
            background-color: #007e5d;
            transform: translateY(-2px);
        }

        /* ===== HERO SECTION ===== */
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 100px 80px;
            margin: 30px 40px;
            background: linear-gradient(145deg, #e9fff5 0%, #ffffff 100%);
            border-radius: 24px;
            box-shadow: 0px 6px 14px rgba(0,0,0,0.05);
        }

        .highlight {
            background-color: #c6f5e3;
            color: #006b47;
            display: inline-block;
            padding: 8px 22px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 17px;
            margin-bottom: 25px;
        }

        .hero-text h1 {
            font-size: 60px;
            font-weight: 800;
            color: #1a1a1a;
            line-height: 1.2;
            margin-bottom: 20px;
        }

        .hero-text span {
            color: #00a86b;
        }

        .hero-text p {
            font-size: 19px;
            color: #444;
            margin-top: 10px;
            line-height: 1.7;
            max-width: 600px;
        }

        .stats {
            display: flex;
            gap: 60px;
            margin-top: 25px;
        }

        .stat-box {
            text-align: left;
        }

        .stat-value {
            font-size: 30px;
            font-weight: bold;
            color: #00a86b;
        }

        .cta-button {
            background-color: #00a86b;
            color: white;
            padding: 14px 32px;
            font-size: 18px;
            border: none;
            border-radius: 14px;
            margin-top: 40px;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0, 168, 107, 0.3);
            transition: 0.3s ease;
        }

        .cta-button:hover {
            background-color: #008e5b;
            transform: translateY(-2px);
        }

        .hero-img img {
            width: 440px;
            border-radius: 20px;
            box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
        }

    </style>

    <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


# ==========================
# NAVBAR
# ==========================
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <div class="navbar-title">🌸 AI Flower Vision</div>
    </div>
    <div class="navbar-right">
        <a href="#fitur">Fitur</a>
        <a href="#carakerja">Cara Kerja</a>
        <a href="#mulai" class="navbar-button">🌱 Mulai Sekarang</a>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================
# HERO SECTION
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<div class='highlight'>Teknologi AI Terdepan untuk Klasifikasi Bunga</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-text'><h1>Kenali <span>Setiap Bunga</span><br>dengan AI</h1></div>", unsafe_allow_html=True)
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
    
# =======================
# BAGIAN FITUR UNGGULAN
# =======================

# === CSS Styling ===
st.markdown(
    """
    <style>
    /* ======== WRAPPER UTAMA ======== */
    .fitur-wrapper {
        width: 100%;
        padding: 80px 60px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%);
        margin: 0;
        border-radius: 0;
    }

    /* ======== JUDUL & SUBJUDUL ======== */
    .section-title {
        color: #0f172a;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .section-subtitle {
        color: #334155;
        font-size: 18px;
        max-width: 700px;
        margin: 0 auto 60px;
        line-height: 1.6;
    }

    /* ======== KONTENER FITUR ======== */
    .features-container {
        display: flex;
        justify-content: center;
        align-items: stretch;
        flex-wrap: wrap;
        gap: 40px;
        width: 100%;
        max-width: 1100px;
    }

    /* ======== CARD FITUR ======== */
    .feature-card {
        flex: 1 1 45%;
        max-width: 480px;
        padding: 35px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        background-color: #ffffff;
        text-align: center;
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 30px rgba(0,0,0,0.15);
    }

    .feature-icon {
        font-size: 40px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #0f172a;
    }

    .feature-text {
        font-size: 15px;
        color: #334155;
        line-height: 1.6;
    }

    /* ======== WRAPPER & SECTION ======== */
    .fitur-wrapper {
        width: 100%;
        padding: 80px 60px;
        text-align: center;
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%);
        border-radius: 0;
        margin: 0;
    }

    .section-title {
        color: #0f172a;
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .section-subtitle {
        color: #334155;
        font-size: 18px;
        max-width: 700px;
        margin: 0 auto 60px;
        line-height: 1.6;
    }
    </style>

    <div class="fitur-wrapper">
        <h1 class="section-title">Fitur Unggulan</h1>
        <p class="section-subtitle">
            Teknologi AI terdepan yang memungkinkan Anda mengeksplorasi dunia bunga 
            dengan cara yang belum pernah ada sebelumnya.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =======================
# TATA LETAK FITUR (2 KOLOM)
# =======================

col1, col2 = st.columns(2)

# ===== Kolom 1 =====
with col1:
    st.markdown(
        """
        <div style='background-color:#E9FBF0; border-radius:15px; padding:35px; 
                    text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.05); 
                    transition:0.3s; min-height:240px; display:flex; flex-direction:column; 
                    justify-content:center;'>
            <div style='font-size:40px;'>🌼</div>
            <h4 style='color:#0f172a; margin-bottom:8px;'>Kenali Jenis Bunga</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5;'>
                Upload foto bunga dan AI akan memberitahu jenis dan nama bunga tersebut.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== Kolom 2 =====
with col2:
    st.markdown(
        """
        <div style='background-color:#EAF3FF; border-radius:15px; padding:35px; 
                    text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.05); 
                    transition:0.3s; min-height:240px; display:flex; flex-direction:column; 
                    justify-content:center;'>
            <div style='font-size:40px;'>🔍</div>
            <h4 style='color:#0f172a; margin-bottom:8px;'>Deteksi Bagian Bunga</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5;'>
                AI dapat mengenali bagian-bagian bunga seperti kelopak, putik, dan daun.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
