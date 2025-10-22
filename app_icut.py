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
            background: linear-gradient(135deg, #dfffe9 0%, #ffffff 100%);
        }

        /* ===== NAVBAR ===== */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #ffffff;
            padding: 18px 50px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
            border-radius: 0 0 16px 16px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar-left h2 {
            margin: 0;
            font-size: 22px;
            font-family: 'Pacifico', cursive;
            color: #009f6b;
            letter-spacing: 0.5px;
        }
        .navbar-right a {
            margin-left: 28px;
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
            color: white !important;
            padding: 10px 22px;
            border-radius: 10px;
            font-weight: 600;
            text-decoration: none;
            transition: 0.3s;
            margin-left: 25px;
        }
        .btn-nav:hover {
            background-color: #009660;
            transform: translateY(-1px);
        }

        /* ===== HERO SECTION ===== */
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 90px 70px 70px 70px;
            background: linear-gradient(135deg, #f5fff9 0%, #ffffff 100%);
            border-radius: 25px;
            box-shadow: 0px 6px 15px rgba(0,0,0,0.05);
            margin-top: 40px;
        }
        .label-small {
            display: inline-block;
            background-color: #c9f7df;
            color: #007a4a;
            font-size: 15px;
            font-weight: 500;
            padding: 8px 16px;
            border-radius: 25px;
            margin-bottom: 14px;
        }
        .hero-text {
            max-width: 55%;
        }
        .hero-text h1 {
            font-size: 58px;
            font-weight: 800;
            line-height: 1.2;
            color: #1a1a1a;
        }
        .hero-text h1 span:nth-child(1) { color: #00a86b; }
        .hero-text h1 span:nth-child(2) { color: #00855a; }
        .hero-text h1 span:nth-child(3) { color: #006644; }

        .hero-text p {
            font-size: 18px;
            color: #444;
            margin-top: 18px;
            line-height: 1.6;
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
            font-size: 32px;
            font-weight: bold;
            color: #00a86b;
        }

        /* ===== CTA BUTTON ===== */
        .cta-button {
            background-color: #00a86b;
            color: white;
            padding: 14px 30px;
            font-size: 19px;
            border: none;
            border-radius: 14px;
            margin-top: 40px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 168, 107, 0.3);
            transition: 0.3s ease;
        }
        .cta-button:hover {
            background-color: #008e5b;
            transform: translateY(-3px);
        }

        /* ===== GAMBAR KANAN ===== */
        .hero-img img {
            width: 460px;
            border-radius: 20px;
            box-shadow: 0px 8px 14px rgba(0,0,0,0.1);
        }
    </style>

    <link href="https://fonts.googleapis.com/css2?family=Pacifico&family=Poppins:wght@400;600;700;800&display=swap" rel="stylesheet">
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

import streamlit as st

import streamlit as st

import streamlit as st

# =======================
# BAGIAN FITUR UNGGULAN
# =======================

st.markdown("""
<style>
/* Judul utama */
.section-title {
    text-align: center;
    color: #0f172a;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 10px;
}

.section-subtitle {
    text-align: center;
    color: #334155;
    font-size: 18px;
    max-width: 700px;
    margin: 0 auto 50px;
}

/* Kontainer fitur */
.features-container {
    display: flex;
    justify-content: center;
    gap: 60px;
    flex-wrap: wrap;
}

/* Setiap kartu fitur */
.feature-card {
    width: 400px;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    transition: all 0.3s ease;
}

/* Efek hover */
.feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 16px 30px rgba(0,0,0,0.25);
}

/* Judul dan teks di dalam kartu */
.feature-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}
.feature-text {
    font-size: 16px;
    line-height: 1.5;
}
</style>

<h1 class='section-title'>Fitur Unggulan</h1>
<p class='section-subtitle'>
    Teknologi AI terdepan yang memungkinkan Anda mengeksplorasi dunia bunga
    dengan cara yang belum pernah ada sebelumnya.
</p>

<div class='features-container'>
# Tata letak 2 kolom
col1, col2 = st.columns(2)

# ===== Kolom 1 =====
with col1:
    st.markdown(
        """
        <div style='background-color:#E9FBF0; border-radius:15px; padding:25px; text-align:center;
                    box-shadow:0 4px 10px rgba(0,0,0,0.05); transition:0.3s;'>
            <div style='font-size:40px; color:#00A86B;'>🌼</div>
            <h4 style='color:#0f172a; margin-bottom:8px;'>Kenali Jenis Bunga</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5;'>
                Upload foto bunga dan AI akan memberitahu jenis dan nama bunga tersebut.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Kolom 2 =====
with col2:
    st.markdown(
        """
        <div style='background-color:#EAF3FF; border-radius:15px; padding:25px; text-align:center;
                    box-shadow:0 4px 10px rgba(0,0,0,0.05); transition:0.3s;'>
            <div style='font-size:40px; color:#2563eb;'>🔍</div>
            <h4 style='color:#0f172a; margin-bottom:8px;'>Deteksi Bagian Bunga</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5;'>
                AI dapat mengenali bagian-bagian bunga seperti kelopak, putik, dan daun.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
