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
# CSS STYLING DASHBOARD
# ==========================
st.markdown("""
<style>
/* ===== BODY & BACKGROUND ===== */
body {
    background: linear-gradient(135deg, #eafaf1 0%, #ffffff 100%);
    font-family: "Poppins", sans-serif;
}

/* Streamlit Container Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #f0fff4, #ffffff);
}

/* ===== NAVBAR ===== */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #ffffff;
    padding: 18px 60px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    border-bottom: 3px solid #00a86b10;
    position: sticky;
    top: 0;
    z-index: 100;
}

/* Logo Kiri */
.navbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
}

.navbar-logo {
    width: 42px;
    height: 42px;
    background-color: #00a86b;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
}

.navbar-title {
    font-family: 'Pacifico', cursive;
    font-size: 28px;
    color: #007b5e;
}

/* Menu Kanan */
.navbar-right a {
    margin-left: 28px;
    text-decoration: none;
    color: #2d2d2d;
    font-weight: 500;
    transition: 0.3s;
}

.navbar-right a:hover {
    color: #00a86b;
}

/* Tombol Mulai Sekarang */
.navbar-button {
    background-color: #00a86b;
    color: white;
    padding: 10px 22px;
    font-size: 16px;
    border-radius: 12px;
    margin-left: 30px;
    box-shadow: 0 3px 10px rgba(0,168,107,0.25);
    transition: 0.3s ease;
    text-decoration: none;
}

.navbar-button:hover {
    background-color: #008e5b;
    transform: translateY(-2px);
}

/* ===== HERO SECTION ===== */
.hero {
    text-align: center;
    padding: 120px 40px 100px;
    background: linear-gradient(135deg, #ecfff4 0%, #ffffff 100%);
    border-radius: 25px;
    box-shadow: 0 6px 15px rgba(0,0,0,0.05);
    margin: 40px auto;
    max-width: 1100px;
}

/* Judul Hero */
.hero h1 {
    font-size: 44px;
    font-weight: 800;
    color: #007b5e;
    margin-bottom: 25px;
}

/* Background kapsul untuk teks kecil */
.hero .highlight {
    display: inline-block;
    background: linear-gradient(90deg, #b3f2d1, #d4ffea);
    color: #007b5e;
    padding: 10px 25px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 18px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# NAVBAR
# ==========================
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <div class="navbar-logo">🌷</div>
        <div class="navbar-title">AI Flower Vision</div>
    </div>
    <div class="navbar-right">
        <a href="#fitur">Fitur</a>
        <a href="#carakerja">Cara Kerja</a>
        <a href="#tentang">Tentang</a>
        <a href="#mulai" class="navbar-button">🌱 Mulai Sekarang</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION
# ==========================
st.markdown("""
<div class="hero">
    <div class="highlight">Teknologi AI Terdepan untuk Klasifikasi Bunga</div>
    <h1>Kenali Keindahan Bunga dengan <br> Sentuhan Kecerdasan Buatan</h1>
</div>
""", unsafe_allow_html=True)
# ==========================
# HERO SECTION
# ==========================
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("<h3 style='color:#00a86b;'>Teknologi AI untuk Klasifikasi Bunga</h3>", unsafe_allow_html=True)
    st.markdown("<h1>Kenali <span style='color:#00a86b;'>Setiap Bunga</span> dengan AI</h1>", unsafe_allow_html=True)
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

    st.markdown("""
        <button class="cta-button">🌺 Mulai Petualangan AI</button>
    """, unsafe_allow_html=True)

with col2:
    image = Image.open("sample_images/1cc501a2ea_jpg.rf.dc455624ba691a864edbf790e48543dd.jpg")  # ganti sesuai path gambar kamu
    st.image(image, use_container_width=True, caption="AI mendeteksi bunga di gambar ini 🌷")

# =======================
# BAGIAN FITUR UNGGULAN
# =======================

st.markdown(
    """
    <style>
    /* ======== WRAPPER UTAMA ======== */
    .fitur-wrapper {
        width: 100%;
        max-width: 1100px;
        margin: 0 auto;              /* agar konten di tengah halaman */
        padding: 60px 40px 80px;
        text-align: center;          /* pastikan teks di tengah */
        display: flex;
        flex-direction: column;
        align-items: center;         /* center horizontal */
        justify-content: center;     /* center vertical dalam container */
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
        align-items: flex-start;
        flex-wrap: wrap;
        gap: 50px;
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

# Tata letak 2 kolom — kode Python berada di luar string
col1, col2 = st.columns(2)

# ===== Kolom 1 =====
with col1:
    st.markdown(
        """
        <div style='background-color:#E9FBF0; border-radius:15px; padding:25px; text-align:center;
                    box-shadow:0 4px 10px rgba(0,0,0,0.05); transition:0.3s;'>
            <div style='font-size:40px; color:#FFF8E7;'>🌼</div>
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
        <div style='background-color:#EAF3FF; border-radius:15px; padding:25px; text-align:center;
                    box-shadow:0 4px 10px rgba(0,0,0,0.05); transition:0.3s;'>
            <div style='font-size:40px; color:#2563eb;'>🔍</div>
            <h4 style='color:#0f172a; margin-bottom:8px;'>Deteksi Bagian Bunga</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5;'>
                AI dapat mengenali bagian-bagian bunga seperti kelopak, putik, dan daun.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

