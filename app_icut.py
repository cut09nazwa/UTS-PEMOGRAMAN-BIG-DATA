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
# KONFIGURASI DASHBOARD
# ==========================
st.set_page_config(
    page_title="AI Flower Vision",
    page_icon="🌸",
    layout="wide",
)

# ==========================
# CSS STYLING TAMPAK PROFESIONAL
# ==========================
st.markdown("""
    <style>
        body {
            background-color: #f3fef6;
        }

        /* Navbar */
        .navbar {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            background-color: #ffffff;
            padding: 14px 40px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar a {
            margin-left: 25px;
            text-decoration: none;
            color: #222;
            font-weight: 500;
        }
        .navbar a:hover {
            color: #00a86b;
        }

        /* Hero Section */
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 80px 60px;
            background: linear-gradient(135deg, #ecfff4 0%, #ffffff 100%);
            border-radius: 20px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
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
        .cta-button {
            background-color: #00a86b;
            color: white;
            padding: 12px 24px;
            font-size: 18px;
            border: none;
            border-radius: 12px;
            margin-top: 30px;
            cursor: pointer;
            transition: 0.3s;
        }
        .cta-button:hover {
            background-color: #008e5b;
        }

        /* Gambar kanan */
        .hero-img img {
            width: 420px;
            border-radius: 20px;
            box-shadow: 0px 6px 12px rgba(0,0,0,0.1);
        }

        /* Card Info */
        .info-box {
            background-color: #f7fff9;
            border-left: 6px solid #00a86b;
            padding: 20px;
            border-radius: 12px;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# NAVBAR
# ==========================
st.markdown("""
    <div class="navbar">
        <a href="#fitur">Fitur</a>
        <a href="#carakerja">Cara Kerja</a>
    </div>
""", unsafe_allow_html=True)

# ==========================
# HERO SECTION (Teks kiri - Gambar kanan)
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

    if st.button("🌺 Mulai Petualangan AI", key="cta"):
        st.session_state['show_upload'] = True

with col2:
    image = Image.open("sample_images/1cc501a2ea_jpg.rf.dc455624ba691a864edbf790e48543dd.jpg")  # ganti dengan gambar kamu
    st.image(image, use_container_width=True, caption="AI mendeteksi bunga di gambar ini 🌷")

# ==========================
# BAGIAN INFORMASI TAMBAHAN
# ==========================
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("🌻 **Bunga Matahari terdeteksi (95% akurasi)**")
st.markdown("🔍 Deteksi Objek: 4 bunga ditemukan di gambar ini.")
st.markdown("""
AI Flower Vision mampu mengenali berbagai jenis bunga dengan tingkat akurasi tinggi.
Cukup unggah gambar dan sistem akan memprosesnya untuk mendeteksi dan mengklasifikasi bunga.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ==========================
# BAGIAN FITUR
# ==========================
st.markdown('<div id="fitur"></div>', unsafe_allow_html=True)
st.subheader("✨ Fitur Utama")
st.write("""
- **Klasifikasi Gambar** – Mengenali jenis bunga dari foto tunggal.
- **Deteksi Objek** – Menandai bunga-bunga yang muncul pada satu gambar.
- **Tampilan Modern** – Desain interaktif dan responsif.
""")

# ==========================
# BAGIAN CARA KERJA
# ==========================
st.markdown('<div id="carakerja"></div>', unsafe_allow_html=True)
st.subheader("⚙️ Cara Kerja Sistem")
st.write("""
1. Pengguna mengunggah gambar bunga melalui form upload.  
2. Sistem akan memilih mode: **Klasifikasi** atau **Deteksi Objek**.  
3. AI memproses gambar dan menampilkan hasil deteksi dengan tingkat akurasi.  
4. Pengguna dapat membaca deskripsi bunga yang teridentifikasi.  
""")
