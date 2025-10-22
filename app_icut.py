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
# CONFIGURASI DASHBOARD
# ==========================
st.set_page_config(
    page_title="AI Flower Vision",
    page_icon="🌸",
    layout="wide",
)

# ==========================
# CSS STYLING
# ==========================
st.markdown("""
    <style>
        .navbar {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            background-color: #ffffff;
            padding: 12px 30px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .navbar a {
            margin-left: 25px;
            text-decoration: none;
            color: #333;
            font-weight: 500;
        }
        .navbar a:hover {
            color: #00a86b;
        }
        .section {
            padding-top: 80px;
        }
    </style>

    <div class="navbar">
        <a href="#fitur">Fitur</a>
        <a href="#carakerja">Cara Kerja</a>
    </div>
""", unsafe_allow_html=True)

# ==========================
# HALAMAN PEMBUKA
# ==========================
st.markdown("### 🌿 Teknologi AI Terdepan untuk Klasifikasi Bunga")
st.markdown("<h1>Kenali <span style='color:#00a86b'>Setiap Bunga</span> dengan AI</h1>", unsafe_allow_html=True)

st.write("""
Platform revolusioner yang menggunakan kecerdasan buatan untuk mengidentifikasi spesies bunga, 
mendeteksi objek, dan memberikan informasi detail tentang setiap bunga yang Anda temukan.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌷 Spesies Bunga", "500+")
with col2:
    st.metric("🤖 Akurasi AI", "98%")
with col3:
    st.metric("👩‍💻 Pengguna Aktif", "1200+")

# Tombol aksi
st.write("")
st.button("🌺 Mulai Petualangan AI")

# ==========================
# GAMBAR CONTOH
# ==========================
st.write("---")
col_img, col_info = st.columns([2, 1])

with col_img:
    # Ganti dengan path gambar kamu di folder project
    image = Image.open("sample_images/flowers.jpg")
    st.image(image, use_container_width=True, caption="Contoh Gambar Bunga")

with col_info:
    st.success("🌹 Mawar Merah terdeteksi (95% akurasi)")
    st.info("🔍 Deteksi Objek: 4 bunga ditemukan di gambar ini.")
    st.write("""
    AI Flower Vision dapat mengenali berbagai jenis bunga dengan tingkat akurasi tinggi.  
    Coba unggah foto bunga favoritmu dan lihat hasil klasifikasinya!
    """)
