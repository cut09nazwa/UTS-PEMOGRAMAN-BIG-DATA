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

# ==========================
# HALAMAN 1: HOME / HALAMAN AWAL
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

st.markdown("""
<a href='#mulai' class='cta-button'
   style='display:inline-block; background-color:#00a86b; color:#ffffff !important;
          padding:12px 28px; border-radius:30px; font-weight:600; text-decoration:none;
          box-shadow:0 6px 16px rgba(0,168,107,0.28); transition:transform .18s ease, box-shadow .18s ease;
          font-size:16px;'>
   🌺 Mulai Petualangan AI
</a>
""", unsafe_allow_html=True)

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
    /* ======== SET BACKGROUND DASHBOARD ======== */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%) !important;
    }

    /* ======== WRAPPER UTAMA ======== */
    .fitur-wrapper {
        width: 100%;
        padding: 60px 50px 40px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%);
        margin: 0 ;
        border-radius: 0;
    }

    /* ======== JUDUL & SUBJUDUL ======== */
    .section-title {
        color: #0f172a;
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .section-subtitle {
        color: #334155;
        font-size: 18px;
        max-width: 720px;
        margin: 0 auto 45px;
        line-height: 1.6;
    }

    /* ======== KONTENER FITUR ======== */
    .features-container {
        display: flex;
        justify-content: center;
        align-items: stretch;
        flex-wrap: wrap;
        gap: 32px;
        width: 100%;
        max-width: 1100px;
        background-color: #ffffff; /* untuk jaga jarak antar card */
        padding: 20px 0;
        border-radius: 20px;
    }

    /* ======== CARD FITUR ======== */
    .feature-card {
        flex: 1 1 45%;
        max-width: 480px;
        padding: 30px 28px;
        border-radius: 20px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        transition: all 0.3s ease;
        background-color: #ffffff;
        text-align: center;
        min-height: 230px; 
    }

    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 14px 28px rgba(0,0,0,0.15);
    }

    .feature-icon {
        font-size: 40px;
        margin-bottom: 14px;
    }

    .feature-title {
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #0f172a;
    }

    .feature-text {
        font-size: 15px;
        color: #334155;
        line-height: 1.6;
        margin: 0 auto;
        max-width: 90%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =======================
# KONTEN FITUR UNGGULAN
# =======================
st.markdown('<div id="mulai"></div>', unsafe_allow_html=True)

st.markdown(
    """
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
                    transition:0.3s; min-height:230px; display:flex; flex-direction:column; 
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
                    transition:0.3s; min-height:230px; display:flex; flex-direction:column; 
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


# ==========================
# BAGIAN PROSES DALAM 4 LANGKAH
# ==========================

# ===== CSS STYLING =====
st.markdown("""
    <style>
    /* ===== WRAPPER SECTION ===== */
    .proses-wrapper {
        width: 100%;
        padding: 50px 40px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: linear-gradient(145deg, #f8fffa 0%, #eaf7ff 100%);
    }

    /* ===== JUDUL & SUBJUDUL ===== */
    .proses-title {
        color: #0f172a;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 6x;
    }

    .proses-subtitle {
        color: #334155;
        font-size: 17px;
        max-width: 700px;
        margin: 0 auto 35px;
        line-height: 1.6;
    }

    /* ===== KONTENER CARD ===== */
    .steps-container {
        display: flex;
        justify-content: center;
        align-items: stretch;
        flex-wrap: wrap;
        gap: 30px;
        max-width: 1100px;
        margin: 0 auto 50px;
    }

    /* ===== CARD LANGKAH ===== */
    .step-card {
        flex: 1 1 22%;
        background: #ffffff;
        border-radius: 20px;
        padding: 30px 25px;
        box-shadow: 0 8px 18px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
        text-align: center;
        min-width: 240px;
    }

    .step-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
    }

    /* ===== NOMOR LINGKARAN ===== */
    .step-number {
        display: inline-block;
        width: 36px;
        height: 36px;
        line-height: 36px;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        margin-bottom: 12px;
    }

    .blue { background: linear-gradient(135deg, #0ea5e9, #0284c7); }
    .purple { background: linear-gradient(135deg, #a855f7, #7e22ce); }
    .green { background: linear-gradient(135deg, #10b981, #059669); }
    .orange { background: linear-gradient(135deg, #f97316, #ea580c); }

    /* ===== TEKS DALAM CARD ===== */
    .step-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .step-desc {
        font-size: 15px;
        color: #475569;
        line-height: 1.5;
    }

    /* ===== TOMBOL ===== */
    .start-button {
        display: inline-block;
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 30px;
        text-decoration: none;
        box-shadow: 0 6px 14px rgba(16,185,129,0.3);
        transition: all 0.3s ease;
        margin-top: 10px;
    }

    .start-button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 18px rgba(16,185,129,0.45);
    }
    </style>
""", unsafe_allow_html=True)


# ===== HTML STRUCTURE =====
st.markdown("""
<div class="proses-wrapper">
    <h1 class="proses-title">Proses Mudah dalam 4 Langkah</h1>
    <p class="proses-subtitle">
        Perjalanan interaktif yang akan membawa Anda memahami teknologi AI 
        dan menganalisis bunga favorit Anda
    </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======= STYLE GLOBAL =======
card_style = """
    background-color: white; border-radius: 15px; padding: 20px 30px; 
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05); margin-bottom: 20px;
"""

# ======= 4 LANGKAH =======
st.markdown(f"""
<div style='{card_style}'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <div style='background-color:#2563EB; color:white; font-weight:bold; 
                    border-radius:50%; width:35px; height:35px; display:flex; 
                    align-items:center; justify-content:center;'>1</div>
        <div>
            <h4 style='margin:0; color:#0f172a;'>Selamat Datang</h4>
            <p style='margin:5px 0 0; color:#334155; font-size:15px;'>
                Mulai perjalanan Anda dengan sambutan yang menarik dan pengenalan fitur AI Flower Vision.
            </p>
        </div>
    </div>
    <div style='font-size:25px; color:#2563EB;'>💐</div>
</div>

<div style='{card_style}'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <div style='background-color:#9333EA; color:white; font-weight:bold; 
                    border-radius:50%; width:35px; height:35px; display:flex; 
                    align-items:center; justify-content:center;'>2</div>
        <div>
            <h4 style='margin:0; color:#0f172a;'>Pelajari Teknologi AI</h4>
            <p style='margin:5px 0 0; color:#334155; font-size:15px;'>
                Pahami bagaimana kecerdasan buatan bekerja untuk mengenali bentuk, warna, dan pola pada bunga.
            </p>
        </div>
    </div>
    <div style='font-size:25px; color:#9333EA;'>🤖</div>
</div>

<div style='{card_style}'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <div style='background-color:#16A34A; color:white; font-weight:bold; 
                    border-radius:50%; width:35px; height:35px; display:flex; 
                    align-items:center; justify-content:center;'>3</div>
        <div>
            <h4 style='margin:0; color:#0f172a;'>Atur Preferensi</h4>
            <p style='margin:5px 0 0; color:#334155; font-size:15px;'>
                Masukkan nama Anda, pilih tema tampilan, dan tentukan tujuan penggunaan untuk pengalaman yang personal.
            </p>
        </div>
    </div>
    <div style='font-size:25px; color:#16A34A;'>🧍‍♀️</div>
</div>

<div style='{card_style}'>
    <div style='display: flex; align-items: center; gap: 20px;'>
        <div style='background-color:#EA580C; color:white; font-weight:bold; 
                    border-radius:50%; width:35px; height:35px; display:flex; 
                    align-items:center; justify-content:center;'>4</div>
        <div>
            <h4 style='margin:0; color:#0f172a;'>Analisis Gambar</h4>
            <p style='margin:5px 0 0; color:#334155; font-size:15px;'>
                Upload foto bunga Anda dan pilih mode analisis untuk mendapatkan hasil klasifikasi atau deteksi objek.
            </p>
        </div>
    </div>
    <div style='font-size:25px; color:#EA580C;'>📸</div>
</div>
""", unsafe_allow_html=True)

# ======= TOMBOL =======
st.markdown("""
<div style='text-align:center; margin-top:30px;'>
    <a href='#ai-tech' 
       style='display:inline-block; background-color:#00a86b; color:#ffffff !important;
              padding:12px 35px; border-radius:30px; font-weight:600; text-decoration:none;
              box-shadow:0 4px 10px rgba(0,168,107,0.3); transition:transform .18s ease, box-shadow .18s ease;
              font-size:16px;'>
       👇🌼 <span style='color:#ffffff;'>Mulai Eksplorasi</span>
    </a>
</div>

<style>
html { scroll-behavior: smooth; }
a.cta-button, a[style*="background-color:#00a86b"] {
    text-decoration: none !important;
    color: #ffffff !important;
}
a.cta-button:hover, a[style*="background-color:#00a86b"]:hover {
    transform: translateY(-3px);
    box-shadow:0 8px 18px rgba(0,168,107,0.28);
}
</style>
""", unsafe_allow_html=True)


# =======================
# TEKNOLOGI AI YANG MENAKJUBKAN
# =======================
st.markdown('<div id="ai-tech"></div>', unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center; padding:70px 0 40px;'>
        <h1 style='color:#0f172a; font-size:42px; font-weight:800; margin-bottom:10px;'>
            Teknologi AI yang Menakjubkan
        </h1>
        <p style='color:#334155; font-size:17px; max-width:700px; margin:0 auto; line-height:1.6;'>
            Pelajari bagaimana kecerdasan buatan dapat mengenali bentuk, warna, dan pola pada bunga.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===== 3 KOLOM (Computer Vision, Neural Networks, Machine Learning) =====
col1, col2, col3 = st.columns(3)

# ===== Kolom 1 =====
with col1:
    st.markdown(
        """
        <div style='background-color:white; border-radius:20px; padding:35px; 
                    text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.05); 
                    transition:0.3s; min-height:300px; display:flex; flex-direction:column; 
                    justify-content:center;'>
            <div style='font-size:45px; background:linear-gradient(135deg,#9333ea,#ec4899);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>📷</div>
            <h4 style='color:#0f172a; margin-bottom:10px;'>Computer Vision</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5; margin-bottom:12px;'>
                AI menganalisis setiap pixel gambar untuk mengenali bentuk, tekstur, dan warna kelopak bunga.
            </p>
            <ul style='list-style:none; padding:0; text-align:left; color:#475569; font-size:14px;'>
                <li>• Deteksi tepi dan kontur</li>
                <li>• Analisis warna RGB</li>
                <li>• Pengenalan pola tekstur</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== Kolom 2 =====
with col2:
    st.markdown(
        """
        <div style='background-color:white; border-radius:20px; padding:35px; 
                    text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.05); 
                    transition:0.3s; min-height:300px; display:flex; flex-direction:column; 
                    justify-content:center;'>
            <div style='font-size:45px; background:linear-gradient(135deg,#9333ea,#ec4899);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🧠</div>
            <h4 style='color:#0f172a; margin-bottom:10px;'>Neural Networks</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5; margin-bottom:12px;'>
                Jaringan saraf tiruan yang meniru cara kerja otak manusia dalam memproses informasi visual.
            </p>
            <ul style='list-style:none; padding:0; text-align:left; color:#475569; font-size:14px;'>
                <li>• Deep learning layers</li>
                <li>• Pattern recognition</li>
                <li>• Feature extraction</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== Kolom 3 =====
with col3:
    st.markdown(
        """
        <div style='background-color:white; border-radius:20px; padding:35px; 
                    text-align:center; box-shadow:0 8px 25px rgba(0,0,0,0.05); 
                    transition:0.3s; min-height:300px; display:flex; flex-direction:column; 
                    justify-content:center;'>
            <div style='font-size:45px; background:linear-gradient(135deg,#9333ea,#ec4899);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>🤖</div>
            <h4 style='color:#0f172a; margin-bottom:10px;'>Machine Learning</h4>
            <p style='color:#334155; font-size:15px; line-height:1.5; margin-bottom:12px;'>
                Algoritma yang belajar dari ribuan gambar bunga untuk meningkatkan akurasi klasifikasi.
            </p>
            <ul style='list-style:none; padding:0; text-align:left; color:#475569; font-size:14px;'>
                <li>• Training dataset 100k+</li>
                <li>• Continuous learning</li>
                <li>• Accuracy optimization</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===== Tombol di bawah =====
st.markdown("""
<div style='text-align:center; margin-top:40px;'>
    <a href='#pengaturan' style='background:linear-gradient(135deg,#9333ea,#ec4899); 
    color:white; padding:12px 35px; border-radius:30px; font-weight:600; text-decoration:none; 
    box-shadow:0 6px 18px rgba(147,51,234,0.3); transition:0.3s;'>↓ Lanjut ke Pengaturan</a>
</div>

<style>
html { scroll-behavior: smooth; }
</style>
""", unsafe_allow_html=True)

# ==========================
# PERSONALISASI USER
# ==========================
# --- Inisialisasi tema ---
if "tema" not in st.session_state:
    st.session_state.tema = "terang"

# --- Warna tema ---
if st.session_state.tema == "terang":
    bg_color = "#E6FAF2"
    text_color = "#0f172a"
    card_color = "#ffffff"
    border_color = "#00a86b"
    accent_color = "#00a86b"
else:
    bg_color = "#0f172a"
    text_color = "#f1f5f9"
    card_color = "#1e293b"
    border_color = "#334155"
    accent_color = "#00a86b"

# --- CSS Styling ---
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    font-family: 'Poppins', sans-serif;
    color: {text_color};
}}
h1 {{
    text-align: center;
    font-weight: 800;
    font-size: 42px;
    margin-bottom: 5px;
}}
p.subtitle {{
    text-align: center;
    font-size: 17px;
    color: #64748b;
    margin-bottom: 40px;
}}
.card {{
    background-color: {card_color};
    border-radius: 22px;
    padding: 40px 60px;
    max-width: 900px;
    margin: 0 auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}}
.section-wrapper {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 40px;
    flex-wrap: wrap;
}}
.left, .right {{
    flex: 1;
    min-width: 320px;
}}
.theme-option {{
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: left;
    cursor: pointer;
    transition: 0.3s;
    font-weight: 500;
}}
.theme-option:hover {{
    border-color: {accent_color};
    box-shadow: 0 4px 12px rgba(0,168,107,0.15);
}}
.selected {{
    border-color: {accent_color};
    background-color: rgba(0,168,107,0.1);
}}
.button {{
    background-color: {accent_color};
    color: white;
    padding: 14px 40px;
    border-radius: 30px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    margin-top: 30px;
    transition: 0.3s;
    box-shadow: 0 6px 16px rgba(0,168,107,0.25);
}}
.button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 20px rgba(0,168,107,0.35);
}}
</style>
""", unsafe_allow_html=True)

# --- Judul dan Subjudul ---
st.markdown("<h1>Personalisasi Pengalaman Anda</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Masukkan nama dan pilih tema tampilan sesuai preferensi Anda</p>", unsafe_allow_html=True)

# --- Kartu utama ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-wrapper'>", unsafe_allow_html=True)

# === Bagian kiri: Input nama ===
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='left'>", unsafe_allow_html=True)
    st.markdown("**👤 Nama Anda**")
    nama = st.text_input("", placeholder="Masukkan nama Anda...", key="nama_input")

    if nama:
        st.markdown(f"<div style='margin-top:10px; background-color:rgba(0,168,107,0.1); padding:12px 18px; border-radius:10px; font-weight:500;'>🌸 Halo, {nama}! Selamat datang di <b>AI Flower Vision</b></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# === Bagian kanan: Pilih tema ===
with col2:
    st.markdown("<div class='right'>", unsafe_allow_html=True)
    st.markdown("**🎨 Tema Tampilan**")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        terang_active = "selected" if st.session_state.tema == "terang" else "theme-option"
        if st.button("☀️ Tema Terang\nTampilan cerah dan segar", key="tema_terang"):
            st.session_state.tema = "terang"
            st.rerun()

    with col_t2:
        gelap_active = "selected" if st.session_state.tema == "gelap" else "theme-option"
        if st.button("🌙 Tema Gelap\nTampilan elegan dan nyaman", key="tema_gelap"):
            st.session_state.tema = "gelap"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end section-wrapper
st.markdown("</div>", unsafe_allow_html=True)  # end card

# --- Tombol Lanjut ---
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if st.button("⬇️ Mulai Analisis Gambar", key="lanjut"):
    st.markdown("<script>window.location.href='#analisis';</script>", unsafe_allow_html=True)
st.markdown(f"<button class='button'>Mulai Analisis Gambar</button>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


