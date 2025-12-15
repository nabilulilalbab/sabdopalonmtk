"""
🎓 SISTEM PREDIKSI NILAI MATEMATIKA SISWA BERBASIS FAKTOR SOSIAL

Aplikasi ini memprediksi nilai matematika siswa berdasarkan faktor sosial:
- Jenis Kelamin
- Kelompok Etnis
- Pendidikan Orang Tua
- Tipe Makan Siang
- Kursus Persiapan Ujian
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ========================================
# KONFIGURASI HALAMAN
# ========================================
st.set_page_config(
    page_title="Prediksi Nilai Matematika Siswa",
    page_icon="🎓",
    layout="centered"
)

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, feature_names
    except FileNotFoundError:
        st.error("⚠️ Model belum di-training! Jalankan 'python main.py' terlebih dahulu.")
        st.stop()

model, feature_names = load_model()

# ========================================
# HEADER
# ========================================
st.title("🎓 Sistem Prediksi Nilai Matematika Siswa")
st.markdown("### Berbasis Faktor Sosial dan Demografi")
st.markdown("---")

# ========================================
# PENJELASAN SISTEM (UNTUK PRESENTASI)
# ========================================
with st.expander("📖 Tentang Sistem Ini", expanded=False):
    st.markdown("### 🎯 Konsep Sistem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📌 Input (5 Fitur):**")
        st.info("""
        1. Jenis Kelamin
        2. Kelompok Etnis (A-E)
        3. Pendidikan Orang Tua
        4. Tipe Makan Siang
        5. Kursus Persiapan Ujian
        """)
        
        st.markdown("**📌 Output:**")
        st.success("""
        - Prediksi Nilai (0-100)
        - Kategori Nilai
        - Insight & Penjelasan
        """)
    
    with col2:
        st.markdown("**🤖 Model:**")
        st.warning("""
        **Algoritma:** Linear Regression
        
        **Dataset:** 1000 siswa
        
        **Metode:** Ordinary Least Squares (OLS)
        
        **Akurasi:** MAE ±11.27 poin
        """)
    
    st.markdown("---")
    st.markdown("### 🏗️ Arsitektur Sistem")
    st.code("""
    Input Pengguna (5 Fitur)
           ↓
    Antarmuka Streamlit
           ↓
    Model Machine Learning (Linear Regression)
           ↓
    Prediksi Nilai Matematika
    """, language="")
    
    st.markdown("---")
    st.markdown("### 📊 Apa itu Kelompok Etnis A-E?")
    
    st.warning("""
    **PENTING:** Kelompok A-E adalah klasifikasi **DEMOGRAFI** berdasarkan 
    **latar belakang SOSIAL-EKONOMI**, BUKAN tentang ras atau suku bangsa.
    """)
    
    # Tabel rata-rata per kelompok
    kelompok_data = {
        "Kelompok": ["Kelompok E", "Kelompok D", "Kelompok C", "Kelompok B", "Kelompok A"],
        "Rata-rata Nilai": [73.82, 67.36, 64.46, 63.45, 61.63],
        "Jumlah Siswa": [140, 262, 319, 190, 89],
        "Status": ["⭐ Tertinggi", "✅ Tinggi", "📊 Menengah", "📉 Rendah", "⚠️ Terendah"]
    }
    st.dataframe(kelompok_data, use_container_width=True)
    
    st.info("""
    **Insight:** Perbedaan 12.19 poin antara Kelompok E dan A menunjukkan 
    **pengaruh SIGNIFIKAN** latar belakang sosial-ekonomi terhadap prestasi akademik.
    
    Kelompok ini mencerminkan:
    - Status ekonomi keluarga
    - Lokasi geografis (urban vs rural)  
    - Akses ke sumber daya pendidikan
    - Lingkungan sosial
    """)

st.markdown("---")

# ========================================
# TABS INFORMASI LENGKAP
# ========================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Cara Kerja", "📊 Analisis Data", "🤖 Algoritma", "💡 Insight"])

with tab1:
    st.markdown("## 🎯 Cara Kerja Sistem")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Input")
        st.info("""
        User memasukkan 5 faktor sosial:
        - Jenis Kelamin
        - Kelompok Etnis
        - Pendidikan Orang Tua
        - Tipe Makan Siang
        - Kursus Persiapan
        """)
    
    with col2:
        st.markdown("### 2️⃣ Proses")
        st.warning("""
        Model Linear Regression:
        - Encoding One-Hot
        - Kalkulasi koefisien
        - Formula: y = β₀ + Σ(βᵢXᵢ)
        - Prediksi nilai
        """)
    
    with col3:
        st.markdown("### 3️⃣ Output")
        st.success("""
        Hasil prediksi:
        - Nilai matematika (0-100)
        - Kategori prestasi
        - Insight & rekomendasi
        - Visual progress bar
        """)
    
    st.markdown("---")
    st.markdown("### 🏗️ Arsitektur Sistem")
    
    col1, col2, col3, col4, col5 = st.columns([1, 0.3, 1, 0.3, 1])
    
    with col1:
        st.info("**USER INPUT**\n\n5 Fitur Sosial")
    
    with col2:
        st.markdown("### →")
    
    with col3:
        st.warning("**STREAMLIT UI**\n\nEncoding & Validasi")
    
    with col4:
        st.markdown("### →")
    
    with col5:
        st.success("**ML MODEL**\n\nPrediksi Nilai")

with tab2:
    st.markdown("## 📊 Analisis Data - Objektif & Terbukti")
    
    st.info("**Dataset:** 1000 siswa | **Missing Values:** 0 | **Fitur:** 5 input + 3 target")
    
    st.markdown("### 📈 Statistik Nilai Matematika")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rata-rata", "66.09", help="Mean dari 1000 siswa")
    
    with col2:
        st.metric("Median", "66.00", help="Nilai tengah")
    
    with col3:
        st.metric("Std Dev", "15.16", help="Standar deviasi")
    
    with col4:
        st.metric("Range", "0-100", help="Minimum - Maximum")
    
    st.markdown("---")
    st.markdown("### 🔍 Gap Sosial-Ekonomi (TERBUKTI DARI DATA)")
    
    gap_data = {
        "Faktor": [
            "Tipe Makan Siang",
            "Kelompok Etnis",
            "Kursus Persiapan",
            "Jenis Kelamin"
        ],
        "Kategori Tinggi": [
            "Standard (70.03)",
            "Kelompok E (73.82)",
            "Selesai (69.70)",
            "Laki-laki (68.73)"
        ],
        "Kategori Rendah": [
            "Bersubsidi (58.92)",
            "Kelompok A (61.63)",
            "Tidak Ikut (64.08)",
            "Perempuan (63.63)"
        ],
        "Gap": [
            "+11.11 poin",
            "+12.19 poin",
            "+5.62 poin",
            "+5.10 poin"
        ],
        "Dampak": ["⭐ Tertinggi", "⭐ Tertinggi", "✅ Signifikan", "✅ Signifikan"]
    }
    
    st.dataframe(gap_data, use_container_width=True)
    
    st.success("""
    **Kesimpulan:**
    - Faktor sosial-ekonomi **TERBUKTI** mempengaruhi nilai secara signifikan
    - Gap 11-12 poin adalah perbedaan **BESAR** dalam skala 0-100
    - Kursus persiapan adalah faktor yang **BISA DIINTERVENSI** oleh sekolah
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Distribusi Kelompok Etnis")
    
    kelompok_detail = {
        "Kelompok": ["Kelompok E", "Kelompok D", "Kelompok C", "Kelompok B", "Kelompok A"],
        "Jumlah Siswa": [140, 262, 319, 190, 89],
        "Persentase": ["14%", "26.2%", "31.9%", "19%", "8.9%"],
        "Rata-rata Nilai": [73.82, 67.36, 64.46, 63.45, 61.63],
        "Median": [74.5, 69.0, 65.0, 63.0, 61.0],
        "Status": ["⭐ Terbaik", "✅ Baik", "📊 Menengah", "📉 Kurang", "⚠️ Rendah"]
    }
    
    st.dataframe(kelompok_detail, use_container_width=True)
    
    st.warning("""
    **PENTING:** Kelompok A-E adalah klasifikasi **DEMOGRAFI SOSIAL-EKONOMI**, 
    BUKAN tentang ras/suku bangsa. Mencerminkan:
    - Status ekonomi keluarga
    - Lokasi geografis (urban vs rural)
    - Akses ke sumber daya pendidikan
    - Lingkungan sosial
    """)

with tab3:
    st.markdown("## 🤖 Algoritma & Training")
    
    st.markdown("### 📚 Linear Regression")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Apa itu Linear Regression?**")
        st.info("""
        Algoritma **supervised learning** yang memodelkan hubungan linear 
        antara variabel input (X) dan output (y).
        
        **Formula:**
        ```
        y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
        ```
        
        **Dimana:**
        - y = Prediksi nilai matematika
        - β₀ = Intercept (konstanta)
        - βᵢ = Koefisien fitur ke-i
        - Xᵢ = Nilai fitur ke-i
        """)
        
        st.markdown("**Kenapa Linear Regression?**")
        st.success("""
        ✅ **Interpretable** - Koefisien jelas
        ✅ **Cepat** - Training <1 detik
        ✅ **Simple** - Tidak overfitting
        ✅ **Explainable** - Mudah dijelaskan
        """)
    
    with col2:
        st.markdown("**Proses Training:**")
        st.warning("""
        **1. Preprocessing**
        - One-Hot Encoding
        - 5 fitur → 12 kolom binary
        
        **2. Split Data**
        - Training: 800 siswa (80%)
        - Testing: 200 siswa (20%)
        
        **3. Training**
        - Metode: OLS (Ordinary Least Squares)
        - Objective: Minimize Σ(y - ŷ)²
        - Waktu: <1 detik
        
        **4. Evaluasi**
        - MAE: 11.27 poin
        - RMSE: 14.16 poin
        - R²: 0.176 (17.6%)
        
        **5. Save Model**
        - model.pkl (1.4 KB)
        """)
    
    st.markdown("---")
    st.markdown("### 🔍 Koefisien Model (Interpretasi)")
    
    st.info("**Intercept (β₀):** 59.09 - Nilai dasar sebelum faktor lain")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 3 Positif (Meningkatkan Nilai):**")
        koef_positif = {
            "Fitur": [
                "Makan Siang Standard",
                "Kelompok E",
                "Jenis Kelamin Laki-laki"
            ],
            "Koefisien": ["+11.52", "+9.08", "+4.52"],
            "Interpretasi": [
                "⭐ Paling berpengaruh!",
                "Latar belakang terbaik",
                "Cenderung lebih tinggi"
            ]
        }
        st.dataframe(koef_positif, use_container_width=True)
    
    with col2:
        st.markdown("**Top 3 Negatif (Menurunkan Nilai):**")
        koef_negatif = {
            "Fitur": [
                "Tidak Ikut Kursus",
                "Pendidikan Ortu: SMA",
                "Pendidikan Ortu: <SMA"
            ],
            "Koefisien": ["-5.87", "-4.09", "-2.90"],
            "Interpretasi": [
                "⚠️ Bisa diintervensi!",
                "Dukungan terbatas",
                "Dukungan minimal"
            ]
        }
        st.dataframe(koef_negatif, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 📊 Evaluasi Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "MAE",
            "11.27 poin",
            help="Mean Absolute Error - rata-rata meleset ±11 poin"
        )
    
    with col2:
        st.metric(
            "RMSE",
            "14.16 poin",
            help="Root Mean Squared Error - lebih sensitif terhadap error besar"
        )
    
    with col3:
        st.metric(
            "R²",
            "17.6%",
            help="Model menjelaskan 17.6% variasi nilai"
        )
    
    st.warning("""
    **❓ Kenapa R² hanya 17.6%?**
    
    Model HANYA menggunakan **5 faktor sosial-demografis**. 
    
    Sisanya (82.4%) dipengaruhi faktor lain yang tidak ada di dataset:
    - IQ / kemampuan kognitif
    - Motivasi belajar
    - Jam belajar per hari
    - Kualitas guru/sekolah
    - Metode pembelajaran
    - Dukungan keluarga (non-material)
    - Kondisi kesehatan
    
    **R² 17.6% menunjukkan faktor sosial MEMANG berpengaruh signifikan!**
    Ini WAJAR dan OBJEKTIF untuk model berbasis faktor sosial saja.
    """)

with tab4:
    st.markdown("## 💡 Insight Penting")
    
    st.markdown("### 🎯 Temuan Utama")
    
    st.success("""
    **1. Makan Siang Standard = Faktor Terkuat (+11.52 poin)**
    
    Ini adalah indikator kuat kondisi ekonomi keluarga. Siswa dari keluarga 
    mampu (makan siang standard) memiliki nilai 11 poin lebih tinggi dibanding 
    siswa yang dapat subsidi makan.
    """)
    
    st.info("""
    **2. Kelompok Etnis = Pengaruh Signifikan (+12.19 poin gap)**
    
    Perbedaan 12 poin antara Kelompok E dan A menunjukkan latar belakang 
    sosial-ekonomi sangat mempengaruhi prestasi. Ini bukan tentang ras, 
    tapi tentang akses ke sumber daya pendidikan.
    """)
    
    st.warning("""
    **3. Kursus Persiapan = Faktor yang BISA DIINTERVENSI (+5.62 poin)**
    
    Ini adalah INSIGHT PALING PENTING! Walaupun faktor ekonomi sulit diubah, 
    sekolah BISA memberikan intervensi melalui program kursus persiapan ujian.
    
    **Rekomendasi:** Fokuskan program persiapan untuk siswa dari keluarga 
    kurang mampu (yang dapat makan siang bersubsidi).
    """)
    
    st.markdown("---")
    st.markdown("### 🎓 Rekomendasi untuk Sekolah")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Apa yang BISA dilakukan:**")
        st.success("""
        1. **Program Kursus Gratis**
           - Untuk siswa kurang mampu
           - Fokus persiapan ujian
           - Dampak: +5.6 poin
        
        2. **Identifikasi Siswa Berisiko**
           - Dari Kelompok A/B
           - Orang tua pendidikan rendah
           - Dapat makan siang bersubsidi
        
        3. **Program Mentoring**
           - Peer tutoring
           - Bimbingan akademik
           - Dukungan psikososial
        """)
    
    with col2:
        st.markdown("**⚠️ Apa yang SULIT diubah:**")
        st.warning("""
        1. **Kondisi Ekonomi Keluarga**
           - Butuh kebijakan makro
           - Di luar kendali sekolah
           
        2. **Pendidikan Orang Tua**
           - Sudah permanen
           - Tidak bisa diubah
           
        3. **Latar Belakang Sosial**
           - Faktor sistemik
           - Butuh intervensi jangka panjang
        """)
    
    st.markdown("---")
    st.markdown("### 📈 Proyeksi Dampak Intervensi")
    
    st.info("""
    **Skenario: Siswa Kurang Mampu**
    - Kondisi awal: Kelompok A, Orang tua SMA tidak lulus, Makan siang bersubsidi
    - Prediksi tanpa intervensi: ~50 poin
    
    **Jika diberi program kursus persiapan:**
    - Prediksi dengan intervensi: ~56 poin (+5.87 poin)
    - Peningkatan: 11.7%
    - ROI: Tinggi (program murah, dampak signifikan)
    """)
    
    st.success("""
    **Kesimpulan:** Investasi dalam program persiapan ujian untuk siswa 
    kurang mampu adalah strategi cost-effective untuk meningkatkan prestasi 
    akademik dan mengurangi kesenjangan sosial-ekonomi.
    """)

st.markdown("---")

# ========================================
# FORM INPUT
# ========================================
st.markdown("## 📝 Masukkan Data Siswa")

# Mapping untuk label Indonesia
gender_label = {"female": "Perempuan", "male": "Laki-laki"}
race_label = {
    "group A": "Kelompok A (Rata-rata: 61.6)",
    "group B": "Kelompok B (Rata-rata: 63.5)",
    "group C": "Kelompok C (Rata-rata: 64.5)",
    "group D": "Kelompok D (Rata-rata: 67.4)",
    "group E": "Kelompok E (Rata-rata: 73.8)"
}
lunch_label = {"standard": "Standard", "free/reduced": "Bersubsidi (Gratis/Diskon)"}
edu_label = {
    "some high school": "SMA Tidak Lulus",
    "high school": "Lulusan SMA",
    "some college": "Kuliah Tidak Lulus",
    "associate's degree": "Diploma (D3)",
    "bachelor's degree": "Sarjana (S1)",
    "master's degree": "Magister (S2)"
}
test_label = {"none": "Tidak Ikut", "completed": "Selesai"}

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "👤 Jenis Kelamin",
        ["female", "male"],
        format_func=lambda x: gender_label[x],
        help="Jenis kelamin siswa"
    )
    
    race = st.selectbox(
        "🌍 Kelompok Etnis",
        ["group A", "group B", "group C", "group D", "group E"],
        format_func=lambda x: race_label[x],
        help="Kelompok demografi siswa (A-E berdasarkan latar belakang sosial-ekonomi)"
    )
    
    lunch = st.selectbox(
        "🍽️ Tipe Makan Siang",
        ["standard", "free/reduced"],
        format_func=lambda x: lunch_label[x],
        help="Tipe makan siang yang diterima siswa"
    )

with col2:
    parental_education = st.selectbox(
        "🎓 Pendidikan Orang Tua",
        ["some high school", "high school", "some college", 
         "associate's degree", "bachelor's degree", "master's degree"],
        format_func=lambda x: edu_label[x],
        help="Tingkat pendidikan tertinggi orang tua/wali"
    )
    
    test_prep = st.selectbox(
        "📚 Kursus Persiapan Ujian",
        ["none", "completed"],
        format_func=lambda x: test_label[x],
        help="Apakah siswa mengikuti kursus persiapan ujian"
    )

st.markdown("---")

# ========================================
# PREDIKSI
# ========================================
if st.button("🔮 Prediksi Nilai Matematika", type="primary", use_container_width=True):
    
    # Encoding manual (HARUS SAMA DENGAN TRAINING)
    # Buat dictionary dengan SEMUA fitur dari training, default 0
    data = {col: 0 for col in feature_names}
    
    # Update nilai berdasarkan input user
    # Gender
    if gender == "male":
        if "gender_male" in data:
            data["gender_male"] = 1
    
    # Race/Ethnicity
    race_col = f"race/ethnicity_{race}"
    if race_col in data:
        data[race_col] = 1
    
    # Parental Education
    parent_col = f"parental level of education_{parental_education}"
    if parent_col in data:
        data[parent_col] = 1
    
    # Lunch
    if lunch == "standard":
        if "lunch_standard" in data:
            data["lunch_standard"] = 1
    
    # Test Prep
    if test_prep == "none":
        if "test preparation course_none" in data:
            data["test preparation course_none"] = 1
    
    # Buat DataFrame dengan urutan kolom yang sama
    df_input = pd.DataFrame([data])
    
    # Prediksi
    try:
        prediction = model.predict(df_input)[0]
        
        # Batasi nilai antara 0-100
        prediction = max(0, min(100, prediction))
        
        # Tampilkan hasil
        st.markdown("## 🎯 Hasil Prediksi")
        
        # Tampilan nilai prediksi besar
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{prediction:.1f}</h1>", 
                    unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 20px;'>Prediksi Nilai Matematika</p>", 
                    unsafe_allow_html=True)
        
        # Progress bar visual
        st.progress(prediction / 100)
        
        # Kategori nilai
        if prediction >= 80:
            kategori = "🌟 Sangat Baik"
            warna = "success"
        elif prediction >= 70:
            kategori = "✅ Baik"
            warna = "success"
        elif prediction >= 60:
            kategori = "⚠️ Cukup"
            warna = "warning"
        else:
            kategori = "❌ Perlu Peningkatan"
            warna = "error"
        
        st.markdown(f"**Kategori:** {kategori}")
        
        # Penjelasan singkat
        st.markdown("### 💡 Penjelasan")
        
        if test_prep == "completed":
            st.info("✅ Siswa mengikuti kursus persiapan ujian - ini dapat meningkatkan nilai sekitar 5-6 poin!")
        else:
            st.warning("⚠️ Siswa tidak mengikuti kursus persiapan ujian - mengikuti kursus dapat meningkatkan nilai sekitar 5-6 poin.")
        
        if lunch == "free/reduced":
            st.info("📊 Siswa menerima makan siang bersubsidi - ini mungkin berkorelasi dengan faktor ekonomi keluarga yang dapat mempengaruhi nilai.")
        else:
            st.success("✅ Siswa menerima makan siang standard - ini berkorelasi positif dengan nilai (+11.5 poin rata-rata).")
        
        # Insight tambahan berdasarkan kelompok
        if race == "group E":
            st.success("🌟 Kelompok E memiliki rata-rata nilai tertinggi (73.8) - prediksi cenderung lebih tinggi.")
        elif race == "group A":
            st.info("📊 Kelompok A memiliki rata-rata nilai terendah (61.6) - namun ini bisa ditingkatkan dengan kursus persiapan!")
        
        # Info tambahan
        with st.expander("ℹ️ Informasi Tambahan"):
            st.markdown(f"""
            **Data Input:**
            - Jenis Kelamin: {gender_label[gender]}
            - Kelompok Etnis: {race_label[race]}
            - Pendidikan Orang Tua: {edu_label[parental_education]}
            - Tipe Makan Siang: {lunch_label[lunch]}
            - Kursus Persiapan: {test_label[test_prep]}
            
            **Tentang Kelompok Etnis:**
            - Kelompok A-E adalah klasifikasi demografi dalam dataset
            - Mencerminkan perbedaan latar belakang sosial-ekonomi
            - Group E (rata-rata 73.8) vs Group A (rata-rata 61.6)
            - Ini menunjukkan pentingnya faktor sosial dalam prestasi akademik
            
            **Catatan:**
            - Prediksi berdasarkan model Linear Regression
            - Model dilatih menggunakan data 1000 siswa
            - Akurasi: MAE ±11.27 poin
            - Prediksi bersifat estimasi, bukan penilaian pasti
            """)
    
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
        st.info("Pastikan model sudah di-training dengan benar menggunakan 'python main.py'")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎓 Sistem Prediksi Nilai Matematika Siswa Berbasis Faktor Sosial</p>
    <p>Dibuat menggunakan Python, Streamlit & Scikit-learn</p>
    <p style='font-size: 12px;'>Model: Linear Regression | Dataset: 1000 siswa | MAE: ±11.27</p>
</div>
""", unsafe_allow_html=True)
