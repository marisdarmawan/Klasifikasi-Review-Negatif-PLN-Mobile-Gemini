import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart CSO - AI Powered Classification",
    page_icon="⚡",
    layout="centered"
)

# --- SETUP API KEY GEMINI ---
# Mengambil API Key dari Streamlit Secrets (Aman dari kebocoran)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
except Exception as e:
    st.error("⚠️ API Key Gemini belum diatur di Streamlit Secrets!")
    st.stop()

# --- LOAD DAFTAR KATEGORI ---
@st.cache_data
def load_kategori():
    # Membaca file Klasifikasi Ulasan.csv (perhatikan pemisahnya adalah titik koma)
    df_tax = pd.read_csv('Klasifikasi Ulasan.csv', sep=';')
    
    # Menggabungkan layer menjadi Path Lengkap
    df_tax['Path_Lengkap'] = df_tax.apply(
        lambda row: f"{str(row['Layer 1']).strip()} > {str(row['Layer 2']).strip()} > {str(row['Layer 3']).strip()} > {str(row['Layer 4']).strip()} > {str(row['Layer 5']).strip()}", 
        axis=1
    )
    
    # Hapus duplikat dan null, lalu jadikan list
    daftar_kategori = df_tax['Path_Lengkap'].dropna().unique().tolist()
    return daftar_kategori

daftar_kategori_pln = load_kategori()

# --- FUNGSI PREDIKSI LLM (GEMINI) ---
def prediksi_gemini(ulasan_teks):
    # Gabungkan semua kategori menjadi satu teks panjang dengan nomor urut
    kategori_str = "\n".join([f"{i+1}. {kat}" for i, kat in enumerate(daftar_kategori_pln)])
    
    # Prompt Engineering Khusus untuk Gemini
    prompt = f"""
    Kamu adalah AI Customer Service PLN yang sangat cerdas.
    Tugasmu adalah mengklasifikasikan keluhan pelanggan HANYA ke salah satu kategori dari daftar di bawah ini, serta menentukan sentimennya.

    DAFTAR KATEGORI TIKET:
    {kategori_str}

    ULASAN PELANGGAN:
    "{ulasan_teks}"

    ATURAN:
    1. Pilih 1 Kategori yang paling tepat dari daftar di atas. Tulis sama persis.
    2. Tentukan sentimen (Positif / Negatif / Netral).
    3. Berikan alasan singkat 1-2 kalimat mengapa kamu memilih kategori tersebut.

    Keluarkan jawabanmu murni dalam format JSON seperti ini:
    {{"kategori": "...", "sentimen": "...", "alasan": "..."}}
    """
    
    # Panggil Model Gemini Flash dengan output JSON yang dipaksa
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1 # Suhu rendah agar AI tidak berhalusinasi
        )
    )
    
    return json.loads(response.text)

# --- USER INTERFACE (UI) ---
st.title("⚡ Smart CSO - AI Powered Classification")
st.markdown("**PLN Mobile Customer Support Ticketing System (PoC)**")
st.divider()

st.subheader("📝 Masukkan Ulasan Pelanggan")
teks_input = st.text_area(
    "Ketik atau Paste Review Pelanggan:", 
    height=120, 
    placeholder="Contoh: mati lampu dari jam 2 siang gak nyala-nyala woy, trus mau lapor di aplikasi muter terus.."
)

if st.button("🔍 Analisis dengan AI", type="primary", use_container_width=True):
    if teks_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner("🧠 AI sedang memikirkan konteks ulasan..."):
            try:
                # Waktu mulai
                start_time = time.time()
                
                # Eksekusi AI
                hasil = prediksi_gemini(teks_input)
                
                # Hitung waktu eksekusi
                waktu_proses = time.time() - start_time
                
                st.success(f"Analisis Selesai! (dalam {waktu_proses:.1f} detik)")
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="🎭 Sentimen", value=hasil.get('sentimen', 'Netral'))
                with col2:
                    st.metric(label="⚡ AI Engine", value="Gemini 2.5 Flash")
                    
                st.info(f"**📂 Kategori Tiket:**\n\n{hasil.get('kategori', 'Tidak Ditemukan')}")
                st.write(f"**🤖 Alasan AI:** {hasil.get('alasan', '-')}")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memanggil AI: {e}")
                
st.divider()
st.caption("Dikembangkan oleh Tim Data Science Divisi Manajemen Digital")
