# backend/ai_engine.py (Versi FINAL Lengkap)

import json
import fitz # PyMuPDF
from PIL import Image

# Impor semua fungsi dan getter dari file-file helper kita
from .konteks_extractor import (
    analisis_halaman_dengan_layoutlmv3,
    tata_ulang_dengan_flan_t5,
    get_models # Impor getter utama
)
from .validasi_konten import (
    _gabungkan_token_menjadi_entitas,
    cek_validitas_isian_data,
    deteksi_tipe_dokumen_dari_hasil_ai,
    ATURAN_VALIDASI
)

def run_ai_pipeline(path_pdf_str: str, nama_file_asli: str) -> dict:
    """
    Fungsi utama pipeline AI. Memuat model (jika perlu) dan memproses satu PDF.
    """
    print(f"--- AI Engine Mulai: Memproses {nama_file_asli} ---")
    
    # Langkah 0: Pastikan semua model sudah dimuat ke memori (GPU/CPU)
    # Getter akan menangani pemuatan hanya jika belum ada.
    get_models()
    
    laporan_final = {}
    try:
        # Langkah 1: Analisis Kontekstual (LayoutLMv3) per halaman
        print("AI Engine: [1/3] Memulai analisis kontekstual (LayoutLMv3)...")
        doc = fitz.open(path_pdf_str)
        hasil_kontekstual_proyek = []
        for page_num in range(len(doc)):
            print(f"  - Menganalisis Halaman {page_num+1}/{len(doc)}")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            hasil_analisis_halaman = analisis_halaman_dengan_layoutlmv3(image)
            if hasil_analisis_halaman is None:
                hasil_analisis_halaman = {"hasil_analisis_kontekstual": []}
            
            hasil_kontekstual_proyek.append({"halaman": page_num + 1, "analisis": hasil_analisis_halaman})
        doc.close()

        # Langkah 2: Rekonstruksi (FLAN-T5) per halaman
        print("AI Engine: [2/3] Memulai rekonstruksi data (FLAN-T5)...")
        hasil_per_halaman = []
        semua_hasil_ekstraksi_dokumen = {}
        for item in hasil_kontekstual_proyek:
            page_num = item["halaman"]
            analisis_mentah = item.get('analisis', {}).get('hasil_analisis_kontekstual', [])
            
            data_terstruktur = {}
            if analisis_mentah:
                entitas_halaman = _gabungkan_token_menjadi_entitas(analisis_mentah)
                if entitas_halaman:
                    data_terstruktur = tata_ulang_dengan_flan_t5(entitas_halaman)
            
            hasil_per_halaman.append({"halaman": page_num, "hasil_ekstraksi": data_terstruktur})
            semua_hasil_ekstraksi_dokumen.update(data_terstruktur)

        laporan_final["detail_per_halaman"] = hasil_per_halaman

        # Langkah 3: Deteksi Tipe Dokumen & Validasi Isian Data
        print("AI Engine: [3/3] Memulai validasi isian data...")
        tipe_dokumen = deteksi_tipe_dokumen_dari_hasil_ai(semua_hasil_ekstraksi_dokumen, nama_file_asli)
        
        # Lakukan validasi untuk setiap halaman yang sudah diekstrak
        for hasil in laporan_final["detail_per_halaman"]:
            laporan_validasi = cek_validitas_isian_data(hasil["hasil_ekstraksi"], tipe_dokumen)
            hasil["validasi_isian_data"] = laporan_validasi

        laporan_final["tipe_dokumen_terdeteksi"] = tipe_dokumen
        print("--- AI Engine Selesai: Proses berhasil ---")
        return laporan_final

    except Exception as e:
        print(f"--- AI Engine Error: Terjadi kesalahan fatal di pipeline ---")
        # Kirim ulang error agar bisa ditangkap oleh endpoint FastAPI
        raise e