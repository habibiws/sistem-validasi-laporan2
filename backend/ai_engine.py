# backend/ai_engine.py
import json
import fitz # PyMuPDF
from PIL import Image

# Impor semua fungsi dan getter dari proyek kita
from backend.konteks_extractor import (
    analisis_halaman_dengan_layoutlmv3,
    # tata_ulang_dengan_indobert_lokal
)
from backend.validasi_konten import (
    _gabungkan_token_menjadi_entitas,
    cek_validitas_isian_data,
    deteksi_tipe_dokumen_dari_hasil_ai,
    tata_ulang_dengan_indobert_lokal,
    ATURAN_VALIDASI
)

# Fungsi utama yang akan dipanggil oleh endpoint AI
def run_ai_pipeline(path_pdf_str: str, nama_file_asli: str) -> dict:
    print(f"--- AI Engine Mulai: Memproses {nama_file_asli} ---")
    laporan_final = {}

    try:
        # 1. Analisis Kontekstual (LayoutLMv3)
        print("AI Engine: [1/3] Memulai analisis kontekstual...")
        doc = fitz.open(path_pdf_str)
        hasil_kontekstual_proyek = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Pastikan analisis halaman tidak crash pada halaman kosong
            hasil_analisis_halaman = analisis_halaman_dengan_layoutlmv3(image) or {"hasil_analisis_kontekstual": []}
            hasil_kontekstual_proyek.append({"halaman": page_num + 1, "analisis": hasil_analisis_halaman})
        doc.close()

        # 2. Rekonstruksi (IndoBERT)
        print("AI Engine: [2/3] Memulai rekonstruksi data...")
        semua_entitas_dokumen = []
        for hasil_halaman in hasil_kontekstual_proyek:
            analisis_mentah = hasil_halaman.get('analisis', {}).get('hasil_analisis_kontekstual', [])
            if analisis_mentah: # Hanya proses jika ada hasil
                entitas_halaman = _gabungkan_token_menjadi_entitas(analisis_mentah)
                semua_entitas_dokumen.extend(entitas_halaman)
        
        if not semua_entitas_dokumen:
            data_terstruktur = {"error": "Tidak ada konten teks yang terdeteksi di dalam dokumen."}
        else:
            semua_entitas_dokumen.sort(key=lambda e: (e.get('halaman_asal', 1), e["box"][1], e["box"][0]))
            data_terstruktur = tata_ulang_dengan_indobert_lokal(semua_entitas_dokumen)
        
        laporan_final["data_terstruktur"] = data_terstruktur

        # 3. Validasi Isian Data
        print("AI Engine: [3/3] Memulai validasi isian data...")
        tipe_dokumen = deteksi_tipe_dokumen_dari_hasil_ai(data_terstruktur, nama_file_asli)
        hasil_validasi = cek_validitas_isian_data(data_terstruktur, tipe_dokumen)
        laporan_final["validasi_isian_data"] = hasil_validasi
        
        print("--- AI Engine Selesai: Proses berhasil ---")
        return laporan_final

    except Exception as e:
        print(f"--- AI Engine Error: Terjadi kesalahan ---")
        # Raise error agar bisa ditangkap oleh endpoint
        raise e