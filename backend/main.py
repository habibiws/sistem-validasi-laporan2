# backend/main.py
import os
import shutil
import json
import glob
import sys
import uuid
import fitz
from pathlib import Path
from datetime import datetime
from typing import List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- PERBAIKAN DI SINI ---
# Hapus titik (.) dari semua impor lokal
from backend.ekstraksi_pdf import ekstrak_aset_terstruktur, simpan_hasil_ke_disk
from backend.validasi_foto import proses_validasi_dengan_petunjuk
from backend.konteks_extractor import load_model, analisis_halaman_dengan_layoutlmv3, visualisasikan_hasil_analisis, load_brain_model
from backend.validasi_konten import cek_kelengkapan_dokumen, rekonstruksi_key_value, _gabungkan_token_menjadi_entitas, tata_ulang_dengan_llm
# --------------------------

# Muat model AI saat startup
app = FastAPI(
    title="Sistem Validasi Laporan Otomatis",
    version="2.3.0-content-validation",
    description="API dengan validasi kelengkapan dokumen.",
    on_startup=[load_model, load_brain_model]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pengaturan Path direktori di dalam kontainer
# BASE_DIR sekarang adalah /app
# BASE_DIR = Path(__file__).resolve().parent.parent 
# Kita akan gunakan path absolut di dalam kontainer
DATA_DIR = Path("data") # Asumsikan data akan di-mount ke /app/data
# Jika Anda me-mount 'data' ke root, gunakan Path("/data")

# Sesuaikan path-path ini jika Anda mengubah struktur volume di docker-compose
INPUT_PDF_DIR = DATA_DIR / "input_pdf"
OUTPUT_EKSTRAKSI_DIR = DATA_DIR / "output_ekstraksi"
SISTEM_VALIDASI_DIR = DATA_DIR / "sistem_validasi"
PATH_MASTER_INDEX = SISTEM_VALIDASI_DIR / "master_index.json"

INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EKSTRAKSI_DIR.mkdir(parents=True, exist_ok=True)
SISTEM_VALIDASI_DIR.mkdir(parents=True, exist_ok=True)

EKSTENSI_GAMBAR = ["jpg", "jpeg", "png", "bmp"]

# aturan kelengkapan dokumen
ATURAN_KELENGKAPAN = {
    "frasa_wajib": [
        "DOKUMEN BERITA ACARA UJI TERIMA KESATU",
        "CHECKLIST VERIFIKASI BA UJI TERIMA",
        "BERITA ACARA",
        "LAPORAN",
        "DAFTAR HADIR UJI TERIMA",
        "BOQ UJI TERIMA",
        "DOKUMENTASI UJI TERIMA",
        "FORM PENGUKURAN OPM",
        "PENGUKURAN OPM",
        "PENGUKURAN OTDR",
        "REPORT OTDR",
        "DOKUMENTASI PEKERJAAN",
        "AS BUILT DRAWING",
        "LAMPIRAN MANCORE",
        "LAMPIRAN KML"
    ]
}

def buat_id_sesi():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(uuid.uuid4())[:8]

def progress_reporter(tahap: str, current: int, total: int):
    sys.stdout.write(f"\r[{tahap}] Memproses... {current}/{total}   ")
    sys.stdout.flush()
    if current == total:
        print()

@app.post("/upload_and_validate", tags=["Proses Utama"])
async def upload_and_validate_multiple_pdfs(files: List[UploadFile] = File(...)):
    id_sesi = buat_id_sesi()
    path_sesi_output = OUTPUT_EKSTRAKSI_DIR / id_sesi
    
    print("\n" + "="*50)
    print(f"Memulai Sesi Baru: {id_sesi}")
    print(f"Menerima {len(files)} file untuk diproses.")
    print("="*50)

    laporan_sesi_keseluruhan = { "id_sesi": id_sesi, "proyek_yang_diproses": [], "hasil_validasi_kelengkapan": [], "total_gambar_diproses": 0, "total_duplikat_ditemukan": 0, "total_file_unik_baru": 0, "semua_detail_duplikat": [], "semua_error_log": [] }
    
    indeks_master = {}
    if PATH_MASTER_INDEX.exists():
        with open(PATH_MASTER_INDEX, "r", encoding="utf-8") as f:
            indeks_master = json.load(f)

    for idx, file in enumerate(files, 1):
        nama_proyek_folder = Path(file.filename).stem
        path_proyek_output = path_sesi_output / nama_proyek_folder
        print(f"\n--- Memproses Proyek {idx}/{len(files)}: {file.filename} ---")
        
        temp_pdf_path = INPUT_PDF_DIR / file.filename
        doc = None
        try:
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            laporan_proyek_final = {}

            #tahap 1: ekstraksi aset dasar
            print("[Tahap 1/5] Memulai ekstraksi aset dasar...")
            def ekstraksi_progress_reporter(current, total):
                progress_reporter("Tahap 1/5 - Ekstraksi Dasar", current, total)
            
            data_mentah = ekstrak_aset_terstruktur(str(temp_pdf_path), progress_callback=ekstraksi_progress_reporter)
            if not data_mentah: raise Exception("Ekstraksi dasar gagal.")
            hasil_ekstraksi = simpan_hasil_ke_disk(data_mentah, str(path_proyek_output))
            print("[Tahap 1/5] Ekstraksi dasar selesai.")

            # tahap 2: analisis kontekstual AI
            print("[Tahap 2/5] Memulai analisis kontekstual per halaman...")
            doc = fitz.open(temp_pdf_path)
            hasil_kontekstual_proyek = []
            total_halaman = len(doc)
            for page_num in range(total_halaman):
                progress_reporter("Tahap 2/5 - Analisis AI", page_num + 1, total_halaman)
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=200)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                hasil_analisis_halaman = analisis_halaman_dengan_layoutlmv3(image)
                hasil_kontekstual_proyek.append({ "halaman": page_num + 1, "analisis": hasil_analisis_halaman })

                gambar_visualisasi = visualisasikan_hasil_analisis(image, hasil_analisis_halaman)
                folder_halaman_output = path_proyek_output / f"halaman_{page_num + 1}"
                folder_halaman_output.mkdir(exist_ok=True)
                nama_file_debug = f"halaman_{page_num + 1}_analisis_visual.png"
                path_output_debug = folder_halaman_output / nama_file_debug
                gambar_visualisasi.save(path_output_debug)

            path_laporan_kontekstual = path_proyek_output / "laporan_kontekstual.json"
            with open(path_laporan_kontekstual, "w", encoding="utf-8") as f: json.dump(hasil_kontekstual_proyek, f, indent=4, ensure_ascii=False)
            print(f"[Tahap 2/5] Analisis kontekstual selesai. Visualisasi disimpan")

            # GANTIKAN BLOK LAMA ANDA DENGAN YANG INI

            # [Tahap 3] Rekonstruksi dengan LLM
            print("[Tahap 3/5] Mempersiapkan data dan memanggil LLM...")

            # Bagian ini mempersiapkan data untuk dikirim ke fungsi LLM
            entitas_per_halaman = {}
            for hasil_halaman in hasil_kontekstual_proyek:
                halaman = hasil_halaman.get('halaman')
                # Menggunakan fungsi pembantu dari validasi_konten untuk merapikan token
                entitas = _gabungkan_token_menjadi_entitas(hasil_halaman.get('analisis', {}).get('hasil_analisis_kontekstual', []))
                if entitas:
                    entitas_per_halaman[halaman] = entitas

            # Ambil Kunci API
            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

            if not GOOGLE_API_KEY:
                print("[PERINGATAN] Kunci API Google belum diatur! Melewati tahap LLM.")
                data_terstruktur = {"error": "API Key tidak diatur."}
            else:
                # Panggil fungsi LLM untuk menata ulang data
                data_terstruktur = tata_ulang_dengan_llm(entitas_per_halaman, GOOGLE_API_KEY)

            # Simpan hasil AKHIR ke file JSON dengan format vertikal (indent=4)
            # TIDAK ADA LAGI PRINT KE TERMINAL
            path_laporan_terstruktur = path_proyek_output / "laporan_data_terstruktur_llm.json"
            with open(path_laporan_terstruktur, "w", encoding="utf-8") as f:
                json.dump(data_terstruktur, f, indent=4, ensure_ascii=False)

            print(f"[Tahap 3/5] Data terstruktur dari LLM berhasil dibuat dan disimpan secara vertikal.")

            # Jangan lupa sesuaikan nomor tahap selanjutnya (misal: Tahap 4/5)
            # ...

            # Tahap baru validasi kelengkapan dokumen
            print("[Tahap 4/5] Memulai validasi kelengkapan dokumen...")
            hasil_validasi_kelengkapan = cek_kelengkapan_dokumen(hasil_kontekstual_proyek, ATURAN_KELENGKAPAN)
            laporan_proyek_final["validasi_kelengkapan"] = hasil_validasi_kelengkapan
            print(f"[Tahap 4/5] Validasi kelengkapan selesai. Status: {hasil_validasi_kelengkapan['status']}")

            # Tahap 4: validasi dupplikasi foto
            print("[Tahap 5/5] Memulai validasi duplikasi foto...")
            list_gambar_absolut = []
            for ext in EKSTENSI_GAMBAR: list_gambar_absolut.extend(glob.glob(str(path_proyek_output / '**' / f'*.{ext}'), recursive=True))
            print(f"Ditemukan {len(list_gambar_absolut)} gambar untuk divalidasi.")
            
            def validasi_progress_reporter(current, total):
                progress_reporter("Tahap 5/5 - Validasi Foto", current, total)

            hasil_validasi_foto = proses_validasi_dengan_petunjuk( list_gambar_proyek=list_gambar_absolut, indeks_master=indeks_master, nama_proyek=file.filename, path_sesi=str(path_sesi_output), progress_callback=validasi_progress_reporter)
            laporan_proyek_final["validasi_duplikasi_foto"] = hasil_validasi_foto
            print(f"[Tahap 5/5] Validasi selesai. Duplikat: {hasil_validasi_foto.get('duplikat_ditemukan', 0)}")
            
            path_laporan_proyek = path_proyek_output / "laporan_validasi_proyek.json"
            with open(path_laporan_proyek, "w", encoding="utf-8") as f: json.dump(laporan_proyek_final, f, indent=4, ensure_ascii=False)
            
            laporan_sesi_keseluruhan["proyek_yang_diproses"].append({"nama_file": file.filename, "status_kelengkapan": hasil_validasi_kelengkapan['status']})
            laporan_sesi_keseluruhan["total_gambar_diproses"] += hasil_validasi_foto["jumlah_gambar_diproses"]
            laporan_sesi_keseluruhan["total_duplikat_ditemukan"] += hasil_validasi_foto["duplikat_ditemukan"]
            laporan_sesi_keseluruhan["total_file_unik_baru"] += hasil_validasi_foto["file_unik_baru_dicatat"]
            laporan_sesi_keseluruhan["semua_detail_duplikat"].extend(hasil_validasi_foto["detail_duplikat"])
            laporan_sesi_keseluruhan["semua_error_log"].extend(hasil_validasi_foto["error_log"])

        except Exception as e:
            print(f"\n[ERROR] Gagal memproses {file.filename}: {e}")
            continue
        finally:
            if doc:
                doc.close()
            if temp_pdf_path.exists():
                os.remove(temp_pdf_path)

    path_laporan_sesi = path_sesi_output / "laporan_sesi_keseluruhan.json"
    with open(path_laporan_sesi, "w", encoding="utf-8") as f:
        json.dump(laporan_sesi_keseluruhan, f, indent=4, ensure_ascii=False)
    print(f"\nLaporan ringkasan sesi disimpan di: {path_laporan_sesi}")
    
    with open(PATH_MASTER_INDEX, "w", encoding="utf-8") as f:
        json.dump(indeks_master, f, indent=4, ensure_ascii=False)
    print("Indeks Master berhasil diperbarui.")
    
    print("="*50)
    print("Sesi keseluruhan selesai.")
    print("="*50 + "\n")

    return JSONResponse(status_code=200, content=laporan_sesi_keseluruhan)

@app.get("/", tags=["Status"])
async def root():
    return {"message": "Selamat Datang di API Sistem Validasi Laporan."}