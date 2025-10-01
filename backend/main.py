# backend/main.py (Versi FINAL dengan Arsitektur Self-Calling API)

import os
import shutil
import json
import glob
import sys
import uuid
import httpx  # Library untuk melakukan HTTP request
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Impor dari file lain dalam proyek ---
# Impor HANYA untuk tugas-tugas dasar, BUKAN untuk AI.
from backend.ekstraksi_pdf import ekstrak_aset_terstruktur, simpan_hasil_ke_disk
from backend.validasi_foto import proses_validasi_dengan_petunjuk

# Impor BARU dari engine AI kita yang terpisah.
# Ini akan dipanggil melalui endpoint, jadi kita tidak perlu mengimpor semuanya.
from backend.ai_engine import run_ai_pipeline

# --- Konfigurasi Aplikasi FastAPI ---
app = FastAPI(
    title="Sistem Validasi Laporan Otomatis",
    version="4.0.0-self-calling-api",
    description="API dengan arsitektur endpoint AI internal untuk stabilitas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua origin
    allow_credentials=True,
    allow_methods=["*"],  # Izinkan semua metode (GET, POST, dll)
    allow_headers=["*"],  # Izinkan semua header
)

# --- Pengaturan Path & Variabel Global ---
# Menggunakan path relatif agar lebih portabel
DATA_DIR = Path("data")
INPUT_PDF_DIR = DATA_DIR / "input_pdf"
OUTPUT_EKSTRAKSI_DIR = DATA_DIR / "output_ekstraksi"
SISTEM_VALIDASI_DIR = DATA_DIR / "sistem_validasi"
PATH_MASTER_INDEX = SISTEM_VALIDASI_DIR / "master_index.json"

# Buat direktori jika belum ada
INPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_EKSTRAKSI_DIR.mkdir(parents=True, exist_ok=True)
SISTEM_VALIDASI_DIR.mkdir(parents=True, exist_ok=True)

EKSTENSI_GAMBAR = ["jpg", "jpeg", "png", "bmp"]

# --- Model Data untuk Endpoint AI ---
# Ini mendefinisikan "paket data" yang akan dikirim ke endpoint AI internal
class AIRequest(BaseModel):
    pdf_path: str
    original_filename: str

# --- Fungsi Helper ---
def buat_id_sesi():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(uuid.uuid4())[:8]

# --- Endpoint-Endpoint API ---

@app.get("/", tags=["Status"])
async def root():
    """Endpoint dasar untuk memeriksa apakah server berjalan."""
    return {"message": "Selamat Datang di API Sistem Validasi Laporan (Arsitektur Self-Calling)."}

@app.post("/internal/run_ai", tags=["Internal"], include_in_schema=False)
async def run_ai_endpoint(request: AIRequest):
    """
    Endpoint INTERNAL yang didedikasikan untuk menjalankan pipeline AI yang berat.
    Tidak untuk diakses langsung oleh frontend.
    """
    try:
        # Panggil fungsi utama dari ai_engine.py
        hasil_ai = run_ai_pipeline(request.pdf_path, request.original_filename)
        return JSONResponse(status_code=200, content=hasil_ai)
    except Exception as e:
        # Jika terjadi error di dalam pipeline AI, laporkan dengan detail
        print(f"[ERROR di AI Endpoint] {e}")
        raise HTTPException(status_code=500, detail=f"AI pipeline failed: {str(e)}")

@app.post("/upload_and_validate", tags=["Proses Utama"])
async def upload_and_validate_multiple_pdfs(request: Request, files: List[UploadFile] = File(...)):
    """
    Endpoint utama yang menerima unggahan file PDF, mengorkestrasi seluruh alur kerja,
    dan mengembalikan laporan validasi akhir.
    """
    id_sesi = buat_id_sesi()
    path_sesi_output = OUTPUT_EKSTRAKSI_DIR / id_sesi
    base_url = str(request.base_url)  # Dapatkan base URL server (misal: http://127.0.0.1:8000/)
    
    print("\n" + "="*50)
    print(f"Memulai Sesi Baru: {id_sesi}")
    print(f"Menerima {len(files)} file untuk diproses.")
    print("="*50)

    laporan_sesi_keseluruhan = {"id_sesi": id_sesi, "proyek_yang_diproses": []}
    indeks_master = {}
    if PATH_MASTER_INDEX.exists():
        with open(PATH_MASTER_INDEX, "r", encoding="utf-8") as f:
            indeks_master = json.load(f)

    for idx, file in enumerate(files, 1):
        nama_proyek_folder = Path(file.filename).stem
        path_proyek_output = path_sesi_output / nama_proyek_folder
        temp_pdf_path = INPUT_PDF_DIR / f"{id_sesi}_{file.filename}"

        print(f"\n--- Memproses Proyek {idx}/{len(files)}: {file.filename} ---")
        
        try:
            # Simpan file yang diunggah sementara
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            laporan_proyek_final = {}

            # [Tahap 1/3] Ekstraksi Aset Dasar (Teks & Gambar)
            print("[Tahap 1/3] Memulai ekstraksi aset dasar...")
            data_mentah = ekstrak_aset_terstruktur(str(temp_pdf_path))
            if not data_mentah: raise Exception("Ekstraksi aset dasar gagal.")
            hasil_ekstraksi = simpan_hasil_ke_disk(data_mentah, str(path_proyek_output))
            laporan_proyek_final["hasil_ekstraksi_dasar"] = hasil_ekstraksi
            print("[Tahap 1/3] Ekstraksi aset dasar selesai.")

            # [Tahap 2/3] Memanggil Endpoint AI Internal
            print("[Tahap 2/3] Memanggil endpoint AI internal untuk proses berat...")
            ai_payload = {"pdf_path": str(temp_pdf_path.resolve()), "original_filename": file.filename}
            
            # Gunakan httpx untuk memanggil endpoint /internal/run_ai di server yang sama
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(f"{base_url}internal/run_ai", json=ai_payload)
                response.raise_for_status()  # Ini akan error jika status code bukan 2xx
                hasil_ai = response.json()
            
            laporan_proyek_final["hasil_pemrosesan_ai"] = hasil_ai
            path_hasil_ai = path_proyek_output / "hasil_ai_endpoint.json"
            with open(path_hasil_ai, "w", encoding="utf-8") as f:
                json.dump(hasil_ai, f, indent=4, ensure_ascii=False)
            print("[Tahap 2/3] Endpoint AI selesai.")

            # [Tahap 3/3] Validasi Duplikasi Foto
            print("[Tahap 3/3] Memulai validasi duplikasi foto...")
            list_gambar_absolut = [str(Path(p["path"]).resolve()) for h in hasil_ekstraksi.get("hasil_per_halaman", []) for p in h.get("path_gambar", [])]
            
            hasil_validasi_foto = proses_validasi_dengan_petunjuk(
                list_gambar_proyek=list_gambar_absolut,
                indeks_master=indeks_master,
                nama_proyek=file.filename,
                path_sesi=str(path_sesi_output)
            )
            laporan_proyek_final["validasi_duplikasi_foto"] = hasil_validasi_foto
            print(f"[Tahap 3/3] Validasi foto selesai. Duplikat: {hasil_validasi_foto.get('duplikat_ditemukan', 0)}")
            
            # Simpan laporan final untuk proyek ini
            path_laporan_proyek = path_proyek_output / "laporan_validasi_proyek.json"
            with open(path_laporan_proyek, "w", encoding="utf-8") as f:
                json.dump(laporan_proyek_final, f, indent=4, ensure_ascii=False)
            
            laporan_sesi_keseluruhan["proyek_yang_diproses"].append({
                "nama_file": file.filename,
                "status_validasi_isian": hasil_ai.get("validasi_isian_data", {}).get("status", "GAGAL_AI")
            })

        except httpx.HTTPStatusError as e:
            print(f"\n[ERROR FATAL] Panggilan ke endpoint AI internal gagal: {e.response.status_code}")
            print(f"   - Detail: {e.response.text}")
            laporan_sesi_keseluruhan["proyek_yang_diproses"].append({"nama_file": file.filename, "status_validasi_isian": "ERROR_ENDPOINT_AI"})
            continue
        except Exception as e:
            print(f"\n[ERROR] Gagal memproses {file.filename}: {e}")
            laporan_sesi_keseluruhan["proyek_yang_diproses"].append({"nama_file": file.filename, "status_validasi_isian": f"ERROR_{type(e).__name__}"})
            continue
        finally:
            if temp_pdf_path.exists():
                os.remove(temp_pdf_path)

    # Simpan laporan sesi keseluruhan dan perbarui indeks master
    path_laporan_sesi = path_sesi_output / "laporan_sesi_keseluruhan.json"
    with open(path_laporan_sesi, "w", encoding="utf-8") as f:
        json.dump(laporan_sesi_keseluruhan, f, indent=4, ensure_ascii=False)
    
    with open(PATH_MASTER_INDEX, "w", encoding="utf-8") as f:
        json.dump(indeks_master, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("Sesi keseluruhan selesai.")
    print("="*50 + "\n")

    return JSONResponse(status_code=200, content=laporan_sesi_keseluruhan)