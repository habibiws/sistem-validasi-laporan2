# backend/validasi_konten.py
# Versi dengan metode pencarian "tanpa spasi" untuk mengatasi tokenization

import re
import json
from typing import List, Dict, Any
from json_repair import repair_json

from backend.konteks_extractor import get_indobert_model_and_tokenizer

ATURAN_VALIDASI = {
    "BAUT": {
        "nama_dokumen": "Berita Acara Uji Terima",
        "field_wajib": [
            "PROYEK",
            "KONTRAK",
            "WITEL",
            "DISTRICT",
            "LOKASI",
            "PELAKSANA",
            "NO_BAUT",
            "TANGGAL",
            "SP",
            "S_PERMOHONAN"

            # Tambahkan field lain yang spesifik untuk BAUT
        ],
        # Kata kunci untuk membantu identifikasi otomatis di masa depan
        "frasa_kunci_identifikasi": ["UJI TERIMA", "BAUT"] 
    },
    "BACT": {
        "nama_dokumen": "Berita Acara Commissioning Test",
        "field_wajib": [
            "PROYEK",
            "KONTRAK",
            "WITEL",
            "DISTRICT",
            "LOKASI",
            "PELAKSANA",
            "TANGGAL",
            "HARI",
            "BULAN",
            "TAHUN"
            # Tambahkan field lain yang spesifik untuk BACT
        ],
        "frasa_kunci_identifikasi": ["COMMISSIONING TEST", "BACT", "BATC"]
    },
    "UMUM": {
        "nama_dokumen": "Dokumen Umum",
        "field_wajib": [
            "PROYEK" # Mungkin hanya proyek yang wajib untuk dokumen tak dikenal
        ]
    }
    }

def cek_kelengkapan_dokumen(laporan_kontekstual: List[Dict[str, Any]], aturan_kelengkapan: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Memvalidasi kelengkapan dokumen berdasarkan keberadaan frasa wajib
    menggunakan metode pencarian 'tanpa spasi' untuk mengatasi tokenization.
    """
    
    frasa_wajib = aturan_kelengkapan.get('frasa_wajib', [])
    if not frasa_wajib:
        return {"status": "DILEWATI", "message": "Tidak ada aturan frasa wajib yang didefinisikan."}

    # Gabungkan semua token dari semua halaman menjadi satu string teks besar
    teks_dokumen_lengkap = ""
    for halaman in laporan_kontekstual:
        analisis_halaman = halaman.get('analisis', {})
        hasil_analisis = analisis_halaman.get('hasil_analisis_kontekstual', [])
        for item in hasil_analisis:
            # Menggunakan 'token' yang sudah di-lowercase dari model jika ada, atau handle token dari tokenizer
            # Untuk LayoutLMV3, token biasanya sudah bersih
            token = item.get('token', '')
            # Menghilangkan karakter khusus awal kata dari beberapa tokenizer (seperti ' ')
            if token.startswith(' '):
                token = token[1:]
            teks_dokumen_lengkap += token

    # --- PERBAIKAN LOGIKA DI SINI ---
    # Normalisasi teks ke huruf kecil dan hapus SEMUA spasi dan karakter non-alfanumerik
    teks_dokumen_normal = re.sub(r'[^a-z0-9]', '', teks_dokumen_lengkap.lower())

    frasa_ditemukan = []
    frasa_tidak_ditemukan = []

    for frasa in frasa_wajib:
        # Normalisasi frasa yang dicari dengan cara yang sama
        frasa_normal = re.sub(r'[^a-z0-9]', '', frasa.lower())
        
        if frasa_normal in teks_dokumen_normal:
            frasa_ditemukan.append(frasa)
        else:
            frasa_tidak_ditemukan.append(frasa)

    status = "LENGKAP" if not frasa_tidak_ditemukan else "TIDAK LENGKAP"

    return {
        "status": status,
        "frasa_ditemukan": frasa_ditemukan,
        "frasa_tidak_ditemukan": frasa_tidak_ditemukan
    }

# Di dalam backend/validasi_konten.py

def _gabungkan_token_menjadi_entitas(hasil_analisis_kontekstual: list) -> list:
    """
    Fungsi yang sudah diperbaiki untuk menggabungkan token menjadi entitas utuh.
    Fungsi ini sekarang menerima daftar token secara langsung.
    """
    if not hasil_analisis_kontekstual:
        return []

    entitas_dict = {}
    # Langsung iterasi ke input karena sudah berupa list of dictionaries
    for token_data in hasil_analisis_kontekstual:
        # Kunci unik adalah gabungan dari koordinat box dan label
        key = (tuple(token_data["box"]), token_data["label"])
        
        if key not in entitas_dict:
            entitas_dict[key] = {
                "tokens": [],
                "halaman_asal": token_data.get('halaman_asal', 1) # Simpan halaman jika ada
            }
        
        entitas_dict[key]["tokens"].append(token_data["token"])

    entitas_final = []
    for (box, label), data in entitas_dict.items():
        teks_lengkap = "".join(data["tokens"]).replace("Ġ", " ").strip()
        
        if teks_lengkap:
            entitas_final.append({
                "text": teks_lengkap,
                "box": list(box),
                "label": label,
                "halaman_asal": data["halaman_asal"]
            })
            
    # Urutkan entitas berdasarkan posisi Y, lalu posisi X
    entitas_final.sort(key=lambda e: (e["box"][1], e["box"][0]))
    return entitas_final

def tata_ulang_dengan_indobert_lokal(final_entities: list) -> dict:
    
    """
    Menggunakan model IndoBERT lokal untuk menata ulang entitas mentah menjadi JSON terstruktur.
    'final_entities' adalah list hasil olahan dari LayoutLMv3.
    """
    if not final_entities:
        return {"error": "Tidak ada entitas yang diterima untuk diproses."}

    # 1. Buat 'input_text' dari final_entities
    # Logika ini sama persis dengan yang kita gunakan di create_bert_dataset.py
    input_lines = []
    for entity in final_entities:
        line = f"teks: \"{entity['text']}\" box: {entity['box']}"
        input_lines.append(line)
    input_text = "\n".join(input_lines)

    # 2. Panggil getter untuk mendapatkan model dan tokenizer yang sudah pasti terisi
    brain_model, brain_tokenizer = get_indobert_model_and_tokenizer()

    # 3. Lakukan Tokenisasi (gunakan variabel lokal)
    inputs = brain_tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # --- PENJAGA FINAL DITAMBAHKAN DI SINI ---
    # Periksa apakah hasil tokenisasi menghasilkan input yang valid.
    # tensor.nelement() menghitung jumlah total elemen dalam tensor.
    if inputs.input_ids.nelement() == 0:
        print("[PERINGATAN] Tokenizer menghasilkan input kosong, tidak dapat menjalankan model 'Otak'.")
        return {"error": "Tokenizer gagal menghasilkan token yang valid dari teks input."}
    # --- AKHIR DARI PENJAGA ---

    # 4. Jalankan Inferensi (gunakan variabel lokal)
    output_ids = brain_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=512,
        num_beams=4,
        early_stopping=True,
        decoder_start_token_id=brain_tokenizer.cls_token_id
    )

    # 5. Decode Hasilnya (gunakan variabel lokal)
    predicted_json_string = brain_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 5. Coba parse dan perbaiki JSON jika perlu
    try:
        # Langsung coba parse
        return json.loads(predicted_json_string)
    except json.JSONDecodeError:
        print("[PERINGATAN] Output model bukan JSON valid. Mencoba memperbaiki...")
        try:
            # Jika gagal, coba perbaiki dengan json_repair
            repaired_json = repair_json(predicted_json_string)
            return json.loads(repaired_json)
        except Exception as e:
            print(f"[ERROR] Gagal memperbaiki JSON. Error: {e}")
            return {"error": "Gagal menghasilkan JSON yang valid dari model.", "raw_output": predicted_json_string}
        
def cek_validitas_isian_data(data_terstruktur: dict, tipe_dokumen: str) -> dict:
    """
    Memvalidasi apakah field-field wajib dalam data terstruktur sudah diisi.
    """


    aturan = ATURAN_VALIDASI.get(tipe_dokumen, ATURAN_VALIDASI["UMUM"])
    FIELD_WAJIB = aturan["field_wajib"]

    # Periksa jika inputnya adalah dictionary error dari langkah sebelumnya
    if "error" in data_terstruktur:
        return {
            "status": "GAGAL",
            "message": "Tidak dapat melakukan validasi karena data terstruktur gagal dibuat.",
            "detail_error": data_terstruktur.get("error")
        }

    field_terisi = []
    field_kosong = []

    for field in FIELD_WAJIB:
        # Cek apakah field ada di data dan nilainya tidak kosong/hanya spasi
        if field in data_terstruktur and str(data_terstruktur[field]).strip():
            field_terisi.append(field)
        else:
            field_kosong.append(field)

    # Tentukan status akhir berdasarkan apakah ada field yang kosong
    status_akhir = "LENGKAP" if not field_kosong else "TIDAK LENGKAP"

    return {
        "status": status_akhir,
        "field_wajib": FIELD_WAJIB,
        "field_terisi": field_terisi,
        "field_kosong": field_kosong
    }

def deteksi_tipe_dokumen_dari_hasil_ai(data_terstruktur: dict, nama_file_pdf: str) -> str:
    """
    Mendeteksi tipe dokumen (BAUT/BACT) dengan prioritas pada hasil ekstraksi AI,
    dan menggunakan nama file sebagai cadangan.
    """
    # --- STRATEGI 1: Cari di Hasil Ekstraksi AI (Paling Akurat) ---
    # Kita cari field yang kemungkinan berisi judul, misal: 'SECTION', 'Tipe_Dokumen', 'JUDUL'
    # (Sesuaikan nama field ini dengan output JSON Impian kita)
    
    # Gabungkan semua nilai teks yang mungkin relevan dari hasil AI
    teks_untuk_diperiksa = ""
    if "SECTION" in data_terstruktur:
        teks_untuk_diperiksa += " " + str(data_terstruktur["SECTION"]).upper()
    if "Tipe_Dokumen" in data_terstruktur:
        teks_untuk_diperiksa += " " + str(data_terstruktur["Tipe_Dokumen"]).upper()
    
    # Cek frasa kunci di teks yang diekstrak
    if "COMMISSIONING TEST" in teks_untuk_diperiksa or "BACT" in teks_untuk_diperiksa or "BATC" in teks_untuk_diperiksa:
        return "BACT"
    if "UJI TERIMA" in teks_untuk_diperiksa or "BAUT" in teks_untuk_diperiksa:
        return "BAUT"

    # --- STRATEGI 2: Cek Nama File (Cadangan) ---
    nama_file_upper = nama_file_pdf.upper()
    if "BACT" in nama_file_upper or "BATC" in nama_file_upper:
        return "BACT"
    if "BAUT" in nama_file_upper:
        return "BAUT"
        
    # --- Fallback Terakhir ---
    return "UMUM"