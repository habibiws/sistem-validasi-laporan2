# backend/ekstraksi_pdf.py

import os
import json
import uuid
import re
from pathlib import Path
from datetime import datetime
import fitz
from typing import Callable

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def ekstrak_aset_terstruktur(
    path_pdf: str, 
    progress_callback: Callable[[int, int], None] = None,
    **opsi_filter
) -> dict | None:
    try:
        doc = fitz.open(path_pdf)
        total_halaman = len(doc)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_id = str(uuid.uuid4()).split('-')[0]
        hasil_in_memory = { "id_proses": f"{timestamp}-{unique_id}", "sumber_pdf": os.path.basename(path_pdf), "hasil_per_halaman": [] }
        
        for page_num in range(total_halaman):
            if progress_callback:
                progress_callback(page_num + 1, total_halaman)

            page = doc.load_page(page_num)
            halaman_ke = page_num + 1
            
            # Langkah 1: Selalu coba ekstrak teks digital dan objek gambar
            page_text = page.get_text("text")
            image_list = page.get_images(full=True)
            
            metode_ekstraksi = "Bawaan"
            
            # Langkah 2: Periksa apakah teks kosong, jika ya, lakukan OCR
            if not page_text.strip() and OCR_AVAILABLE:
                metode_ekstraksi = "OCR"
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img, lang='ind+eng')
                except Exception:
                    metode_ekstraksi = "Gagal (Error OCR)"

            hasil_halaman = { "halaman": halaman_ke, "konten_teks": page_text, "konten_gambar": [], "metode_ekstraksi": metode_ekstraksi }

            # Proses objek gambar yang sudah diekstrak
            for img_info in image_list:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                hasil_halaman["konten_gambar"].append({ "ext": base_image["ext"], "data": base_image["image"], "width": base_image["width"], "height": base_image["height"] })
            
            hasil_in_memory["hasil_per_halaman"].append(hasil_halaman)
        
        doc.close()
        return hasil_in_memory
        
    except Exception as e:
        print(f"\n[ERROR] Terjadi error saat ekstraksi: {e}")
        return None

def simpan_hasil_ke_disk(data_ekstraksi: dict, path_proyek: str) -> dict:
    id_proses_file = data_ekstraksi["id_proses"]
    os.makedirs(path_proyek, exist_ok=True)
    
    hasil_dengan_path = { "id_proses": id_proses_file, "sumber_pdf": data_ekstraksi["sumber_pdf"], "hasil_per_halaman": [] }
    
    for data_halaman in data_ekstraksi["hasil_per_halaman"]:
        halaman_ke = data_halaman["halaman"]
        folder_halaman = os.path.join(path_proyek, f"halaman_{halaman_ke}")
        os.makedirs(folder_halaman, exist_ok=True)
        
        path_halaman = { "halaman": halaman_ke, "path_teks": None, "path_gambar": [], "metode_ekstraksi": data_halaman["metode_ekstraksi"], "jumlah_gambar": len(data_halaman["konten_gambar"]) }
        
        if data_halaman["konten_teks"].strip():
            path_teks_output = os.path.join(folder_halaman, "teks.txt")
            with open(path_teks_output, "w", encoding="utf-8") as f:
                f.write(data_halaman["konten_teks"])
            path_halaman["path_teks"] = os.path.relpath(path_teks_output, Path(path_proyek).parent).replace("\\", "/")
        
        for idx, gambar in enumerate(data_halaman["konten_gambar"]):
            nama_file_gambar = f"img_{idx}.{gambar['ext']}"
            path_gambar_output = os.path.join(folder_halaman, nama_file_gambar)
            with open(path_gambar_output, "wb") as f:
                f.write(gambar['data'])
            path_halaman["path_gambar"].append({ "path": os.path.relpath(path_gambar_output, Path(path_proyek).parent).replace("\\", "/"), "width": gambar["width"], "height": gambar["height"], "format": gambar["ext"] })
        
        hasil_dengan_path["hasil_per_halaman"].append(path_halaman)
    
    path_file_summary = os.path.join(path_proyek, "_summary.json")
    with open(path_file_summary, "w", encoding="utf-8") as f:
        json.dump(hasil_dengan_path, f, indent=4)
        
    return hasil_dengan_path