# backend/validasi_foto.py
import os
import json
import re
from typing import Dict, List, Any, Callable
from PIL import Image
import pytesseract

def bersihkan_teks(teks_mentah: str) -> str:
    if not teks_mentah: return ""
    teks_bersih = re.sub(r'\s+', ' ', teks_mentah.strip())
    teks_bersih = re.sub(r'[^\w\s\-:.,/|°]', '', teks_bersih)
    teks_bersih = re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\1-\2-\3', teks_bersih)
    return teks_bersih.strip()

def ekstrak_metadata_gambar(path_gambar: str) -> str:
    try:
        with Image.open(path_gambar) as img:
            img_gray = img.convert('L')
            img_processed = img_gray.point(lambda x: 0 if x < 128 else 255, '1')
            config_ocr = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,:-/|°'
            teks_mentah = pytesseract.image_to_string(img_processed, config=config_ocr)
            return bersihkan_teks(teks_mentah)
    except FileNotFoundError:
        raise FileNotFoundError(f"File gambar tidak ditemukan: {path_gambar}")
    except Exception as e:
        raise Exception(f"Error saat memproses gambar {path_gambar}: {str(e)}")

def proses_validasi_dengan_petunjuk(
    list_gambar_proyek: List[str], 
    indeks_master: Dict[str, Any], 
    nama_proyek: str, 
    path_sesi: str,
    progress_callback: Callable[[int, int], None] = None
) -> Dict[str, Any]:
    detail_duplikat, error_log = [], []
    jumlah_berhasil_diproses, file_unik_baru = 0, 0
    total_gambar = len(list_gambar_proyek)

    if total_gambar == 0:
        return { "status": "dilewati", "message": "Tidak ada gambar untuk divalidasi.", "jumlah_gambar_diproses": 0, "duplikat_ditemukan": 0, "file_unik_baru_dicatat": 0 }

    for i, path_gambar_input in enumerate(list_gambar_proyek, 1):
        if progress_callback:
            progress_callback(i, total_gambar)
            
        try:
            metadata_teks = ekstrak_metadata_gambar(path_gambar_input)
            if not metadata_teks or len(metadata_teks.strip()) < 5:
                jumlah_berhasil_diproses += 1; continue
            
            if metadata_teks in indeks_master:
                path_relatif_duplikat = os.path.relpath(path_gambar_input, path_sesi)
                duplikat_info = { "duplikat_ditemukan": path_relatif_duplikat.replace("\\", "/"), "duplikat_dari_petunjuk": indeks_master[metadata_teks] }
                detail_duplikat.append(duplikat_info)
            else:
                path_relatif_file = os.path.relpath(path_gambar_input, path_sesi)
                petunjuk_baru = { "sesi_asli": os.path.basename(path_sesi), "proyek_asli": nama_proyek, "path_relatif_di_sesi": path_relatif_file.replace("\\", "/") }
                indeks_master[metadata_teks] = petunjuk_baru
                file_unik_baru += 1
            
            jumlah_berhasil_diproses += 1
        except Exception as e:
            error_log.append(f"Error pada file {os.path.basename(path_gambar_input)}: {e}")
            
    return { "status": "selesai", "jumlah_gambar_diproses": total_gambar, "berhasil_diproses": jumlah_berhasil_diproses, "duplikat_ditemukan": len(detail_duplikat), "file_unik_baru_dicatat": file_unik_baru, "detail_duplikat": detail_duplikat, "error_log": error_log }