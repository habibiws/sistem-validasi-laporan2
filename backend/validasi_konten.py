# backend/validasi_konten.py
# Versi dengan metode pencarian "tanpa spasi" untuk mengatasi tokenization

import re
import json
import google.generativeai as genai
from typing import List, Dict, Any
from json_repair import repair_json

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

# Di dalam file backend/validasi_konten.py

def _gabungkan_token_menjadi_entitas(hasil_analisis: list) -> list:
    """Fungsi ini tidak berubah, hanya memastikan outputnya urut."""
    if not hasil_analisis:
        return []
    entitas_dict = {}
    for token_data in hasil_analisis:
        key = (tuple(token_data["box"]), token_data["label"])
        if key not in entitas_dict:
            entitas_dict[key] = {"tokens": []}
        entitas_dict[key]["tokens"].append(token_data["token"])

    entitas_final = []
    for (box, label), data in entitas_dict.items():
        teks_lengkap = "".join(data["tokens"]).replace("Ġ", " ").strip()
        # Simpan juga informasi halaman asal
        halaman_asal = 1 # Default
        for token_data in hasil_analisis:
            if tuple(token_data["box"]) == key[0] and token_data["label"] == key[1]:
                halaman_asal = token_data.get('halaman_asal')
                break
        
        if teks_lengkap:
            entitas_final.append({
                "text": teks_lengkap,
                "box": list(box),
                "label": label,
                "halaman_asal": halaman_asal
            })
    # Urutkan entitas berdasarkan halaman, lalu posisi Y, lalu posisi X
    entitas_final.sort(key=lambda e: (e.get('halaman_asal', 1), e["box"][1], e["box"][0]))
    return entitas_final

# Di dalam file backend/validasi_konten.py

def rekonstruksi_key_value(semua_hasil_analisis_dokumen: list) -> dict:
    """
    Versi Hibrida: Menggabungkan pemasangan berbasis baris (horizontal)
    dan berbasis kolom (vertikal) untuk akurasi maksimal.
    """
    print("   - Memulai rekonstruksi Key-Value (versi hibrida)...")
    entitas = _gabungkan_token_menjadi_entitas(semua_hasil_analisis_dokumen)
    
    hasil_per_halaman = {}
    if not entitas:
        return {}

    # Kelompokkan entitas berdasarkan halaman
    entitas_per_halaman = {}
    for ent in entitas:
        halaman = ent.get('halaman_asal', 1)
        if halaman not in entitas_per_halaman:
            entitas_per_halaman[halaman] = []
        entitas_per_halaman[halaman].append(ent)

    # Proses halaman per halaman
    for halaman, entitas_di_halaman in entitas_per_halaman.items():
        pasangan_kv = {}
        # Tandai entitas yang sudah dipasangkan
        sudah_dipasangkan = set()

        # --- TAHAP 1: PEMASANGAN HORIZONTAL (KASUS MUDAH) ---
        lines = []
        if entitas_di_halaman:
            # Kelompokkan entitas menjadi baris-baris
            entitas_di_halaman.sort(key=lambda e: (e['box'][1], e['box'][0]))
            current_line = [entitas_di_halaman[0]]
            for i in range(1, len(entitas_di_halaman)):
                prev_ent = current_line[-1]
                curr_ent = entitas_di_halaman[i]
                if abs(((prev_ent['box'][1] + prev_ent['box'][3]) / 2) - ((curr_ent['box'][1] + curr_ent['box'][3]) / 2)) < 10:
                    current_line.append(curr_ent)
                else:
                    lines.append(sorted(current_line, key=lambda e: e['box'][0]))
                    current_line = [curr_ent]
            lines.append(sorted(current_line, key=lambda e: e['box'][0]))

        # Proses setiap baris untuk menemukan pasangan kiri-kanan
        for i, line in enumerate(lines):
            keys_in_line = [(idx, e) for idx, e in enumerate(line) if e['label'].endswith(("_KEY", "_HEADER"))]
            values_in_line = [(idx, e) for idx, e in enumerate(line) if e['label'].endswith("_VALUE")]

            if keys_in_line and values_in_line:
                key_text = " ".join(k[1]['text'] for k in keys_in_line).replace(":", "").strip()
                value_text = " ".join(v[1]['text'] for v in values_in_line).strip()
                if key_text and value_text:
                    pasangan_kv[key_text] = value_text
                    # Tandai semua entitas di baris ini sebagai sudah dipasangkan
                    for ent in line:
                        sudah_dipasangkan.add(tuple(ent['box']))
        
        # --- TAHAP 2: PEMASANGAN VERTIKAL (KASUS SULIT SEPERTI TANDA TANGAN) ---
        keys_tersisa = [e for e in entitas_di_halaman if e['label'].endswith(("_KEY", "_HEADER")) and tuple(e['box']) not in sudah_dipasangkan]
        values_tersisa = [e for e in entitas_di_halaman if e['label'].endswith("_VALUE") and tuple(e['box']) not in sudah_dipasangkan]

        for key_ent in keys_tersisa:
            key_box = key_ent['box']
            key_text = key_ent['text'].replace(":", "").strip()
            if not key_text: continue

            kandidat_terbaik = None
            jarak_terdekat = float('inf')

            for val_ent in values_tersisa:
                val_box = val_ent['box']
                # Cari value yang ada DI BAWAH key
                if val_box[1] > key_box[3]:
                    # Cek apakah mereka tumpang tindih secara horizontal (berada di kolom yang sama)
                    overlap_x = max(0, min(key_box[2], val_box[2]) - max(key_box[0], val_box[0]))
                    if overlap_x > 0: # Jika ada tumpang tindih di sumbu X
                        jarak = val_box[1] - key_box[3] # Jarak vertikal
                        if jarak < jarak_terdekat:
                            jarak_terdekat = jarak
                            kandidat_terbaik = val_ent
            
            if kandidat_terbaik:
                pasangan_kv[key_text] = kandidat_terbaik['text']
                # Tandai keduanya agar tidak digunakan lagi
                sudah_dipasangkan.add(tuple(key_ent['box']))
                sudah_dipasangkan.add(tuple(kandidat_terbaik['box']))
        
        # Simpan hasil untuk halaman ini jika ada
        if pasangan_kv:
            hasil_per_halaman[str(halaman)] = pasangan_kv

    print(f"   - Rekonstruksi selesai.")
    return hasil_per_halaman

# Di file: backend/validasi_konten.py

def tata_ulang_dengan_llm(entitas_per_halaman: dict, api_key: str) -> dict:
    """
    Menggunakan LLM untuk membersihkan dan menata ulang hasil ekstraksi mentah dari LayoutLM.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt_input_text = "Data Ekstraksi Mentah:\n"
    for page, entities in entitas_per_halaman.items():
        prompt_input_text += f"\n--- Halaman {page} ---\n"
        for ent in entities:
            prompt_input_text += f"- Teks: '{ent['text']}', Label: {ent['label']}\n"
    
    instruction_prompt = f"""
    Anda adalah AI ahli yang tugasnya membersihkan data ekstraksi dari dokumen.
    Di bawah ini adalah daftar teks dan label yang diekstrak dari beberapa halaman dokumen.
    Tugas Anda adalah menata ulang data ini menjadi format JSON yang bersih dan logis.

    Aturan:
    1. Kelompokkan hasil per halaman. Kunci utama adalah nomor halaman dalam bentuk string.
    2. Pasangkan KEY dan VALUE yang paling relevan. Jangan membuat kunci duplikat.
    3. Abaikan entitas yang tidak relevan atau aneh.
    4. Kembalikan HANYA JSON hasil akhir tanpa teks penjelasan atau markdown.

    {prompt_input_text}

    JSON Hasil Akhir:
    """
    
    # Menghapus semua print() yang tidak perlu dari sini
    
    json_text_untuk_debug = "" 
    try:
        response = model.generate_content(instruction_prompt)
        response_text = response.text
        
        start_index = response_text.find('{')
        end_index = response_text.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_text_untuk_debug = response_text[start_index : end_index + 1]
            try:
                return json.loads(json_text_untuk_debug)
            except json.JSONDecodeError:
                try:
                    repaired_json_text = repair_json(json_text_untuk_debug)
                    return json.loads(repaired_json_text)
                except Exception as e_repair:
                    return {"error": f"Gagal memperbaiki dan mengonversi JSON: {e_repair}", "raw_text": json_text_untuk_debug}
        else:
            return {"error": "Tidak ada blok JSON valid di respons LLM.", "raw_response": response_text}

    except Exception as e:
        # Kita juga hapus print dari sini
        return {"error": f"Kesalahan tidak terduga: {e}", "raw_response": json_text_untuk_debug}