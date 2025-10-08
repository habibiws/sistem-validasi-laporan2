# backend/konteks_extractor.py
import torch
from PIL import Image
import pytesseract
import json
from json_repair import repair_json
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

MODEL_MATA, PROCESSOR_MATA = None, None
MODEL_OTAK, TOKENIZER_OTAK = None, None

def load_models():
    global MODEL_MATA, PROCESSOR_MATA, MODEL_OTAK, TOKENIZER_OTAK
    
    # Deteksi device (GPU jika tersedia di Codespaces berbayar, atau CPU)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Menggunakan device: {DEVICE} ---")

    if MODEL_MATA is None:
        print("Memuat model 'Mata' (LayoutLMv3)...")
        MODEL_ID_LM = "habibiws/sistem-validasi-laporan2-models"
        SUBFOLDER_LM = "layoutlmv3-finetuned-laporan-100%209-data-100e-koreksi"
        PROCESSOR_MATA = LayoutLMv3Processor.from_pretrained(MODEL_ID_LM, subfolder=SUBFOLDER_LM)
        MODEL_MATA = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID_LM, subfolder=SUBFOLDER_LM).to(DEVICE)
        print(f"Model 'Mata' berhasil dimuat ke {DEVICE}.")
    
    if MODEL_OTAK is None:
        print("Memuat model 'Otak' (FLAN-T5)...")
        MODEL_ID_T5 = "habibiws/sistem-validasi-laporan2-models"
        SUBFOLDER_T5 = "flan-t5-finetuned-penataan"
        TOKENIZER_OTAK = AutoTokenizer.from_pretrained(MODEL_ID_T5, subfolder=SUBFOLDER_T5)
        MODEL_OTAK = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID_T5, subfolder=SUBFOLDER_T5).to(DEVICE)
        print(f"Model 'Otak' berhasil dimuat ke {DEVICE}.")

def get_models():
    if MODEL_MATA is None or MODEL_OTAK is None:
        load_models()
    return (MODEL_MATA, PROCESSOR_MATA), (MODEL_OTAK, TOKENIZER_OTAK)

# ... (Salin semua fungsi helper dari `app.py` terakhir kita ke sini)
# yaitu: analisis_halaman_dengan_layoutlmv3, _gabungkan_token_menjadi_entitas, tata_ulang_dengan_flan_t5

def analisis_halaman_dengan_layoutlmv3(image: Image.Image) -> list:
    """Menggunakan Manual OCR dan LayoutLMv3 untuk mengekstrak token dan label."""
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words, boxes = [], []
        for i in range(len(ocr_data["text"])):
            if int(ocr_data["conf"][i]) > 0 and ocr_data["text"][i].strip():
                words.append(ocr_data["text"][i])
                (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
                img_width, img_height = image.size
                boxes.append([int(x / img_width * 1000), int(y / img_height * 1000), int((x + w) / img_width * 1000), int((y + h) / img_height * 1000)])
    except Exception as e:
        print(f"OCR Gagal: {e}")
        return []

    if not words:
        return []

    # Persiapkan input untuk model
    encoding = PROCESSOR_MATA.tokenizer(text=words, boxes=boxes, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    pixel_values = PROCESSOR_MATA.image_processor(image, return_tensors="pt").pixel_values
    encoding["pixel_values"] = pixel_values

    # Pindahkan data input (tensor) ke device yang sama dengan model
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = MODEL_MATA(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = PROCESSOR_MATA.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
    token_boxes = encoding["bbox"].squeeze().tolist()
    
    # Handle kasus ketika output hanya satu nilai, bukan list
    if not isinstance(predictions, list):
        predictions = [predictions]
    if token_boxes and not isinstance(token_boxes[0], list):
        token_boxes = [token_boxes]

    final_tokens = []
    for token, box, pred_id in zip(tokens, token_boxes, predictions):
        if token in [PROCESSOR_MATA.tokenizer.cls_token, PROCESSOR_MATA.tokenizer.sep_token, PROCESSOR_MATA.tokenizer.pad_token] or not box:
            continue
        final_tokens.append({"token": token, "label": MODEL_MATA.config.id2label[pred_id], "box": [int(coord) for coord in box]})

    return final_tokens

def _gabungkan_token_menjadi_entitas(hasil_analisis_kontekstual: list) -> list:
    """Menggabungkan token yang berdekatan dengan label yang sama menjadi entitas tunggal."""
    if not hasil_analisis_kontekstual:
        return []
    
    entitas_dict = {}
    for token_data in hasil_analisis_kontekstual:
        key = (tuple(token_data["box"]), token_data["label"])
        if key not in entitas_dict:
            entitas_dict[key] = []
        entitas_dict[key].append(token_data["token"])
        
    entitas_final = []
    for (box, label), tokens in entitas_dict.items():
        teks_lengkap = "".join(tokens).replace("Ġ", " ").strip()
        if teks_lengkap:
            entitas_final.append({"text": teks_lengkap, "box": list(box), "label": label})
            
    # Urutkan entitas berdasarkan posisi Y dan X
    entitas_final.sort(key=lambda e: (e["box"][1], e["box"][0]))
    return entitas_final

def tata_ulang_dengan_flan_t5(final_entities: list) -> dict:
    """Menggunakan FLAN-T5 untuk menata ulang entitas menjadi struktur JSON."""
    if not final_entities:
        return {"error": "Tidak ada entitas untuk diproses."}

    prefix = "Translate from Indonesian to JSON: "
    input_lines = [f"teks: \"{entity['text']}\" box: {entity['box']}" for entity in final_entities]
    input_text = prefix + "\n".join(input_lines)
    
    inputs = TOKENIZER_OTAK(input_text, max_length=512, truncation=True, return_tensors="pt")
    
    # Pindahkan input_ids ke device yang sama dengan model
    input_ids = inputs.input_ids.to(DEVICE)
    
    output_ids = MODEL_OTAK.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    predicted_json_string = TOKENIZER_OTAK.decode(output_ids[0], skip_special_tokens=True)
    
    # Perbaiki format JSON yang mungkin tidak sempurna
    potential_json = f"{{{predicted_json_string}}}"
    try:
        return json.loads(potential_json)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json(potential_json))
        except Exception:
            return {"error": "Gagal menghasilkan JSON valid.", "raw_output": predicted_json_string}
        


# backend/konteks_extractor.py (Versi FINAL)

import torch
import pytesseract
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --- PERBAIKAN FINAL: Atur Path Tesseract Secara Eksplisit ---
# Salin blok kode yang sama dari skrip test_ocr.py
try:
    # GANTI PATH INI jika lokasi instalasi Tesseract Anda berbeda
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print("[INFO] Path Tesseract diatur secara manual untuk aplikasi.")
except Exception:
    print("[INFO] Gagal mengatur path Tesseract (mungkin bukan di Windows).")
    pass
# -----------------------------------------------------------------

# Impor yang dibutuhkan untuk kedua model
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    BertTokenizer,
    EncoderDecoderModel,
    BertConfig,
    EncoderDecoderConfig
)

# --- Variabel Global untuk Model ---
MODEL = None
PROCESSOR = None
BRAIN_MODEL = None
BRAIN_TOKENIZER = None

# --- Fungsi Pemuatan (Loaders) ---

def load_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        # Ganti dengan nama repo model Anda di Hub
        MODEL_ID = "habibiws/sistem-validasi-laporan2-models"
        # Tentukan subfolder tempat model spesifik ini berada
        MODEL_SUBFOLDER = "layoutlmv3-finetuned-laporan-100%209-data-100e-koreksi"

        print(f"Mengunduh/memuat model 'Mata' dari Hub: {MODEL_ID}/{MODEL_SUBFOLDER}...")
        PROCESSOR = LayoutLMv3Processor.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        print("Model 'Mata' AI berhasil dimuat.")

def load_brain_model():
    global BRAIN_MODEL, BRAIN_TOKENIZER
    if BRAIN_MODEL is None:
        MODEL_ID = "habibiws/sistem-validasi-laporan2-models"
        MODEL_SUBFOLDER = "indobert-finetuned-penataan"

        print(f"Mengunduh/memuat model 'Otak' dari Hub: {MODEL_ID}/{MODEL_SUBFOLDER}...")
        BRAIN_TOKENIZER = BertTokenizer.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        BRAIN_MODEL = EncoderDecoderModel.from_pretrained(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        print("Model 'Otak' AI berhasil dimuat.")

def get_layoutlm_model_and_processor():
    """Getter yang aman untuk model LayoutLM."""
    if MODEL is None or PROCESSOR is None:
        load_model()
    return MODEL, PROCESSOR

def get_indobert_model_and_tokenizer():
    """Getter yang aman untuk model IndoBERT."""
    if BRAIN_MODEL is None or BRAIN_TOKENIZER is None:
        load_brain_model()
    return BRAIN_MODEL, BRAIN_TOKENIZER

# --- Fungsi Utilitas & Analisis (Tidak Berubah) ---

# Di dalam backend/konteks_extractor.py

# Di dalam backend/konteks_extractor.py

# Di dalam backend/konteks_extractor.py

# Di dalam backend/konteks_extractor.py

def analisis_halaman_dengan_layoutlmv3(image: Image.Image) -> dict:
    """
    Menganalisis gambar halaman menggunakan LayoutLMv3.
    Versi "Manual OCR" untuk bypass masalah processor.
    """
    model, processor = get_layoutlm_model_and_processor()
    
    print("   - Langkah 1/2: Menjalankan OCR manual dengan Pytesseract...")
    
    # --- PERBAIKAN FINAL: LAKUKAN OCR SECARA MANUAL ---
    # Kita panggil pytesseract langsung, yang sudah terbukti bekerja.
    # 'image_to_data' memberikan kita teks beserta koordinat bounding box-nya.
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        words = []
        boxes = []
        # Proses output dari pytesseract untuk mendapatkan 'words' dan 'boxes'
        for i in range(len(ocr_data["text"])):
            # Hanya ambil kata yang memiliki confidence score dan bukan string kosong
            if int(ocr_data["conf"][i]) > 0 and ocr_data["text"][i].strip():
                words.append(ocr_data["text"][i])
                
                # Konversi format box dari (left, top, width, height) ke (x1, y1, x2, y2)
                # dan normalisasi ke skala 1000
                (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
                img_width, img_height = image.size
                
                x1 = int(x / img_width * 1000)
                y1 = int(y / img_height * 1000)
                x2 = int((x + w) / img_width * 1000)
                y2 = int((y + h) / img_height * 1000)
                boxes.append([x1, y1, x2, y2])

    except Exception as e:
        print(f"[ERROR FATAL] Pytesseract manual gagal. Error: {e}")
        return {"hasil_analisis_kontekstual": []}

    if not words:
        print("   - Peringatan: OCR manual tidak menemukan teks apa pun di halaman ini.")
        return {"hasil_analisis_kontekstual": []}
    # ----------------------------------------------------

    print(f"   - Langkah 2/2: Menjalankan tokenisasi dan prediksi label...")
    # Sekarang kita berikan hasil OCR manual kita ke tokenizer
    encoding = processor.tokenizer(
        text=words,
        boxes=boxes,
        truncation=True,        # <-- TAMBAHKAN INI
        padding="max_length",   # <-- TAMBAHKAN INI JUGA UNTUK KONSISTENSI
        max_length=512,         # <-- DAN TENTUKAN BATASNYA
        return_tensors="pt"
    )
    
    pixel_values = processor.image_processor(image, return_tensors="pt").pixel_values
    encoding["pixel_values"] = pixel_values
    
    device = model.device
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    predictions = logits.argmax(-1).squeeze().tolist()
    
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
    token_boxes = encoding["bbox"].squeeze().tolist()

    if not isinstance(predictions, list): predictions = [predictions]
    if token_boxes and not isinstance(token_boxes[0], list): token_boxes = [token_boxes]

    final_tokens = []
    for token, box, pred_id in zip(tokens, token_boxes, predictions):
        if token in [processor.tokenizer.cls_token, processor.tokenizer.sep_token, processor.tokenizer.pad_token] or not box:
            continue
        
        final_tokens.append({
            "token": token,
            "label": model.config.id2label[pred_id],
            "box": [int(coord) for coord in box]
        })

    print(f"   - Ekstraksi selesai, {len(final_tokens)} token ditemukan.")
    
    return {"hasil_analisis_kontekstual": final_tokens}

def visualisasikan_hasil_analisis(image: Image.Image, hasil_analisis: dict) -> Image.Image:
    # ... (fungsi ini tidak perlu diubah)
    pass # Placeholder, asumsikan kode lengkap sudah ada di file Anda