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
    """Memuat model LayoutLMv3 ('Mata') yang sudah di-fine-tune."""
    global MODEL, PROCESSOR
    if MODEL is None:
        MODEL_NAME = "layoutlmv3-finetuned-laporan-100%209-data-100e-koreksi" # Sesuaikan jika perlu
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
        
        print(f"Memuat model 'Mata' AI dari '{MODEL_PATH}'...")
        PROCESSOR = LayoutLMv3Processor.from_pretrained(MODEL_PATH)
        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
        print("Model 'Mata' AI berhasil dimuat.")

def load_brain_model():
    """
    Memuat model IndoBERT ('Otak') dengan injeksi konfigurasi manual
    untuk memastikan parameter decoder sudah benar. INI VERSI PALING STABIL.
    """
    global BRAIN_MODEL, BRAIN_TOKENIZER
    if BRAIN_MODEL is None:
        MODEL_NAME = "indobert-finetuned-penataan"
        BASE_MODEL_FOR_CONFIG = "indobenchmark/indobert-base-p1"
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
        
        print(f"Memuat model 'Otak' AI dari '{MODEL_PATH}' (Mode Injeksi Konfigurasi)...")

        # 1. Muat tokenizer.
        BRAIN_TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH)
        
        # 2. Buat konfigurasi encoder dan decoder secara manual.
        encoder_config = BertConfig.from_pretrained(BASE_MODEL_FOR_CONFIG)
        decoder_config = BertConfig.from_pretrained(BASE_MODEL_FOR_CONFIG)
        
        # 3. Buat konfigurasi gabungan dan paksa parameter yang benar.
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        config.decoder_start_token_id = BRAIN_TOKENIZER.cls_token_id
        config.eos_token_id = BRAIN_TOKENIZER.sep_token_id
        config.pad_token_id = BRAIN_TOKENIZER.pad_token_id
        config.bos_token_id = BRAIN_TOKENIZER.cls_token_id
        config.decoder.is_decoder = True
        config.decoder.add_cross_attention = True

        # 4. Muat 'weights' dari file lokal dengan konfigurasi paksa yang baru.
        BRAIN_MODEL = EncoderDecoderModel.from_pretrained(MODEL_PATH, config=config)
        
        print("Model 'Otak' AI berhasil dimuat.")

# --- Fungsi Akses Aman (Getters) ---

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