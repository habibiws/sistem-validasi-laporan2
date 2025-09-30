# backend/konteks_extractor.py
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, BertTokenizer, EncoderDecoderModel
import torch

MODEL = None
PROCESSOR = None
BRAIN_MODEL = None
BRAIN_TOKENIZER = None

def load_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        MODEL_NAME = "models/layoutlmv3-finetuned-laporan-100%209-data-100e-koreksi"
        print(f"Memuat model AI '{MODEL_NAME}' dari folder lokal...")
        PROCESSOR = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=True)
        MODEL = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME)
        print("Model AI berhasil dimuat dan siap digunakan.")

def load_brain_model():
    """Memuat model IndoBERT (Otak) yang sudah di-fine-tune saat startup."""
    global BRAIN_MODEL, BRAIN_TOKENIZER
    if BRAIN_MODEL is None:
        # --- PASTIKAN NAMA FOLDER INI SESUAI DENGAN NAMA FOLDER MODEL ANDA ---
        MODEL_NAME = "indobert-finetuned-penataan" 
        
        # Menggunakan path dinamis yang sudah kita buat sebelumnya
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
        
        print(f"Memuat model 'Otak' AI dari '{MODEL_PATH}'...")
        BRAIN_TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH)
        BRAIN_MODEL = EncoderDecoderModel.from_pretrained(MODEL_PATH)
        print("Model 'Otak' AI berhasil dimuat dan siap digunakan.")

def analisis_halaman_dengan_layoutlmv3(image: Image.Image) -> dict:
    global MODEL, PROCESSOR
    if MODEL is None or PROCESSOR is None: raise RuntimeError("Model belum dimuat.")

    print("   - Menerapkan strategi 'Sliding Window' untuk ekstraksi komprehensif...")
    
    encoding = PROCESSOR(
        image,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_overflowing_tokens=True,
        stride=128,
        return_tensors="pt"
    )
    
    encoding.pop('overflow_to_sample_mapping', None)
    
    all_tokens = []
    
    # --- PERBAIKAN UTAMA ADA DI SINI ---
    # Ekstrak tensor pixel_values dari dalam list
    pixel_values = encoding.pixel_values[0] 
    num_batches = len(encoding.input_ids)
    
    with torch.no_grad():
        for i in range(num_batches):
            # Gunakan pixel_values yang sama untuk setiap batch teks
            batch_input = {
                "input_ids": encoding.input_ids[i].unsqueeze(0),
                "attention_mask": encoding.attention_mask[i].unsqueeze(0),
                "bbox": encoding.bbox[i].unsqueeze(0),
                "pixel_values": pixel_values.unsqueeze(0), # <-- Pastikan ada batch dimension
            }
            
            outputs = MODEL(**batch_input)

            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            token_ids = batch_input["input_ids"].squeeze().tolist()
            boxes = batch_input["bbox"].squeeze().tolist()
            tokens_text = PROCESSOR.tokenizer.convert_ids_to_tokens(token_ids)

            for token, box, pred_id in zip(tokens_text, boxes, predictions):
                if token in [PROCESSOR.tokenizer.cls_token, PROCESSOR.tokenizer.sep_token, PROCESSOR.tokenizer.pad_token]:
                    continue
                all_tokens.append({
                    "token": token,
                    "label": MODEL.config.id2label[pred_id],
                    "box": [int(coord) for coord in box]
                })

    # Hapus duplikat yang mungkin muncul di area tumpang tindih
    unique_tokens = []
    seen_tokens = set()
    for token in all_tokens:
        token_id = (token['token'], tuple(token['box']))
        if token_id not in seen_tokens:
            unique_tokens.append(token)
            seen_tokens.add(token_id)
    
    print(f"   - Ekstraksi selesai, total {len(unique_tokens)} token unik berhasil diekstrak.")
    return {"hasil_analisis_kontekstual": unique_tokens}

# Fungsi visualisasi tidak diubah
def visualisasikan_hasil_analisis(image: Image.Image, hasil_analisis: dict) -> Image.Image:
    img_visual = image.copy(); draw = ImageDraw.Draw(img_visual); width, height = img_visual.size
    def unnormalize_box(box, width, height): return [int(box[0]/1000*width), int(box[1]/1000*height), int(box[2]/1000*width), int(box[3]/1000*height)]
    label_colors = {"KEY":"blue", "VALUE":"green", "OTHER":"red", "DEFAULT":"red"}
    try: font = ImageFont.truetype("arial.ttf", 15)
    except IOError: font = ImageFont.load_default()
    data_analisis = hasil_analisis.get("hasil_analisis_kontekstual", [])
    for item in data_analisis:
        pixel_box = unnormalize_box(item["box"], width, height); label = item["label"]; color = label_colors.get(str(label), label_colors["DEFAULT"])
        draw.rectangle(pixel_box, outline=color, width=2)
        text_position = (pixel_box[0]+5, pixel_box[1]-15)
        if text_position[1] < 0: text_position = (pixel_box[0]+5, pixel_box[1]+5)
        draw.text(text_position, str(label), fill=color, font=font)
    return img_visual