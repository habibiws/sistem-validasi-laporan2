# Gunakan gambar dasar resmi dari Python
FROM python:3.11-slim

# Instal Tesseract OCR di dalam kontainer Linux
RUN apt-get update && apt-get install -y tesseract-ocr

# Atur direktori kerja di dalam kontainer
WORKDIR /app

# --- OPTIMISASI UTAMA DI SINI ---
# 1. Salin HANYA file requirements.txt terlebih dahulu
COPY requirements.txt requirements.txt

# 2. Jalankan instalasi. Langkah ini akan disimpan di cache.
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# 3. BARU setelah itu, salin sisa kode aplikasi Anda.
COPY . .
# ---------------------------------

# Beritahu Docker port mana yang akan diekspos
EXPOSE 8000

# Perintah untuk menjalankan server saat kontainer dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]